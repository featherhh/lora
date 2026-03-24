import os
import sys
import argparse
import json
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    set_seed,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from tqdm import tqdm
from datetime import datetime
import shutil
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed as accelerate_set_seed

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

class LoraFinetuner:
    def __init__(self, config):
        self.config = config
        self.accelerator = self._init_accelerator()
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.optimizer = None
        self.scheduler = None
        self.start_epoch = 0
        
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        # 保存配置文件
        with open(os.path.join(self.config.output_dir, "config.json"), "w") as f:
            json.dump(vars(config), f, indent=2)

    def _init_accelerator(self):
        """初始化Accelerator用于分布式训练和混合精度"""
        project_config = ProjectConfiguration(
            project_dir=self.config.output_dir,
            logging_dir=os.path.join(self.config.output_dir, "logs")
        )
        
        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            project_config=project_config,
            log_with=self.config.logging_backend,
            logging_dir=os.path.join(self.config.output_dir, "logs")
        )
        
        # 设置随机种子确保可复现性
        accelerate_set_seed(self.config.seed)
        
        return accelerator

    def load_tokenizer(self):
        """加载预训练分词器"""
        logger.info(f"Loading tokenizer from {self.config.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=self.config.use_fast_tokenizer
        )
        
        # 设置pad_token（如果未设置）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        return self.tokenizer

    def load_model(self):
        """加载预训练模型并配置LoRA"""
        logger.info(f"Loading model from {self.config.model_name_or_path}")
        
        # 量化配置（如果启用）
        quantization_config = None
        if self.config.quantization_bits in [4, 8]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.quantization_bits == 4,
                load_in_8bit=self.config.quantization_bits == 8,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=torch.float16 if self.config.bnb_4bit_compute_dtype == "float16" else torch.bfloat16
            )
        
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            quantization_config=quantization_config,
            device_map={"": self.accelerator.process_index} if not self.accelerator.distributed_type == "NO" else None,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch.float16 if self.config.torch_dtype == "float16" else torch.bfloat16 if self.config.torch_dtype == "bfloat16" else torch.float32
        )
        
        # 为量化训练准备模型
        if self.config.quantization_bits in [4, 8]:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self.config.gradient_checkpointing
            )
        
        # 配置梯度检查点以节省内存
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False  # 梯度检查点需要禁用缓存
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules.split(','),
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            task_type="CAUSAL_LM",
            inference_mode=False,
        )
        
        # 应用LoRA适配器
        model = get_peft_model(model, lora_config)
        
        # 打印可训练参数数量
        model.print_trainable_parameters()
        
        self.model = model
        return self.model

    def load_and_preprocess_data(self):
        """加载并预处理数据集"""
        logger.info(f"Loading dataset from {self.config.dataset_path}")
        
        # 加载数据集
        if self.config.dataset_format == "json":
            dataset = load_dataset(
                "json", 
                data_files=self.config.dataset_path,
                split="train"
            )
        elif self.config.dataset_format == "csv":
            dataset = load_dataset(
                "csv", 
                data_files=self.config.dataset_path,
                split="train"
            )
        else:
            raise ValueError(f"Unsupported dataset format: {self.config.dataset_format}")
        
        # 分割训练集和验证集
        if self.config.validation_split_ratio > 0:
            dataset = dataset.train_test_split(test_size=self.config.validation_split_ratio)
        else:
            dataset = DatasetDict({
                "train": dataset,
                "validation": dataset.select(range(min(100, len(dataset))))  # 用小部分数据作为验证集
            })
        
        # 预处理函数
        def preprocess_function(examples):
            # 合并文本字段（根据实际数据集调整）
            if self.config.text_field in examples:
                texts = examples[self.config.text_field]
            else:
                # 如果指定的字段不存在，尝试合并所有文本字段
                texts = []
                for i in range(len(examples[next(iter(examples))])):
                    text_parts = []
                    for key in examples:
                        text_parts.append(f"{key}: {examples[key][i]}")
                    texts.append("\n".join(text_parts))
            
            # 分词
            return self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
                return_tensors="pt"
            )
        
        # 应用预处理
        with self.accelerator.main_process_first():
            dataset = dataset.map(
                preprocess_function,
                batched=True,
                num_proc=self.config.preprocessing_num_workers,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=not self.config.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        
        self.dataset = dataset
        return self.dataset

    def create_dataloaders(self):
        """创建训练和验证数据加载器"""
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # 因果语言模型不使用掩码语言建模
        )
        
        # 训练集加载器
        train_dataset = self.dataset["train"]
        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            seed=self.config.seed
        ) if self.accelerator.distributed_type != "NO" else None
        
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            batch_size=self.config.per_device_train_batch_size,
            collate_fn=data_collator,
            drop_last=True,
        )
        
        # 验证集加载器
        eval_dataset = self.dataset["validation"]
        eval_sampler = DistributedSampler(
            eval_dataset,
            shuffle=False
        ) if self.accelerator.distributed_type != "NO" else None
        
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.config.per_device_eval_batch_size,
            collate_fn=data_collator,
            drop_last=False,
        )
        
        return train_dataloader, eval_dataloader

    def configure_optimizers(self, train_dataloader):
        """配置优化器和学习率调度器"""
        # 准备模型参数
        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        
        # 优化器
        self.optimizer = AdamW(
            params,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay
        )
        
        # 学习率调度器
        total_training_steps = len(train_dataloader) * self.config.num_train_epochs // self.config.gradient_accumulation_steps
        warmup_steps = int(total_training_steps * self.config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )
        
        return self.optimizer, self.scheduler

    def load_checkpoint(self):
        """加载检查点以继续训练"""
        if not self.config.resume_from_checkpoint:
            return
        
        checkpoint_path = self.config.resume_from_checkpoint
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(
            os.path.join(checkpoint_path, "pytorch_model.bin"),
            map_location={"": self.accelerator.process_index}
        )
        
        # 加载模型状态
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # 加载优化器和调度器状态
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # 加载训练进度
        self.start_epoch = checkpoint["epoch"] + 1
        
        logger.info(f"Resumed training from epoch {self.start_epoch}")

    def save_checkpoint(self, epoch, train_loss, eval_loss):
        """保存训练检查点"""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型状态
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "config": vars(self.config)
        }
        
        torch.save(
            checkpoint,
            os.path.join(checkpoint_dir, "pytorch_model.bin")
        )
        
        # 保存LoRA配置
        self.model.save_pretrained(checkpoint_dir)
        
        # 保存分词器
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # 创建最新检查点的软链接
        latest_link = os.path.join(self.config.output_dir, "checkpoint-latest")
        if os.path.exists(latest_link):
            if os.path.islink(latest_link):
                os.unlink(latest_link)
            else:
                shutil.rmtree(latest_link)
        os.symlink(checkpoint_dir, latest_link)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # 清理旧检查点（如果启用）
        if self.config.max_checkpoints > 0:
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """清理旧的检查点以节省空间"""
        checkpoints = []
        for entry in os.scandir(self.config.output_dir):
            if entry.is_dir() and entry.name.startswith("checkpoint-epoch-"):
                try:
                    epoch = int(entry.name.split("-")[-1])
                    checkpoints.append((epoch, entry.path))
                except ValueError:
                    continue
        
        # 按 epoch 排序并保留最新的 N 个
        checkpoints.sort(reverse=True, key=lambda x: x[0])
        if len(checkpoints) > self.config.max_checkpoints:
            for epoch, path in checkpoints[self.config.max_checkpoints:]:
                logger.info(f"Removing old checkpoint: {path}")
                shutil.rmtree(path)

    def evaluate(self, eval_dataloader):
        """评估模型性能"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not self.accelerator.is_local_main_process):
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # 聚合所有进程的损失
                loss = self.accelerator.gather(loss.repeat(self.config.per_device_eval_batch_size))
                total_loss += loss.sum().item()
                num_batches += 1
        
        avg_loss = total_loss / (num_batches * self.config.per_device_eval_batch_size * self.accelerator.num_processes)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        self.model.train()
        return avg_loss, perplexity

    def train(self):
        """主训练循环"""
        # 加载组件
        self.load_tokenizer()
        self.load_model()
        self.load_and_preprocess_data()
        train_dataloader, eval_dataloader = self.create_dataloaders()
        optimizer, scheduler = self.configure_optimizers(train_dataloader)
        
        # 准备加速
        model, optimizer, train_dataloader, eval_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, eval_dataloader, scheduler
        )
        
        # 加载检查点
        self.load_checkpoint()
        
        # 训练日志
        if self.accelerator.is_main_process:
            run_name = f"{self.config.model_name_or_path.split('/')[-1]}-lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            logger.info(f"Starting training run: {run_name}")
            logger.info(f"Training epochs: {self.start_epoch} to {self.config.num_train_epochs-1}")
            logger.info(f"Total training steps: {len(train_dataloader) * self.config.num_train_epochs // self.config.gradient_accumulation_steps}")
        
        # 开始训练
        best_eval_loss = float("inf")
        for epoch in range(self.start_epoch, self.config.num_train_epochs):
            if self.accelerator.distributed_type != "NO":
                train_dataloader.sampler.set_epoch(epoch)
            
            model.train()
            total_train_loss = 0.0
            progress_bar = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Epoch {epoch}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for step, batch in progress_bar:
                with self.accelerator.accumulate(model):
                    batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                    outputs = model(** batch)
                    loss = outputs.loss
                    
                    # 反向传播
                    self.accelerator.backward(loss)
                    
                    # 梯度裁剪
                    if self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    
                    # 更新参数
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # 记录损失
                total_train_loss += loss.detach().item()
                progress_bar.set_postfix({"loss": loss.detach().item()})
                
                # 定期评估
                if (step + 1) % self.config.eval_steps == 0 and step != 0:
                    eval_loss, perplexity = self.evaluate(eval_dataloader)
                    logger.info(
                        f"Epoch {epoch}, Step {step+1}: "
                        f"Train loss: {total_train_loss/(step+1):.4f}, "
                        f"Eval loss: {eval_loss:.4f}, "
                        f"Perplexity: {perplexity:.4f}"
                    )
                    
                    # 记录到日志系统
                    if self.accelerator.is_main_process and self.config.logging_backend:
                        self.accelerator.log({
                            "train_loss": total_train_loss/(step+1),
                            "eval_loss": eval_loss,
                            "perplexity": perplexity,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "step": step
                        }, step=epoch * len(train_dataloader) + step)
            
            #  epoch结束时评估
            avg_train_loss = total_train_loss / len(train_dataloader)
            eval_loss, perplexity = self.evaluate(eval_dataloader)
            
            logger.info(
                f"Epoch {epoch} complete: "
                f"Avg train loss: {avg_train_loss:.4f}, "
                f"Eval loss: {eval_loss:.4f}, "
                f"Perplexity: {perplexity:.4f}"
            )
            
            # 记录到日志系统
            if self.accelerator.is_main_process and self.config.logging_backend:
                self.accelerator.log({
                    "avg_train_loss": avg_train_loss,
                    "eval_loss": eval_loss,
                    "perplexity": perplexity,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epoch": epoch
                }, step=(epoch + 1) * len(train_dataloader))
            
            # 保存检查点
            if self.accelerator.is_main_process:
                self.save_checkpoint(epoch, avg_train_loss, eval_loss)
                
                # 保存最佳模型
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_model_dir = os.path.join(self.config.output_dir, "best_model")
                    self.model.save_pretrained(best_model_dir)
                    self.tokenizer.save_pretrained(best_model_dir)
                    logger.info(f"Saved best model to {best_model_dir}")
        
        # 训练结束，保存最终模型
        if self.accelerator.is_main_process:
            final_model_dir = os.path.join(self.config.output_dir, "final_model")
            self.model.save_pretrained(final_model_dir)
            self.tokenizer.save_pretrained(final_model_dir)
            logger.info(f"Training completed. Final model saved to {final_model_dir}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LoRA Finetuning for Large Language Models")
    
    # 模型和数据配置
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--dataset_format", type=str, default="json", choices=["json", "csv"], help="Format of the dataset file")
    parser.add_argument("--text_field", type=str, default="text", help="Name of the field containing text data in the dataset")
    parser.add_argument("--output_dir", type=str, default="./lora_results", help="Directory to save training results")
    
    # LoRA配置
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank of the LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha parameter for LoRA scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers")
    parser.add_argument("--lora_target_modules", type=str, required=True, help="Comma-separated list of target modules for LoRA")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"], help="Bias type for LoRA")
    
    # 训练配置
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per GPU/CPU for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size per GPU/CPU for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate (after the potential warmup period)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to apply to all layers except bias/LayerNorm weights")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Ratio of total training steps used for warmup")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 parameter for Adam optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 parameter for Adam optimizer")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon parameter for Adam optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm (for gradient clipping)")
    
    # 数据处理配置
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--validation_split_ratio", type=float, default=0.1, help="Ratio of data to use for validation")
    parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="Number of workers to use for preprocessing")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached preprocessed datasets")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Use fast tokenizer if available")
    
    # 量化配置
    parser.add_argument("--quantization_bits", type=int, default=0, choices=[0, 4, 8], help="Number of bits for quantization (0 = no quantization)")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true", help="Use double quantization for 4-bit")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["fp4", "nf4"], help="Quantization type for 4-bit")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16", choices=["float16", "bfloat16"], help="Compute dtype for 4-bit quantization")
    
    # 其他配置
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code when loading model")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"], help="Torch dtype for model weights")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision training type")
    parser.add_argument("--eval_steps", type=int, default=100, help="Number of steps between evaluations")
    parser.add_argument("--logging_backend", type=str, default="tensorboard", choices=["tensorboard", "wandb", None], help="Logging backend to use")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--max_checkpoints", type=int, default=3, help="Maximum number of checkpoints to keep (0 = keep all)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置日志级别
    logger.setLevel(logging.INFO if dist.is_initialized() and dist.get_rank() == 0 else logging.WARN)
    
    # 初始化微调器并开始训练
    finetuner = LoraFinetuner(args)
    finetuner.train()

if __name__ == "__main__":
    main()
