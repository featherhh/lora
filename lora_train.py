import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import warnings

warnings.filterwarnings("ignore")

# ====================== 1. 基础配置（可根据需求修改） ======================
# 模型&数据集配置
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # 替换为你的模型路径/名称
DATASET_NAME = "tatsu-lab/alpaca"  # 替换为你的数据集路径（本地/开源）
OUTPUT_DIR = "./lora_finetuned_model"  # 微调后模型保存路径
# LoRA核心参数
LORA_R = 8  # LoRA秩（越小显存占用越少，一般8/16/32）
LORA_ALPHA = 16  # 缩放系数（通常设为2*LORA_R）
LORA_DROPOUT = 0.05  # Dropout概率
TARGET_MODULES = ["q_proj", "v_proj"]  # 目标模块（不同模型需调整，如Baichuan是W_pack）
# 训练超参数
BATCH_SIZE = 4  # 批次大小（根据显存调整）
LEARNING_RATE = 2e-4  # 学习率（LoRA微调建议1e-4~5e-4）
NUM_TRAIN_EPOCHS = 3  # 训练轮数
MAX_STEPS = -1  # 若设为正数，会覆盖epoch数
GRADIENT_ACCUMULATION_STEPS = 2  # 梯度累积（显存不足时增大）


# ====================== 2. 加载并预处理数据集 ======================
def format_dataset(example):
    """格式化数据：将指令+输入+输出拼接为模型可识别的格式"""
    instruction = example["instruction"]
    input_text = example["input"]
    output_text = example["output"]

    # 拼接模板（可根据模型调整）
    if input_text.strip() == "":
        prompt = f"### Instruction:\n{instruction}\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Response:\n"
    # 最终文本：prompt + 回答 + 结束符
    example["text"] = prompt + output_text + "</s>"
    return example


# 加载数据集并格式化
dataset = load_dataset(DATASET_NAME)
dataset = dataset.map(format_dataset, remove_columns=["instruction", "input", "output"])
# 划分训练集和验证集（8:2）
dataset = dataset["train"].train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ====================== 3. 加载模型&Tokenizer ======================
# 4bit量化配置（节省显存，若无量化需求可删除）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4bit加载模型
    bnb_4bit_use_double_quant=True,  # 双量化
    bnb_4bit_quant_type="nf4",  # 量化类型
    bnb_4bit_compute_dtype=torch.bfloat16  # 计算精度
)

# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad token（LLaMA默认无pad token）
tokenizer.padding_side = "right"  # 右填充（避免影响注意力）

# 加载模型（带4bit量化）
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",  # 自动分配模型到GPU/CPU
    trust_remote_code=True  # 加载自定义模型时需要
)
# 为4bit训练做准备
model = prepare_model_for_kbit_training(model)

# ====================== 4. 配置LoRA适配器 ======================
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",  # 不训练bias参数
    task_type="CAUSAL_LM"  # 任务类型：因果语言模型
)
# 将LoRA适配器挂载到主模型上
model = get_peft_model(model, lora_config)
# 打印可训练参数（LoRA仅训练少量参数，约占原模型的0.1%）
model.print_trainable_parameters()


# ====================== 5. 数据编码&数据整理器 ======================
def tokenize_function(examples):
    """对数据集进行编码"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,  # 最大序列长度（根据显存调整）
        padding="max_length"
    )


# 编码训练集和验证集
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# 数据整理器（处理批次数据，自动生成标签）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 因果语言模型不需要掩码语言建模
)

# ====================== 6. 配置训练参数 ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    max_steps=MAX_STEPS,
    logging_steps=10,  # 每10步打印一次日志
    evaluation_strategy="epoch",  # 每个epoch验证一次
    save_strategy="epoch",  # 每个epoch保存一次模型
    fp16=True,  # 混合精度训练（GPU支持的话开启）
    optim="paged_adamw_8bit",  # 优化器（节省显存）
    report_to="none",  # 不使用wandb等日志工具
    remove_unused_columns=False,
    load_best_model_at_end=True,  # 训练结束后加载最优模型
)

# ====================== 7. 开始训练 ======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# 启动训练
trainer.train()

# 保存微调后的LoRA适配器（仅保存增量参数，体积很小，约几十MB）
model.save_pretrained(OUTPUT_DIR)
print(f"LoRA微调完成，模型保存至：{OUTPUT_DIR}")