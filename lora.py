import os

import pandas as pd
import torch
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback  # 新增：导入早停回调
)
from peft import LoraConfig, get_peft_model, TaskType

# ====================== 核心配置（路径已经适配好，不用改） ======================
# 你的模型文件夹路径（就是刚下载好的qwen2.5文件夹）
MODEL_PATH = "./qwen2.5"
# 微调后LoRA权重的保存路径
OUTPUT_DIR = "./qwen2.5-lora"
# 训练数据的保存路径
DATA_PATH = "data/data.json"

# ====================== 显卡显存配置（根据你的显卡调整） ======================
# 4-bit量化配置：16GB显存就能流畅跑7B模型，显存不够就用这个
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

# ====================== 1. 加载模型和分词器 ======================
print("正在加载本地模型和分词器...")
# 加载模型（自动识别你的显卡，分配显存）
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

# 加载分词器（适配Qwen2.5）
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Qwen2.5必须设置这个，否则会报错
tokenizer.padding_side = "right"

#梯度检查点
# 【必开】梯度检查点，省40%显存
model.gradient_checkpointing_enable()
model.enable_input_require_grads()


# ====================== 2. 配置LoRA参数（微调核心） ======================
print("正在配置LoRA微调参数...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 固定：文本生成/对话任务
    r=8,               # LoRA秩：8-64之间，越大效果越好，显存占用越高
    lora_alpha=16,      # 缩放因子，一般设为r的2倍
    # Qwen2.5推荐微调这7个模块，效果远好于只调注意力层
    target_modules=[
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"

    ],
    lora_dropout=0.05,  # 防止过拟合
    bias="none",
)

# 把LoRA挂载到模型上
model = get_peft_model(model, lora_config)
print("="*50)
model.print_trainable_parameters()  # 打印可训练参数，正常只有0.1%左右
print("="*50)

# ====================== 3. 准备训练数据集 ======================
print("正在处理训练数据...")
# 这里给你示例数据，先跑通，之后换成你自己的就行
# 格式：[{"instruction": "你的指令", "input": "额外输入（没有就空着）", "output": "想要的回答"}]
with open(DATA_PATH, "r", encoding="utf-8") as f:
    train_data = json.load(f)

# ========== 新增：自动拆分训练集/评估集（无需要手动建文件） ==========
import random
random.seed(42)  # 固定随机种子，结果可复现
eval_size = 10   # 抽10条当评估集
eval_data = random.sample(train_data, eval_size)  # 随机抽10条
train_data = [x for x in train_data if x not in eval_data]  # 剩下的当训练集
print(f"✅ 自动拆分数据：训练集{len(train_data)}条，评估集{len(eval_data)}条")



# 把数据格式化成Qwen2.5的官方对话模板
def format_data(example):
    # Qwen2.5官方对话模板，必须按这个来，否则微调后模型不会对话
    prompt = f"<|im_start|>system\n你是一个专业、友好的AI助手。<|im_end|>\n"
    prompt += f"<|im_start|>user\n{example['instruction']}\n{example['input']}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    # 分词处理
    tokenized = tokenizer(prompt, truncation=True, max_length=512, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()  # 标签和输入一致，用于自回归训练
    return tokenized

# 加载并处理数据集
# raw_dataset = Dataset.from_list(json.load(open(DATA_PATH, "r", encoding="utf-8")))
# tokenized_dataset = raw_dataset.map(format_data, remove_columns=raw_dataset.column_names)


# 处理训练集（原来的代码）
raw_dataset = Dataset.from_list(train_data)
tokenized_dataset = raw_dataset.map(format_data, remove_columns=raw_dataset.column_names)

# ========== 新增：处理评估集 ==========
raw_eval_dataset = Dataset.from_list(eval_data)
tokenized_eval_dataset = raw_eval_dataset.map(format_data, remove_columns=raw_eval_dataset.column_names)


# 数据整理器（自动补全padding，适配批量训练）
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# ====================== 4. 训练参数配置 ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,  # 单卡batch size，显存不足就改成1
    gradient_accumulation_steps=4,   # 梯度累积，等效于增大batch size，显存不足就改成8
    learning_rate=5e-5,              # LoRA推荐学习率，1e-5 ~ 5e-5之间
    num_train_epochs=7,              # 训练轮数，3-5轮就足够，多了容易过拟合
    logging_steps=5,                 # 每5步打印一次训练日志
    save_strategy="epoch",           # 每训练完一轮保存一次权重
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    report_to="none",  # 不上报到任何日志平台，纯本地运行
    optim="paged_adamw_8bit",  # 8bit优化器，进一步节省显存
    # ====================== 新增：早停核心参数 ======================
    eval_strategy="epoch",  # 每轮训练后评估一次Loss
    load_best_model_at_end=True,  # 训练结束后加载效果最好的模型
    metric_for_best_model="loss",  # 以Loss为标准选最优模型
    greater_is_better=False,  # Loss越小越好（必须设False）
)

# ====================== 5. 启动训练 ======================
print("开始训练！")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]

)

# 开始训练
trainer.train()

# ====================== 6. 保存微调后的LoRA权重 ======================
print(f"正在保存LoRA权重到：{os.path.abspath(OUTPUT_DIR)}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "="*50)
print("✅ 微调完成！")
print(f"📂 LoRA权重保存路径：{os.path.abspath(OUTPUT_DIR)}")
print("="*50)