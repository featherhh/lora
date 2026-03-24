from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import torch

# ====================== 配置区（适配你的路径） ======================
BASE_MODEL_PATH = "./qwen2.5"
LORA_MODEL_PATH = "./qwen2.5-lora"

# ====================== 【Windows专属】极致显存优化配置 ======================
# 4bit量化加载，显存直降一半
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # Windows下优先用fp16，兼容性更好
)

# ====================== 加载模型（修复Windows兼容性） ======================
print("正在加载基础模型和分词器...")
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 加载基础模型（Windows下的稳妥配置）
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="sequential",  # Windows下用sequential比auto更稳妥
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_cache=False,  # 训练时关了，推理时也关，省显存
)

# 【关键修复】加载LoRA权重前，先把模型移到GPU，避免offload报错
print("正在加载LoRA微调权重...")
model = PeftModel.from_pretrained(
    model,
    LORA_MODEL_PATH,
    torch_dtype=torch.float16,
    is_trainable=False,  # 推理模式，不需要训练
)

# 把LoRA权重合并到基础模型里（可选，合并后推理更快，显存占用略高）
# 如果你想合并，取消下面这行的注释：
# model = model.merge_and_unload()

# 模型移到评估模式
model.eval()


# ====================== 测试对话 ======================
def chat(question):
    # 构建Qwen2.5官方对话模板
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

    # 分词并移到GPU
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    # 生成回答（Windows下优化的生成参数）
    with torch.no_grad():  # 推理模式，不计算梯度，省显存
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id  # Windows下必须显式设置
        )

    # 解码并返回结果
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|im_start|>assistant\n")[-1].strip()


# ====================== 开始测试 ======================
print("\n" + "=" * 50)
print("开始测试微调后的模型...")
print("=" * 50)

# 测试几个问题
test_questions = [
    "什么是LoRA微调？",
    "1+1等于几？",
    "用一句话形容夏天"
]

for i, question in enumerate(test_questions):
    print(f"\n【问题 {i + 1}】：{question}")
    print(f"【回答】：{chat(question)}")

print("\n" + "=" * 50)
print("✅ 测试完成！")
print("=" * 50)

# 清空显存缓存
torch.cuda.empty_cache()