from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True
)
# 加载LoRA适配器
lora_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# 推理示例
prompt = "### Instruction:\n请解释什么是LoRA微调\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = lora_model.generate(
    **inputs,
    max_new_tokens=200,  # 生成最大长度
    temperature=0.7,     # 随机性
    do_sample=True
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))