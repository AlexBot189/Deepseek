import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

# 设置 PyTorch 使用的最大 CPU 核心数
torch.set_num_threads(4)  # 控制最大 CPU 核心数

# 加载模型和分词器
model_path = "/home/test1/DeepSeek-R1-Distill-Qwen-1.5B"  # 修改为你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 使用 16 位精度减少内存和计算负担
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

# 输入文本
prompt = "你好，请写一首关于春天的诗。"
inputs = tokenizer(prompt[:512], return_tensors="pt")  # 限制最大长度为512个字符

# 生成文本时控制参数，减少内存占用
outputs = model.generate(
    inputs.input_ids,
    max_length=512,  # 控制生成文本的最大长度
    temperature=0.7,
    do_sample=True,
    num_return_sequences=1,  # 只生成一个序列
    num_beams=3,  # 例如，设置为4时会启用beam search，这有时能提高效率
#    early_stopping=True  # 提前停止，减少计算量
)

# 解码输出
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("模型回复：\n", response)

# 清理不再使用的变量，释放内存
del inputs
del outputs
gc.collect()  # 强制进行垃圾回收
