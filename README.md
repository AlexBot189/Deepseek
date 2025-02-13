# Deepseek
Deepseek-R1部署

# 部署和运行原始模型
一、安装依赖
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git wget curl &&
#安装PyTorch的依赖（如需GPU支持）
sudo apt install -y nvidia-cuda-toolkit  # 仅限NVIDIA GPU用户

二、配置Python虚拟环境
为避免依赖冲突，创建独立环境：
#创建名为 deepseek 的虚拟环境
python3 -m venv ~/deepseek-env

#激活虚拟环境
source ~/deepseek-env/bin/activate

#激活后，终端提示符前会显示 (deepseek-env)
#退出虚拟环境的命令是 deactivate

三、安装PyTorch和依赖库
根据是否使用GPU选择安装命令：
方案A：仅CPU运行
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

方案B：GPU加速（需NVIDIA显卡）
#先确认CUDA版本（假设已安装CUDA 11.8）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

四、获取DeepSeek模型文件
推荐手动下载


五、编写推理代码
创建文件 run_deepseek.py，写入以下内容：

from transformers import AutoModelForCausalLM, AutoTokenizer

#加载模型和分词器
model_path = "./deepseek-7b"  # 修改为你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

#输入文本
prompt = "你好，请写一首关于春天的诗。"
inputs = tokenizer(prompt, return_tensors="pt")

#生成文本
outputs = model.generate(
    inputs.input_ids,
    max_length=200,
    temperature=0.7,
    do_sample=True
)

#解码输出
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("模型回复：\n", response)


六、运行模型
#确保在虚拟环境中
source ~/deepseek-env/bin/activate

#运行推理脚本
python run_deepseek.py

# #########################
缺点：原始模型参数很大，精度很高，需要占用CPU和消耗过多的DDR，运行很慢，都需要通过量化和剪枝才能在有限的资源上运行
动态量化更耗CPU和DDR，所以需要使用离线量化后的模型，直接拿来用
https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF
# #########################























# 部署量化后的模型
1、手动量化：（手动量化需要硬件性能较好的电脑）
llama.cpp 是专为CPU优化的高效推理引擎，支持GGUF格式量化模型
#转换模型为GGUF格式
# 安装转换工具
pip install llama-cpp-python
# 将PyTorch模型转换为GGUF
python -m llama_cpp.convert \
  --input-model ./path/to/deepseek-r1.5B \
  --output-model ./deepseek-r1.5B-Q4.gguf \
  --quantize q4_0  # 4-bit量化

# 尝试转换与量化
#下载完整的 PyTorch 格式模型（包含 pytorch_model.bin 和 config.json）
./deepseek-qwen-1.5B/
  ├── config.json
  ├── pytorch_model.bin
  └── tokenizer.json

  #安装转换工具
  git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make  # 编译 llama.cpp
pip install torch transformers  # 用于转换脚本

#修改转换脚本适配 Qwen 架构
在 convert.py 中查找层名称匹配逻辑
if "qwen" in model_name.lower():
    # 替换层名称映射
    layer_mapping = {
        "transformer.h.{}.ln_1.weight": "model.layers.{}.input_layernorm.weight",
        "transformer.h.{}.attn.c_attn.weight": "model.layers.{}.attention.wq.weight",  # 需拆分 Q/K/V
        # 其他层匹配...
    }

 #将 PyTorch 模型转换为 FP16 GGUF 格式
python3 convert.py \
  --input-dir ./deepseek-qwen-1.5B \
  --output-dir ./output-gguf \
  --outtype f16  # 先转未量化格式测试兼容性

  #量化模型：选择 4-bit 量化（平衡速度与精度）
./quantize ./output-gguf/deepseek-qwen-1.5B.f16.gguf ./output-gguf/deepseek-qwen-1.5B.Q4_K_M.gguf Q4_K_M

#运行测试
./main -m ./output-gguf/deepseek-qwen-1.5B.Q4_K_M.gguf \
  -n 128  # 生成128个token
  -p "你好，请介绍一下上海。"


# 使用llama推理
  from llama_cpp import Llama

#加载量化模型
llm = Llama(
    model_path="./deepseek-r1.5B-Q4.gguf",
    n_threads=8,  # 设置CPU线程数
    n_gpu_layers=0  # 纯CPU运行
)
#生成文本
output = llm.create_completion("你好，请写一首诗。", max_tokens=100)
print(output["choices"][0]["text"])

2、使用llama.cpp框架来运行量化后的模型
使用https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF 里面下载的各种精度的模型
./llama-simple-chat -m ~/DeepSeek-R1-Distill-Qwen-1.5B/gguf/DeepSeek-R1-Distill-Qwen-1.5B-IQ2_M.gguf



# 各精度模型的性能对比
![image](https://github.com/user-attachments/assets/9f9d5663-34ac-433a-9888-2faf4ee741ee)

配置	                                模型大小	        内存占用 (CPU)	生成速度 (tokens/s)	输出质量
原始 PyTorch (FP32)	5.8 GB	            ~6.0 GB	            2-3	                                        高
GGUF Q4_K_M	            0.9 GB	            ~1.2 GB	            8-12	                                中高
GGUF Q2_K	                0.5 GB	            ~0.7 GB	            15-20	                                    中

