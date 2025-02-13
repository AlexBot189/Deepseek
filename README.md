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
1、使用llama.cpp框架来运行量化后的模型
