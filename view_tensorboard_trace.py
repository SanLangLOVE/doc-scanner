import subprocess
import sys
import os

# 检查并安装 tensorboard
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'show', 'tensorboard'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except subprocess.CalledProcessError:
    print("未检测到 tensorboard，正在自动安装...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorboard'])

print("\nTensorBoard 已安装。\n")

# 检查 trace 文件
trace_file = 'trace_inference.json'
if not os.path.exists(trace_file):
    print(f"未找到 {trace_file}，请先运行模型推理生成该文件。")
    sys.exit(1)

print(f"已检测到 {trace_file}。\n")

print("请在终端执行以下命令启动 TensorBoard：\n")
print("    tensorboard --logdir=./\n")
print("然后在浏览器打开 http://localhost:6006 ，在 PROFILE 标签页选择 trace_inference.json 文件进行分析。\n") 