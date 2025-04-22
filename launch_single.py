import os
import torch

def main():
    # 确保只使用一张 GPU（A100）
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # 直接调用你们的训练脚本 main.py
    # 等价于：python main.py --archs FineParser --benchmark FineDiving
    import sys
    sys.argv = ["main.py", "--archs", "FineParser", "--benchmark", "FineDiving"]

    # 启动 main.py 中的训练流程
    import main  # 确保 main.py 的入口函数名是 main()
    main.main()  # 如果你们是用 main_worker(...)，就改成调用它

if __name__ == "__main__":
    main()
