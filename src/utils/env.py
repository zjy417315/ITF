import os
import cv2

# 1. 限制底层科学计算库的多线程调度
# 避免与 PyTorch DataLoader 的 multiprocessing 冲突导致 CPU 满载假死
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 2. 彻底禁用 OpenCV 的多线程机制
cv2.setNumThreads(0)

# 3. 如果有其他针对该项目的特定环境变量，也可以统一加在这里
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # 为后续可能需要的完全确定性训练做准备