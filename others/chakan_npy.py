import numpy as np

# 加载 .npy 文件
arr = np.load('/root/video_features/output/raft/test_video_raft.npy')

# 打印数组
print(arr)

# 打印数组的形状和数据类型等信息
print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")
