import os
import h5py  # 用于读取 MATLAB v7.3 格式
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# 规范化路径
mat_file = r"K:\postgraduate\BIRNAT\real data\BIRNAT_domino.mat"
output_folder = r"D:\postgraduate\work\LRU\BIRNAT\real_data\BIRNAT_domino"
os.makedirs(output_folder, exist_ok=True)

# 检查 .mat 文件格式
def load_mat_file(file_path):
    try:
        data = sio.loadmat(file_path)  # 适用于 v7.2 及以下
        print("Loaded using scipy.io.loadmat")
        return data
    except NotImplementedError:
        print("MATLAB v7.3 detected, using h5py...")
        with h5py.File(file_path, 'r') as f:
            return {key: np.array(f[key]) for key in f.keys()}

# 加载 .mat 文件
data = load_mat_file(mat_file)

# 查看文件中的变量
print("Available keys:", data.keys())

# 选择要显示的变量 'meas_bayer'
key_name = "pic"  # 修改为你的 .mat 文件中的实际变量名
if key_name not in data:
    raise KeyError(f"Key '{key_name}' not found. Available keys: {data.keys()}")

# 获取 'meas_bayer' 数据
image_data = data[key_name]

# 确保数据格式正确
if image_data.ndim == 3:  # 假设数据为 (帧数, 高度, 宽度) 或 (帧数, 高度, 宽度, 通道)
    # image_data = image_data.transpose(2, 1, 0)
    frames, height, width = image_data.shape
    is_color = False
    print(f"Detected grayscale video with {frames} frames.")
elif image_data.ndim == 4 and image_data.shape[1] == 3:  # (帧数, 通道数, 高度, 宽度)
    frames, channels, height, width = image_data.shape
    is_color = True
    print(f"Detected color video with {frames} frames.")
else:
    raise ValueError(f"Unexpected data shape {image_data.shape}, cannot handle this format.")

# 显示和保存每一帧图像
for i in range(frames):
    frame = image_data[i]  # 获取第 i 帧

    # 如果是彩色图像，转换为 RGB
    if is_color:
        frame = np.transpose(frame, (2, 1, 0))  # 将通道轴移到最后 (高度, 宽度, 通道数)
    else:
        frame = np.transpose(frame, (0, 1))

    # 创建一个与图像尺寸相同的画布
    fig = plt.figure(figsize=(frame.shape[1]/100, frame.shape[0]/100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # 显示图像（灰度图像或彩色图像）
    ax.imshow(frame, cmap='gray' if not is_color else None)

    # 保存图像
    output_path = os.path.join(output_folder, f'frame_{i + 1}.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

    print(f'Saved frame {i + 1} to {output_path}')
