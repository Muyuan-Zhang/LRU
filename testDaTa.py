import matplotlib.pyplot as plt

# 数据
x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# psnr_mean = [12.9178, 15.8646, 16.9881, 19.4319, 23.5137, 32.3105, 32.5605, 32.7493, 33.3891]  # base52
# ssim_mean = [0.5055, 0.6621, 0.7645, 0.8146, 0.8723, 0.9378, 0.9403, 0.9418, 0.9504]
psnr_mean = [12.1551, 16.0076, 20.9255, 24.9629, 29.3226, 33.0919, 33.4574, 33.8414, 33.8559]  # base
# ssim_mean = [0.5362, 0.7670, 0.8403, 0.8953, 0.9253, 0.9461, 0.9502, 0.9418, 0.9550]

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制 PSNR 曲线
plt.plot(x, psnr_mean, label='PSNR', color='blue', marker='o')

# 绘制 SSIM 曲线
# plt.plot(x, ssim_mean, label='SSIM', color='red', marker='s')

# 添加标题和标签
plt.title('PSNR and SSIM over Data Points')
plt.xlabel('Data Point')
plt.ylabel('Value')

# 显示图例
plt.legend()

# 显示图形
plt.grid(True)
plt.show()
