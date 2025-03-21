import matplotlib.pyplot as plt
import pandas as pd

#######
#   绘制图片代码
#######
# 示例数据
data = {
    'Method': ['Ours', 'ADMM-Net', 'GAP-Net', 'DGSMP', 'DNU', 'Our Unfolding', 'ONR', 'Previous Unfolding'],
    'FLOPS(G)': [40, 38, 36, 34, 32, 30, 28, 25],  # 横坐标
    'PSNR': [32, 34, 36, 38, 40, 42, 44, 46],  # 纵坐标
    'Params(M)': [5, 7, 9, 3, 4, 6, 8, 10]  # 点的大小（模型参数）
}

df = pd.DataFrame(data)

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制散点图，点的大小由 'Params(M)' 决定
scatter = plt.scatter(
    df['FLOPS(G)'],  # 横坐标
    df['PSNR'],  # 纵坐标
    s=df['Params(M)'] * 50,  # 点的大小，乘以一个系数以调整显示效果
    alpha=0.6,  # 点的透明度
    c='blue',  # 点的颜色
    edgecolors='black'  # 点的边缘颜色
)

# 添加标签
for i, method in enumerate(df['Method']):
    plt.text(df['FLOPS(G)'][i], df['PSNR'][i], method, fontsize=9, ha='right')

# 添加标题和标签
plt.title('Comparison of Methods: FLOPS vs PSNR (Point Size = Model Parameters)')
plt.xlabel('FLOPS(G)')
plt.ylabel('PSNR(dB)')

# 显示网格
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图表
plt.show()
