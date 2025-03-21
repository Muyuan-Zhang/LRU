import scipy.io
import matplotlib.pyplot as plt
# 加载MAT文件
mat_data = scipy.io.loadmat('/home/yychen/zhangmuyuan/DPU/test_datasets/simulation/crash32_cacti.mat')

# 获取orig变量
orig = mat_data['orig']

# 绘制每张图片
for i in range(orig.shape[2]):
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
    ax.imshow(orig[:,:,i], cmap='gray')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 移除边距
    plt.show()