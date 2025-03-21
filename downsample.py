import torch
import torch.nn as nn

class DownsampleLayer(nn.Module):
    def __init__(self, in_channels):
        super(DownsampleLayer, self).__init__()

        # 逐点卷积层，输出通道数翻倍
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1)

        # 像素反洗牌操作，用于下采样，倍率设为2
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)

    def forward(self, x):
        # 逐点卷积操作
        x = self.pointwise_conv(x)

        # 像素反洗牌实现下采样
        x = self.pixel_unshuffle(x)

        return x

# 测试代码
input_tensor = torch.randn(1, 64, 128, 128)  # 假设输入尺寸为 [batch, channels, height, width]
downsample_layer = DownsampleLayer(in_channels=64)
output_tensor = downsample_layer(input_tensor)

print("Input shape:", input_tensor.shape)
print("Output shape:", output_tensor.shape)
