import os

from PIL import Image, ImageDraw

# 1. 加载图片
image_paths = [
               # r'D:\postgraduate\work\LRU\gt\kobe_cacti\frame_28.png',
               # r'D:\postgraduate\work\LRU\gt\kobe_cacti\frame_30.png',
               # r'D:\postgraduate\work\LRU\gt\kobe_cacti\frame_27.png',
               # r'D:\postgraduate\work\LRU\gt\kobe_cacti\frame_27.png',
               r'D:\postgraduate\work\LRU\model_fig\base\base_6th\kobe\DPU_base_27.png',
               r'D:\postgraduate\work\LRU\model_fig\base\base_6th\kobe\DPU_base_28.png',
               r'D:\postgraduate\work\LRU\model_fig\base\base_6th\kobe\DPU_base_29.png',
               r'D:\postgraduate\work\LRU\model_fig\base\base_6th\kobe\DPU_base_30.png',
]
# image_paths = [r'D:\postgraduate\work\LRU\model_fig\base\base_6th\kobe\DPU_base_29.png']
images = [Image.open(path) for path in image_paths]

# 2. 确保图片是 RGB 模式
images = [img.convert("RGB") for img in images]
# 2. 选择局部区域
x1, y1 = 64, 64  # 左上角坐标
x2, y2 = 96, 96  # 右下角坐标

# 3. 裁剪每张图片的局部区域
cropped_images = [img.crop((x1, y1, x2, y2)) for img in images]

# 4. 放大局部区域
zoom_scale = 4  # 放大倍数
zoomed_images = [img.resize((img.width * zoom_scale, img.height * zoom_scale), Image.BILINEAR) for img in
                 cropped_images]

# 5. 将放大后的局部区域嵌入到原始图片的右下角
for i, (original, zoomed, image_path) in enumerate(zip(images, zoomed_images, image_paths)):
    # 创建一个新的画布，大小为原始图片的大小
    combined_image = original.copy()

    # 计算放大区域的嵌入位置（右下角）
    offset_x = original.width - zoomed.width
    offset_y = original.height - zoomed.height

    # 将放大后的局部区域粘贴到原始图片的右下角
    combined_image.paste(zoomed, (offset_x, offset_y))

    # 6. 在原始图片上绘制裁剪区域的方框
    draw = ImageDraw.Draw(combined_image)
    # 裁剪区域的方框
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    # 7. 在嵌入的放大区域上绘制方框
    embed_box_x1 = offset_x
    embed_box_y1 = offset_y
    embed_box_x2 = offset_x + zoomed.width
    embed_box_y2 = offset_y + zoomed.height
    draw.rectangle([embed_box_x1, embed_box_y1, embed_box_x2, embed_box_y2], outline="red", width=2)

    # 8. 生成保存路径
    # 获取原始文件名（不带路径）
    original_filename = os.path.basename(image_path)
    # 去掉文件扩展名
    filename_without_ext = os.path.splitext(original_filename)[0]
    # 生成新的文件名
    output_filename = f"{filename_without_ext}_combined_frame.png"
    # 生成保存路径
    output_path = os.path.join(os.path.dirname(image_path), output_filename)

    # 保存结果
    combined_image.save(output_path)
    print(f'Saved combined image to {output_path}')

    # 直接显示图片，不添加额外空白和标题
    combined_image.show()
