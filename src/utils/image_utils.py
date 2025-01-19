import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def im2tensor(input_img, size):
    """将PIL图像转换为tensor"""
    transform_list = [
        transforms.Resize(size=size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
    transform = transforms.Compose(transform_list)
    image = input_img.convert('RGB')
    input_tensor = transform(image)
    return input_tensor

def tensor2im(input_image, imtype=np.uint8):
    """将tensor转换为numpy图像"""
    image_numpy = input_image.data[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def cal(input_img, max_size):
    """计算保持宽高比的新尺寸"""
    ratio = min(max_size / input_img.width, max_size / input_img.height)
    new_width = int(input_img.width * ratio)
    new_height = int(input_img.height * ratio)
    return [new_width, new_height] 