import os
import json
from PIL import Image
from src.pipeline.inference import main

def load_config(config_path='config/config.json'):
    """加载JSON配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_single_image(image_config, global_config):
    """处理单张图像"""
    # 构建输入路径
    input_path = os.path.join(global_config['paths']['input_dir'], image_config['name'])
    if not os.path.exists(input_path):
        print(f"Warning: Input file not found: {input_path}")
        return False

    # 创建输出目录
    output_dir = global_config['paths']['results_dir']
    os.makedirs(output_dir, exist_ok=True)

    # 加载输入图像
    input_img = Image.open(input_path)

    # 获取推理参数
    mode = image_config.get('mode', global_config['inference']['mode'])
    max_size = global_config['inference']['max_size']
    time = global_config['inference']['time']

    # 运行推理
    input_img, output_img = main(time, mode, input_img, max_size)

    if output_img is not None:
        # 保存结果
        output_name = f"output_{image_config['name']}"
        output_path = os.path.join(output_dir, output_name)
        output_img.save(output_path)
        print(f"Results saved to {output_path}")
        return True
    return False

def main_process():
    """主处理流程"""
    # 加载配置
    config = load_config()

    # 处理所有图像
    success_count = 0
    total_count = len(config['input_images'])

    for image_config in config['input_images']:
        if process_single_image(image_config, config):
            success_count += 1

    print(f"\nProcessing completed: {success_count}/{total_count} images successful")

if __name__ == '__main__':
    main_process() 