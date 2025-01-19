import os
import torch
from PIL import Image
from ..models.generator import ResnetGenerator
from ..utils.image_utils import im2tensor, tensor2im, cal
import json

def main(time, mode, input_img, max_size):
    """
    主推理函数
    Args:
        time: 时间戳
        mode: 推理模式
        input_img: 输入图像
        max_size: 最大尺寸
    Returns:
        tuple: (输入图像, 输出图像)
    """
    # 加载配置
    config_path = 'config/config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    types = config['paths']['types']
    device = torch.device(config['inference']['device'])
    
    # 计算尺寸
    calcu_size = cal(input_img, max_size)
    
    # 设置路径
    input_pth = os.path.join(config['paths']['weights_dir'], 
                            types + '_pth', 
                            f'{mode}.pth')
    out_dir = os.path.join(config['paths']['results_dir'], 
                          f'{types}_{mode}')
    os.makedirs(out_dir, exist_ok=True)
    
    model = None
    input_tensor = None
    try:
        # 清理显存
        torch.cuda.empty_cache()
        if model or input_tensor:
            del model
            del input_tensor
            
        # 加载模型
        model = ResnetGenerator(
            config['model']['input_nc'],
            config['model']['output_nc'],
            config['model']['ngf'],
            config['model']['n_blocks']
        ).to(device)
        model.load_state_dict(torch.load(input_pth, map_location=device))
        model.eval()
        
        # 推理过程
        with torch.no_grad():
            input_tensor = im2tensor(input_img, (calcu_size[1], calcu_size[0])).to(device)
            output_tensor = model(input_tensor.unsqueeze(0))
            output_img = tensor2im(output_tensor)
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
        
    finally:
        # 清理资源
        torch.cuda.empty_cache()
        if model is not None:
            del model
        if input_tensor is not None:
            del input_tensor
        torch.cuda.empty_cache()
    
    # 处理输出
    output_img = Image.fromarray(output_img)
    input_img = input_img.resize(calcu_size)
    
    return input_img, output_img 