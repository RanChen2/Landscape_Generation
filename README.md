# LandscapeGAN

基于GAN的景观生成项目，专注于居住区景观的生成。支持多种场景类型，包括居住区和公园等。

## 项目结构

```
landscape_gan/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖项
├── config/
│   └── config.json             # 配置文件
├── src/
│   ├── models/                 # 模型定义
│   │   ├── generator.py       # 生成器模型
│   │   └── blocks.py         # ResNet模块
│   ├── utils/                  # 工具函数
│   │   └── image_utils.py    # 图像处理工具
│   └── pipeline/              # 推理流程
│       └── inference.py      # 推理主流程
├── weights/                    # 模型权重
│   ├── residential_pth/       # 居住区模型权重
│   ├── park_pth/             # 公园模型权重
│   └── LandscapeSuperMix/    # 混合模型权重
└── samples/                    # 示例图片
```

## 功能特点

- 支持多种场景类型（居住区、公园等）
- 基于ResNet的生成器架构
- 支持批量处理多张图片
- 自动维持图像宽高比
- GPU加速支持
- 内存优化设计

## 安装

1. 克隆项目：
```bash
git clone [项目地址]
cd landscape_gan
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 配置文件

在 `config/config.json` 中设置：
```json
{
    "model": {
        "input_nc": 3,          # 输入通道数
        "output_nc": 3,         # 输出通道数
        "ngf": 64,             # 生成器特征通道基数
        "n_blocks": 9          # ResNet块数量
    },
    "paths": {
        "input_dir": "samples",                # 输入图片目录
        "weights_dir": "weights",              # 权重文件目录
        "results_dir": "result",               # 结果输出目录
        "types": "residential"                 # 场景类型
    },
    "inference": {
        "max_size": 512,        # 最大图像尺寸
        "device": "cuda",       # 运行设备 (cuda/cpu)
        "mode": "default",      # 推理模式
        "time": "20240417"      # 时间戳
    },
    "input_images": [           # 待处理图片列表
        {
            "name": "test.jpg", # 图片名称
            "mode": "default"   # 处理模式
        }
    ]
}
```

## 使用方法

1. 准备输入图片：
   - 将待处理图片放入 `samples` 目录
   - 在配置文件中添加图片信息

2. 选择模型类型：
   - 在配置文件中设置 `types` 为 `residential` 或 `park`
   - 确保对应的权重文件存在

3. 运行推理：
```bash
python main.py
```

## 注意事项

1. 显存管理：
   - 代码已优化显存使用
   - 自动清理不需要的资源
   - 支持大尺寸图像处理

2. 输出说明：
   - 输出图像保持原始宽高比
   - 文件名格式：`output_原文件名`
   - 保存在配置的输出目录中

3. 错误处理：
   - 自动跳过处理失败的图片
   - 提供处理成功率统计
   - 详细的错误信息输出