
# AGATENet: A Adaptive Gated Experts Transformer Network for Unified Scene Dehazing

Xin Gao, Zhiyu Lyu, and Weijie Ren


---

> **Abstract:** *Real-world haze exhibits heterogeneous compositions and continuous concentration gradients, posing significant challenges to accurate haze removal and image restoration. While existing single image dehazing models perform well under specific haze conditions, they often lack a unified framework that generalizes across diverse real-world scenarios. 
To address this limitation, we propose AGATENet—a novel framework for unified image dehazing. 
The core of our approach decouples the dehazing task into two critical levels: content adaptive processing and scenario specific decision making.
At the content level, a dual-branch LAB/RGB input strategy explicitly separates dehazing and scene reconstruction, thereby mitigating multi-objective optimization conflicts. 
Moreover, we introduce a Precise Masked Sparse Attention mechanism to achieve haze density adaptive feature processing, thereby guiding the network to concentrate on restoring haze degraded regions.
Simultaneously, a TransGuide Calibrator enhances cross-level feature alignment through complementary fusion of encoder and decoder representations.
At the scenario level, a gated expert selection mechanism dynamically activates the most suitable pathway from pre-trained experts according to the input image's haze characteristics, enabling end-to-end adaptive dehazing without manual intervention.
Extensive experiments demonstrate that AGATENet outperforms state-of-the-art (SOTA) methods across complex haze conditions.*
---

## 目录

- [网络架构](#网络架构)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据集准备](#数据集准备)
- [训练](#训练)
- [测试](#测试)
- [预训练模型和结果](#预训练模型和结果)
- [评估指标](#评估指标)

## 网络架构

AGATENet采用自适应门控专家Transformer网络结构，主要包含以下关键组件：

1. **双分支LAB/RGB输入策略**：明确分离去雾和场景重建任务
2. **精确掩码稀疏注意力机制**：实现雾霾密度自适应特征处理
3. **TransGuide校准器**：通过编码器和解码器表示的互补融合增强跨级特征对齐
4. **门控专家选择机制**：根据输入图像的雾霾特性动态激活最合适的预训练专家路径

![网络架构图](fig3.png)

## 项目结构

```
├── README.md                  # 项目说明文档
├── data/                      # 数据处理模块
│   ├── __init__.py
│   ├── data_augment.py        # 数据增强
│   └── data_load.py           # 数据加载
├── dehaze/                    # 数据集目录
│   ├── test/                  # 测试集
│   ├── train/                 # 训练集
│   └── val/                   # 验证集
├── eval.py                    # 模型评估脚本
├── loss.py                    # 损失函数定义
├── main.py                    # 主入口文件
├── models/                    # 模型定义
│   ├── former.py              # AGATENet核心代码
│   ├── ****.py  
├── train.py                   # 训练脚本
├── valid.py                   # 验证脚本

```

## 环境配置

项目使用以下主要依赖：

- Python 3.8+
- PyTorch 1.8+
- CUDA 
- torchvision
- numpy
- PIL/Pillow
- scikit-image
- matplotlib
- albumentations

## 数据集准备

1. 在项目根目录创建 `dehaze` 文件夹
2. 按照以下结构组织数据集：

```
dehaze/
├── train/
│   ├── hazy/    # 雾霾图像
│   └── gt/      # 清晰参考图像
├── val/
│   ├── hazy/
│   └── gt/
└── test/
    ├── hazy/
    └── gt/
```

3. 确保雾霾图像和清晰图像文件名一一对应

## 训练

使用以下命令开始训练：

```bash
python main.py --mode train --data NH23 --data_dir dehaze --batch_size 4 --num_epoch 3000
```

主要训练参数：

- `--model_name`: 模型名称，默认为AGATENet
- `--data_dir`: 数据集路径，默认为dehaze
- `--data`: 数据集名字，默认为NH23
- `--batch_size`: 批次大小，默认为2
- `--num_epoch`: 训练迭代次数，默认为300
- `--learning_rate`: 学习率，默认为2e-5
- `--valid_freq`: 验证频率，默认为10
- `--save_freq`: 模型保存频率，默认为10

训练过程中的模型会保存在 `results/{model_name}/{data}/Training-Results/` 目录下。

## 测试

使用预训练模型进行测试：

```bash
python main.py --mode test --data_dir dehaze --test_model results/AGATENet/Training-Results/Best.pkl --save_image True
```

主要测试参数：

- `--mode`: 设置为test进行测试
- `--test_model`: 预训练模型路径
- `--save_image`: 是否保存结果图像，默认为True

测试结果会保存在 `results/{model_name}/images/{data}/` 目录下。

## 预训练模型和结果

下表提供了不同实验设置下的预训练模型和结果下载链接：

| 结果类型 | 百度网盘链接 | 提取码 |
|---------|------------|-------|
| Dense结果 | [https://pan.baidu.com/s/1VZAmODp7MprHhsykJBZ7FQ](https://pan.baidu.com/s/1VZAmODp7MprHhsykJBZ7FQ) | w6c8 |
| NHhaze结果 | [https://pan.baidu.com/s/1fAVYUYNt7hsvpTj5bGZLbw](https://pan.baidu.com/s/1fAVYUYNt7hsvpTj5bGZLbw) | 6x3i |
| indoor结果 | [https://pan.baidu.com/s/1UIe_D0BzYh3m7se8ENQwyw](https://pan.baidu.com/s/1UIe_D0BzYh3m7se8ENQwyw) | 925h |
| outdoor结果 | [https://pan.baidu.com/s/11ueR9aE1NXbCQ7UQn9COfw](https://pan.baidu.com/s/11ueR9aE1NXbCQ7UQn9COfw) | 86ev |
| All in one结果 | [https://pan.baidu.com/s/1aAkwpvKFk3wHCLJEQ8OdrA](https://pan.baidu.com/s/1aAkwpvKFk3wHCLJEQ8OdrA) | s5s2 |
| all in one+moe结果 | [https://pan.baidu.com/s/11QpmGLuyYQdHDADgL6VdtQ](https://pan.baidu.com/s/11QpmGLuyYQdHDADgL6VdtQ) | eit5 |

## 评估指标

模型性能通过以下指标进行评估：

 **峰值信噪比 (PSNR)**
**结构相似性 (SSIM)**


## 注意事项
1. 如需调整模型参数，请修改 `main.py` 中的相关配置


## 专家网络训练与评估
- 每个专家都是完整的AGATENet模型
- 专家参数在训练时被冻结，只训练门控网络
- 在训练混合专家模型的时候，需要提供单独场景的预训练专家模型（AGATENet训练得到）
- 专家网络的配置与使用与上述同理
