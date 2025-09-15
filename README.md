# AGATENet: Adaptive Gated Experts Transformer Network for Unified Scene Dehazing

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

## Table of Contents

- [Network Architecture](#network-architecture)
- [Project Structure](#project-structure)
- [Environment Configuration](#environment-configuration)
- [Dataset Preparation](#dataset-preparation)
- [培训](#training)
- [Testing](#testing)
- [Pre-trained Models and Results](#pre-trained-models-and-results)
- [Evaluation Metrics](#evaluation-metrics)

## Network Architecture

AGATENet employs an adaptive gated experts Transformer network structure, comprising the following key components:

1. **Dual-branch LAB/RGB input strategy**: Explicitly separates dehazing and scene reconstruction tasks
2. **Precise Masked Sparse Attention mechanism**: Achieves haze density adaptive feature processing
3. **TransGuide Calibrator**: Enhances cross-level feature alignment through complementary fusion of encoder and decoder representations
4. **Gated expert selection mechanism**: Dynamically activates the most suitable pathway from pre-trained experts according to the haze characteristics of input images

## Project Structure

```
├── README.md                  # Project documentation
├── data/                      # Data processing module
│   ├── __init__.py
│   ├── data_augment.py        # Data augmentation
│   └── data_load.py           # Data loading
├── dehaze/                    # Dataset directory
│   ├── test/                  # Test set
│   ├── train/                 # Training set
│   └── val/                   # Validation set
├── eval.py                    # Model evaluation script
├── loss.py                    # Loss function definition
├── main.py                    # Main entry file
├── models/                    # Model definitions
│   ├── former.py              # AGATENet core code
│   ├── ****.py  
├── train.py                   # Training script
├── valid.py                   # Validation script

```

## Environment Configuration

The project uses the following main dependencies:

- Python 3.8+
- PyTorch 1.8+
- CUDA 
- torchvision
- numpy
- PIL/Pillow
- scikit-image
- matplotlib
- albumentations

## Dataset Preparation

1. Create a `dehaze` folder in the project root directory
2. Organize the dataset according to the following structure:

```
dehaze/
├── train/
│   ├── hazy/    # Hazy images
│   └── gt/      # Clear reference images
├── val/
│   ├── hazy/
│   └── gt/
└── test/
    ├── hazy/
    └── gt/
```

3. Ensure that hazy images and clear images have corresponding filenames

##培训

Start training with the following command:

```bash
python main.py --mode train --data NH23 --data_dir dehaze --batch_size 4 --num_epoch 3000
```

Main training parameters:

- `--model_name`: Model name, default is AGATENet
- `--data_dir`: Dataset path, default is dehaze
- `--data`: Dataset name, default is NH23
- `--batch_size`: Batch size, default is 4
- `--num_epoch`: Number of training iterations, default is 300
- `--learning_rate`: Learning rate, default is 2e-5
- `--valid_freq`: Validation frequency, default is 10
- `--save_freq`: Model saving frequency, default is 10

Models during training will be saved in the `results/{model_name}/{data}/Training-Results/` directory.

## Testing

Test using pre-trained models:

```bash
python main.py --mode test --data_dir dehaze --test_model results/AGATENet/Training-Results/Best.pkl --save_image True
```

Main testing parameters:

- `--mode`: Set to test for testing
- `--test_model`: Pre-trained model path
- `--save_image`: Whether to save result images, default is True

Test results will be saved in the `results/{model_name}/images/{data}/` directory.

## Pre-trained Models and Results

The following table provides links to download pre-trained models and results under different experimental settings:

| Result Type | Baidu Netdisk Link | Extraction Code |
|------------|-------------------|----------------|
| Dense Results | [https://pan.baidu.com/s/1VZAmODp7MprHhsykJBZ7FQ](https://pan.baidu.com/s/1VZAmODp7MprHhsykJBZ7FQ) | w6c8 |
| NHhaze Results | [https://pan.baidu.com/s/1fAVYUYNt7hsvpTj5bGZLbw](https://pan.baidu.com/s/1fAVYUYNt7hsvpTj5bGZLbw) | 6x3i |
| Indoor Results | [https://pan.baidu.com/s/1UIe_D0BzYh3m7se8ENQwyw](https://pan.baidu.com/s/1UIe_D0BzYh3m7se8ENQwyw) | 925h |
| Outdoor Results | [https://pan.baidu.com/s/11ueR9aE1NXbCQ7UQn9COfw](https://pan.baidu.com/s/11ueR9aE1NXbCQ7UQn9COfw) | 86ev |
| All-in-one Results | [https://pan.baidu.com/s/1aAkwpvKFk3wHCLJEQ8OdrA](https://pan.baidu.com/s/1aAkwpvKFk3wHCLJEQ8OdrA) | s5s2 |
| All-in-one + moe Results | [https://pan.baidu.com/s/11QpmGLuyYQdHDADgL6VdtQ](https://pan.baidu.com/s/11QpmGLuyYQdHDADgL6VdtQ) | eit5 |

## Evaluation Metrics

Model performance is evaluated using the following metrics:

**Peak Signal-to-Noise Ratio (PSNR)**
**Structural Similarity Index (SSIM)**

## Notes
1. To adjust model parameters, please modify the relevant configurations in `main.py`

## Expert Network (MOE) Training and Evaluation
- Each expert is a complete AGATENet model
- Expert parameters are frozen during training, only the gating network is trained
- When training the mixture of experts model, pre-trained expert models (obtained from AGATENet training) for individual scenarios are required
- The configuration and usage of expert networks are similar to the above
