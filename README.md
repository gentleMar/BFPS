# Code for Paper: BFPS — A Boundary-Focused Polyp Segmentation Model via Frequency Domain Separation (ICME 2025)

This repository contains the official implementation of our ICME 2025 paper:  
**"BFPS: A Boundary-Focused Polyp Segmentation Model via Frequency Domain Separation"**.  

If you have any questions, feel free to reach out.

---

## Environment Setup

```bash
# Create a conda environment
conda create -n BFPS python=3.9
conda activate BFPS

# Install PyTorch (CUDA 11.8)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install "numpy<2" albumentations==1.3.1 timm thop medpy
```

## Directory Structure

```md
BFPS/
├── pretrained/
│   └── mit_b4.pth	# see SegFormer repo
├── data/
│   └── polyp/	# see PraNet repo
│       ├── TrainDataset/
│       │   └── ...
│       └── TestDataset/
│           └── ...
├── [code files]
```
Pretrained weights are available in [SegFormer](https://github.com/NVlabs/SegFormer). Datasets are available in [PraNet](https://github.com/DengPingFan/PraNet).

## Run Training

```bash
python train.py
```
You can modify the training settings directly in train.py. After training is completed, the model will run evaluation on the test set.





