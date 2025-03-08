# DCGAN on CelebA Dataset

This repository contains an implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** trained on the **CelebA** dataset to generate realistic human face images.

## 📌 Features
- Implemented in **PyTorch**
- Uses **DCGAN** architecture for high-quality image generation
- Trained on the **CelebA dataset**
- Supports **CUDA acceleration** for faster training

## 📂 Repository Structure
```
📦 DCGAN-CelebA
├── 📜 Assignment_3_GAN_Lab_22070126014.ipynb  # Jupyter Notebook with DCGAN implementation
├── 📜 README.md                               # Project documentation
└── 📂 data/                                   # Directory for dataset (not included in repo)
```

## 🛠️ Setup Instructions
### 1️⃣ Install Dependencies
Ensure you have Python and PyTorch installed. Install dependencies using:
```bash
pip install torch torchvision numpy matplotlib
```

### 2️⃣ Download CelebA Dataset
You need to download the **CelebA dataset** manually. Extract it and set the correct path in the notebook:
```python
dataset = datasets.ImageFolder(root="/path/to/celeba", transform=transform)
```

### 3️⃣ Run the Notebook
Open the Jupyter Notebook and execute the cells to train the DCGAN:
```bash
jupyter notebook Assignment_3_GAN_Lab_22070126014.ipynb
```

## 🎯 Model Architecture
The model consists of:
- **Generator**: Uses transpose convolution layers to generate realistic face images.
- **Discriminator**: A CNN-based binary classifier that distinguishes real vs. fake images.

## 📊 Training Details
- **Image Size**: 64x64
- **Latent Dimension**: 100
- **Batch Size**: 32
- **Learning Rates**:
  - Generator: `0.0002`
  - Discriminator: `0.00005`
- **Optimization**: Adam (`beta1=0.5`, `beta2=0.98`)

## 📸 Results
Once trained, the model generates synthetic face images similar to CelebA samples. Example output:
```
[![generated_epoch_9](https://github.com/user-attachments/assets/57b6ed91-0b7e-4b34-8a19-dfd0294861df)](https://github.com/AnshulJain2004/DCGAN-CelebA/blob/main/generated_epoch_9.png)

```

## 🏆 Acknowledgments
- **DCGAN Paper**: *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* (Radford et al., 2015)
- **Dataset**: CelebA dataset by **Liu et al.**

---
