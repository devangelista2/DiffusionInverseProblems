# Diffusion Inverse Problems

## Installation
To replicate the experiments presented in the paper, just clone this repository locally and create the required `model_weights` folder, by running:

```
git clone https://github.com/devangelista2/DiffusionInverseProblems.git
cd DiffusionInverseProblems
mkdir model_weights
```

## Datasets
We performed experiments over multiple datasets. In the following we report a description of how to download each of them.

### CelebA-grayscale

The models are trained on a grey-scale version of CelebA faces, in the 64x64 cropped format. The dataset in .npy format we used is available on HuggingFace, and can be downloaded by:

```
git clone https://huggingface.co/datasets/TivoGatto/celeba_grayscale
```