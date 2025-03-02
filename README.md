<h1 align="center" >
    Dual Residual Dense Network for Image Denoising
</h1>

## Overview
This repository contains the implementation of the **Dual Residual Dense Network (DRDNet)** for image denoising, originally proposed in 2022. Due to the lengthy and rigorous review process, the work was finally published in 2025 in *Engineering Applications of Artificial Intelligence* ([DOI: 10.1016/j.engappai.2025.110275](https://doi.org/10.1016/j.engappai.2025.110275)).

> **Note:** Since the code was initially developed in 2022, some parts may be outdated or require modifications for compatibility with the latest deep learning frameworks.

## Methodology

DRDNet is designed to effectively remove noise from images while maintaining computational efficiency. It leverages:
- **Residual Learning & Dense Connectivity:** To mitigate vanishing gradient issues and improve feature propagation.
- **Batch Renormalization:** To enhance training stability, particularly for small batch sizes.
- **Dilated Convolutions:** To expand the receptive field without increasing computational cost.

The model was trained on the **DIV2K** dataset and evaluated on **Kodak24** and **BSDS300**, achieving state-of-the-art results in terms of PSNR and SSIM.



## Steps to Use

### 1. Prepare the Dataset

Convert the dataset into 64x64 image patches using the scripts in the `data_preprocessing` folder.

### 2. Train the Model

Train the network for a noise level (NL) of 10 using the following command:

```bash
python main.py --NL 10 --BatchReNormalization True --input_shape (64,64,3)
```

### 3. Test the Model

Evaluate the trained model on a test dataset using:

```bash
python main.py --NL 10 --BatchReNormalization True --input_shape (64,64,3) --test_data_name 'Kodak_Test_C_NL30.h5'
```

## Citation

If you use this work in your research, please cite:

```bibtex
@article{batool2025drdnet,
  author = {Batool, Isma and Imran, Muhammad},
  title = {A Dual Residual Dense Network for Image Denoising},
  journal = {Engineering Applications of Artificial Intelligence},
  volume = {147},
  pages = {110275},
  year = {2025},
  doi = {10.1016/j.engappai.2025.110275}
}
```


---

For questions or issues, please open an issue or contact the authors:

- **Muhammad Imran** ([muhammad.imran@ufl.edu](mailto\:muhammad.imran@ufl.edu))
- **Isma Batool** ([isma.batool@buitms.edu.pk](mailto\:isma.batool@buitms.edu.pk))






