# Overview
This repository provides the official implementation of our proposed framework, Dehazing-DiffGAN, as presented in the study "Dehazing-DiffGAN paper" The model integrates the generative strengths of both diffusion processes and generative adversarial networks (GANs) to address the challenges of image dehazing, particularly in remote sensing applications. 


# Abstract
Haze resulting from atmospheric particles severely degrades the visual quality of images, creating significant challenges for downstream computer vision tasks. Conventional dehazing methods, particularly under heavy haze conditions, frequently produce artifacts such as blurring or structural inconsistencies. To address these limitations, we propose Dehazing-DiffGAN, a novel two-phase architecture that combines the denoising capabilities of diffusion models with the perceptual refinement power of GANs. In the first phase, a diffusion-based generator progressively eliminates haze while preserving both global and local image features. The second phase employs a GAN-based enhancer to recover fine-grained visual details and enhance overall realism. We validate our model on two challenging benchmarks—SateHaze1k and DNH-HAZE—demonstrating that Dehazing-DiffGAN achieves superior performance compared to current state-of-the-art methods, especially in terms of PSNR and LPIPS. These results highlight the model’s robustness and generalizability for high-fidelity image restoration in complex remote sensing environments.

## 📊 Results

To thoroughly evaluate the effectiveness of the proposed Dehazing-DiffGAN framework, we performed extensive benchmarking against over fifteen leading dehazing approaches. Our comparative study encompasses a broad range of traditional and deep learning-based methods, including but not limited to: DCP, AOD-Net, FCTF-Net, PFF-Net, GridDehaze-Net, FFA-Net, SCA-Net, AIDTransformer, EDED-Net, ConvIR-S, ADND-Net, PhDNet, and HSMD-Net. These models were selected for their prominence and diversity in architectural design, and the evaluation was conducted under various haze densities using two widely adopted datasets.


👉 The analysis employed multiple quantitative metrics—primarily PSNR, SSIM, and LPIPS—to measure reconstruction accuracy, structural similarity, and perceptual quality. The results are summarized in two detailed tables:

- **Table 1** compares methods based on PSNR, SSIM, and LPIPS, offering a holistic view of both pixel-wise fidelity and visual realism. 
- **Table 2** focuses on models that report only PSNR and SSIM scores, with additional information on model complexity measured in terms of parameter count (in millions), facilitating a trade-off analysis between accuracy and computational overhead.


## 📂 Dataset Result Visualizations

- 🔗 **Results on DNH-HAZE Dataset**: [View Folder](https://drive.google.com/drive/folders/1dT_7bAajfdD9oXRbTqBJX-aZFt1tIgaB?usp=sharing)  
- 🏆 Our Dehazing-DiffGAN achieved **best LPIPS** in the **CVPR NTIRE 2024 Dense and Non-Homogeneous Dehazing Challenge**  
  📄 [Challenge Report (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Ancuti_NTIRE_2024_Dense_and_Non-Homogeneous_Dehazing_Challenge_Report_CVPRW_2024_paper.html)  
  → Search for team name: **"PSU Team"**

- 🔗 **Results on SateHaze1k Dataset**: [View Folder](https://drive.google.com/drive/folders/1kLl5qYFmgxtLqFOZAAauACP42ObWTm16?usp=sharing)  
- 📄 Related paper: [WACV 2020 SateHaze1k Dataset](https://openaccess.thecvf.com/content_WACV_2020/html/Huang_Single_Satellite_Optical_Imagery_Dehazing_using_SAR_Image_Prior_Based_WACV_2020_paper.html)  


<p align="center">
  <img src="https://github.com/AnasHXH/Dehazing-DiffGAN/blob/main/table_1.png" width="800"/>
  <br><em>Figure 1: Comparative results on SateHaze1k (Thin and Moderate Fog).</em>
</p>


<p align="center">
  <img src="https://github.com/AnasHXH/Dehazing-DiffGAN/blob/main/table_2.png" width="800"/>
  <br><em>Figure 1: Comparative results on SateHaze1k (Thin and Moderate Fog).</em>
</p>




  
📷 The following figures illustrate qualitative results of Dehazing-DiffGAN under thin, moderate, and dense haze conditions, demonstrating its effectiveness in restoring image clarity across varying atmospheric scenarios.

<p align="center">
  <img src="https://github.com/AnasHXH/Dehazing-DiffGAN/blob/main/thin.jpg" width="800"/>
  <br><em>Figure 1: Comparative results on SateHaze1k (Thin and Moderate Fog).</em>
</p>

<p align="center">
  <img src="https://github.com/AnasHXH/Dehazing-DiffGAN/blob/main/mod.jpg" width="800"/>
  <br><em>Figure 2: Additional moderate fog comparison results.</em>
</p>

<p align="center">
  <img src="https://github.com/AnasHXH/Dehazing-DiffGAN/blob/main/ntire.jpg" width="800"/>
  <br><em>Figure 3: Results on DNH-HAZE dataset (real-world haze).</em>
</p>


## Datasets
This work utilizes two remote sensing image datasets:

<ul> <li> <a href="https://codalab.lisn.upsaclay.fr/competitions/17529#learn_the_details" target="_blank"><strong>DNH-HAZE Dataset</strong></a> — A high-resolution real-world haze dataset used in the NTIRE dehazing challenges, featuring 6000×4000 images captured under various natural haze conditions. </li> <li> <a href="https://drive.google.com/drive/folders/1kLl5qYFmgxtLqFOZAAauACP42ObWTm16" target="_blank"><strong>SateHaze1k Dataset</strong></a> — A remote sensing haze dataset curated for evaluating dehazing performance on satellite images under thin, moderate, and thick fog levels. </li> </ul>

## 🚀 Usage

### 🔍 Inference

You can easily run inference on sample images using our provided Colab notebook or local script.

- 📓 **Colab Notebook**:  
  Run Dehazing-DiffGAN directly in the browser with Google Colab:  
  [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">](https://colab.research.google.com/drive/179lOH2iVtM-VRuSn6KNoUl-TbEFjAet6?usp=sharing)

- 🧪 **Local Notebook**:  
  We also provide a simple and clean Jupyter notebook for local inference:  
  `easy_inference.ipynb`

> Both notebooks allow you to load a pre-trained model and perform dehazing on your own images or samples from our datasets.

---

### 🏋️‍♂️ Training

> 🔧 **Coming soon!**  
Training scripts and configuration options will be released in an upcoming update.

---

### 📈 Testing and Evaluation

> 📊 **Coming soon!**  
We are preparing a full evaluation pipeline for reproducing results on the DNH-HAZE and SateHaze1k datasets.



## 📚 Citation

If you use any part of this work, please cite it using the following BibTeX format:

```
@misc{anas2024dehazingdiffgan,
title = {D},
author = {A},
year = {2024},
howpublished = {\url{https://github.com/AnasHXH/Dehazing-DiffGAN}},
note = {Under Review}
}
