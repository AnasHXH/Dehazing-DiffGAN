# Overview
This repository is an implementation of "Dehazing-DiffGAN: Sequential Fusion of Diffusion Models and GANs for High-Fidelity Remote Sensing Image Dehazing" paper. 

# Abstract
Atmospheric haze reduces image clarity, posing challenges for computer vision. Current methods often fail under dense haze, producing blurry or unrealistic results. To tackle this, we introduce Dehazing-DiffGAN, a two-stage deep learning model combining diffusion models and GANs. The first stage uses a diffusion process to remove haze and recover global and local features, while the second stage refines visual details using a GAN. This fusion produces high-quality, realistic outputs. Evaluated on tough datasets like SateHaze1k and DNH-HAZE, our method outperforms state-of-the-art models, achieving superior PSNR and LPIPS scores. Dehazing-DiffGAN proves to be a robust and effective solution for remote sensing image enhancement under challenging conditions.

## 📊 Results

In this repository, we conducted an extensive evaluation of the proposed **Dehazing-DiffGAN** by benchmarking it against more than fifteen cutting-edge dehazing techniques. This comparison includes several well-established models such as DCP, AOD-Net, FCTF-Net, PFF-Net, GridDehaze-Net, FFA-Net, SCA-Net, AIDTransformer, EDED-Net, ConvIR-S, ADND-Net, PhDNet, and HSMD-Net, among others, as presented in Tables 1 and 2. The comparative analysis was carried out using a range of quantitative metrics to rigorously assess each model's effectiveness in restoring image clarity, preserving structural details, and enhancing perceptual quality under various haze conditions.

👉 The following two tables present a comprehensive comparison of Dehazing-DiffGAN against these models.  
- **Table 1** includes methods evaluated using **three metrics**: PSNR, SSIM, and LPIPS — offering detailed insights into both pixel-level fidelity and perceptual similarity.  
- **Table 2** includes models that report results using **only PSNR and SSIM**, and also includes the number of model parameters (in millions), offering a trade-off between performance and complexity.


## 📂 Dataset Result Visualizations

- 🔗 **Results on DNH-HAZE Dataset**: [View Folder](https://drive.google.com/drive/folders/1dT_7bAajfdD9oXRbTqBJX-aZFt1tIgaB?usp=sharing)  
- 🏆 Our Dehazing-DiffGAN achieved **best LPIPS** in the **CVPR NTIRE 2024 Dense and Non-Homogeneous Dehazing Challenge**  
  📄 [Challenge Report (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Ancuti_NTIRE_2024_Dense_and_Non-Homogeneous_Dehazing_Challenge_Report_CVPRW_2024_paper.html)  
  → Search for team name: **"PSU Team"**

- 🔗 **Results on SateHaze1k Dataset**: [View Folder](https://drive.google.com/drive/folders/1kLl5qYFmgxtLqFOZAAauACP42ObWTm16?usp=sharing)  
- 📄 Related paper: [WACV 2020 SateHaze1k Dataset](https://openaccess.thecvf.com/content_WACV_2020/html/Huang_Single_Satellite_Optical_Imagery_Dehazing_using_SAR_Image_Prior_Based_WACV_2020_paper.html)  


<table>
<thead>
<tr>
<th>Model</th><th>Year</th>
<th colspan="3">Thin Fog</th>
<th colspan="3">Moderate Fog</th>
<th colspan="3">Thick Fog</th>
<th colspan="3">Average</th>
</tr>
<tr>
<th></th><th></th>
<th>PSNR</th><th>SSIM</th><th>LPIPS</th>
<th>PSNR</th><th>SSIM</th><th>LPIPS</th>
<th>PSNR</th><th>SSIM</th><th>LPIPS</th>
<th>PSNR</th><th>SSIM</th><th>LPIPS</th>
</tr>
</thead>
<tbody>
<tr><td>DCP</td><td>2010</td><td>15.99</td><td>0.835</td><td>0.10</td><td>14.75</td><td>0.821</td><td>0.13</td><td>10.99</td><td>0.617</td><td>0.27</td><td>13.91</td><td>0.758</td><td>0.17</td></tr>
<tr><td>AOD-Net</td><td>2017</td><td>18.47</td><td>0.863</td><td>0.08</td><td>17.63</td><td>0.855</td><td>0.12</td><td>15.75</td><td>0.742</td><td>0.23</td><td>17.28</td><td>0.820</td><td>0.14</td></tr>
<tr><td>FCTF-Net</td><td>2020</td><td>19.71</td><td>0.875</td><td>0.10</td><td>23.10</td><td>0.926</td><td>0.06</td><td>18.56</td><td>0.800</td><td>0.20</td><td>20.46</td><td>0.867</td><td>0.12</td></tr>
<tr><td>PFF-Net</td><td>2019</td><td>16.01</td><td>0.821</td><td>0.17</td><td>18.59</td><td>0.688</td><td>0.49</td><td>16.06</td><td>0.575</td><td>0.61</td><td>16.89</td><td>0.695</td><td>0.42</td></tr>
<tr><td>GridDehaze-Net</td><td>2019</td><td>19.36</td><td>0.857</td><td>0.09</td><td>21.91</td><td>0.905</td><td>0.07</td><td>17.83</td><td>0.773</td><td>0.20</td><td>19.70</td><td>0.845</td><td>0.12</td></tr>
<tr><td>FFA-Net</td><td>2020</td><td>24.26</td><td>0.910</td><td>0.06</td><td>25.39</td><td>0.930</td><td>0.08</td><td>21.83</td><td>0.836</td><td>0.16</td><td>23.83</td><td>0.892</td><td>0.10</td></tr>
<tr><td>SCA-Net</td><td>2023</td><td>19.70</td><td>0.882</td><td>0.06</td><td>24.75</td><td>0.934</td><td>0.05</td><td>18.40</td><td>0.812</td><td>0.13</td><td>20.95</td><td>0.876</td><td>0.08</td></tr>
<tr><td>AIDTransformer</td><td>2023</td><td>24.82</td><td>0.904</td><td>0.048</td><td>27.20</td><td>0.918</td><td>0.039</td><td>22.21</td><td>0.831</td><td>0.107</td><td>24.74</td><td>0.884</td><td>0.065</td></tr>
<tr><td>EDED-Net</td><td>2024</td><td>24.81</td><td><b>0.924</b></td><td>0.05</td><td>25.65</td><td><b>0.939</b></td><td>0.05</td><td>22.46</td><td>0.857</td><td>0.13</td><td>24.31</td><td><b>0.907</b></td><td>0.08</td></tr>
<tr><td>PGSformer</td><td>2024</td><td>25.53</td><td>0.9176</td><td>0.0660</td><td>26.62</td><td>0.9328</td><td>0.0665</td><td>23.59</td><td><b>0.8628</b></td><td>0.1378</td><td>25.25</td><td>0.9044</td><td>0.0901</td></tr>
<tr><td><b>Dehazing-DiffGAN</b></td><td>-</td><td><b>27.084</b></td><td>0.8863</td><td><b>0.0413</b></td><td><b>30.09</b></td><td>0.9247</td><td><b>0.0315</b></td><td><b>24.45</b></td><td>0.8125</td><td><b>0.0689</b></td><td><b>27.21</b></td><td>0.8745</td><td><b>0.0472</b></td></tr>
</tbody>
</table>


<table>
<thead>
<tr>
<th>Model</th><th>Year</th><th>Params (M)</th>
<th colspan="2">Thin Fog</th>
<th colspan="2">Moderate Fog</th>
<th colspan="2">Thick Fog</th>
<th colspan="2">Average</th>
</tr>
<tr>
<th></th><th></th><th></th>
<th>PSNR</th><th>SSIM</th>
<th>PSNR</th><th>SSIM</th>
<th>PSNR</th><th>SSIM</th>
<th>PSNR</th><th>SSIM</th>
</tr>
</thead>
<tbody>
<tr><td>cGAN</td><td>2020</td><td>-</td><td>24.164</td><td>0.906</td><td>25.311</td><td>0.9264</td><td>25.073</td><td>0.864</td><td>24.849</td><td>0.8988</td></tr>
<tr><td>UMWTransformer</td><td>2022</td><td>4.5</td><td>24.29</td><td>0.919</td><td>26.65</td><td>0.946</td><td>20.07</td><td>0.825</td><td>23.67</td><td>0.8967</td></tr>
<tr><td>FocalNet</td><td>2023</td><td><b>3.74</b></td><td>24.16</td><td>0.916</td><td>25.99</td><td>0.947</td><td>21.69</td><td>0.847</td><td>23.947</td><td>0.9033</td></tr>
<tr><td>PSMB-Net</td><td>2023</td><td>12.45</td><td>26.75</td><td>0.928</td><td>27.48</td><td>0.946</td><td>25.15</td><td>0.889</td><td>26.46</td><td>0.921</td></tr>
<tr><td>ARDD-Net</td><td>2023</td><td>-</td><td>26.84</td><td>0.9257</td><td>26.47</td><td>0.9321</td><td>26.83</td><td>0.9316</td><td>26.71</td><td>0.930</td></tr>
<tr><td>ConvIR-S</td><td>2024</td><td>14.83</td><td>25.11</td><td><b>0.978</b></td><td>26.79</td><td><b>0.978</b></td><td>22.65</td><td><b>0.950</b></td><td>24.85</td><td><b>0.9687</b></td></tr>
<tr><td>ADND-Net</td><td>2024</td><td>160.14</td><td>26.91</td><td>0.9274</td><td>26.67</td><td>0.9358</td><td><b>26.94</b></td><td>0.9358</td><td>26.84</td><td>0.933</td></tr>
<tr><td>PhDnet</td><td>2024</td><td>10.03</td><td>23.13</td><td>0.8969</td><td>26.47</td><td>0.9421</td><td>19.251</td><td>0.8069</td><td>22.95</td><td>0.882</td></tr>
<tr><td>HSMD-Net</td><td>2024</td><td>9.91</td><td>26.91</td><td>0.928</td><td>28.07</td><td>0.957</td><td>24.81</td><td>0.880</td><td>26.597</td><td>0.919</td></tr>
<tr><td>SCSNet</td><td>2024</td><td>-</td><td>26.146</td><td>0.9415</td><td>28.350</td><td>0.9566</td><td>24.654</td><td>0.9015</td><td>26.383</td><td>0.936</td></tr>
<tr><td>RSDehamba</td><td>2024</td><td>1.80</td><td>26.750</td><td>0.9306</td><td>27.45</td><td>0.9468</td><td>23.53</td><td>0.8698</td><td>25.91</td><td>0.9157</td></tr>
<tr><td><b>Dehazing-DiffGAN</b></td><td>-</td><td>161</td><td><b>27.08</b></td><td>0.8863</td><td><b>30.09</b></td><td>0.9247</td><td>24.45</td><td>0.8125</td><td><b>27.21</b></td><td>0.8745</td></tr>
</tbody>
</table>




  
The figure below details the performances of the proposed weight initialization method on four public remote senging datasets, namely, UC-Merced, AID, KSA, and PatternNet.

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

ul> <li> <a href="https://codalab.lisn.upsaclay.fr/competitions/17529#learn_the_details" target="_blank"><strong>DNH-HAZE Dataset</strong></a> — A high-resolution real-world haze dataset used in the NTIRE dehazing challenges, featuring 6000×4000 images captured under various natural haze conditions. </li> <li> <a href="https://drive.google.com/drive/folders/1kLl5qYFmgxtLqFOZAAauACP42ObWTm16" target="_blank"><strong>SateHaze1k Dataset</strong></a> — A remote sensing haze dataset curated for evaluating dehazing performance on satellite images under thin, moderate, and thick fog levels. </li> </ul>

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
title = {Dehazing-DiffGAN: Sequential Fusion of Diffusion Models and GANs for High-Fidelity Remote Sensing Image Dehazing},
author = {Anas M. Ali and Bilel Benjdira and Wadii Boulila and Anis Koubaa},
year = {2024},
howpublished = {\url{https://github.com/AnasHXH/Dehazing-DiffGAN}},
note = {Under Review}
}
