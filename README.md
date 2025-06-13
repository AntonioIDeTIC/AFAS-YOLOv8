# AFAS-YOLOv8
<p align="justify"> 
Welcome to the official repository for "Addressing False Alarms from High-Voltage Structures in Subpixel Fire Detection" (currently under review). This repository contains all the necessary resources to replicate the experiments presented in our study.
</p>

<p align="justify"> 
Note that you will need to <a href="https://docs.ultralytics.com/modes/train/" target="_blank">perform training</a> using a pre-trained YOLOv8 model to run this code effectively.
</p>

## 📂 Dataset Access and Usage

<p align="justify"> This study utilizes three open-access image datasets for research purposes: the FLIR ADAS, TarDAL M3FD, and Powerline Image Dataset (PID). 
To replicate this project's results, please download these datasets and merge them with the images and labels we provided. 
The paper provides references for downloading and citing the datasets. This research team's image annotations retain their original naming convention, with an additional identifier appended to ensure consistency.
</p>

### Terms and Conditions
* FLIR ADAS Dataset: Please refer to the <a href="https://www.flir.com/oem/adas/adas-dataset-agree/" target="_blank">FLIR ADAS Terms of Use</a> for conditions on using FLIR ADAS data.
* Powerline Image Dataset (PID): The <a href="https://data.mendeley.com/datasets/n6wrv4ry6v/8" target="_blank">Powerline Image Dataset Terms of Use</a> apply for this images and labels.
* TarDAL M3FD Dataset: Usage of the M3FD dataset is subject to the <a href="https://github.com/JinyuanLiu-CV/TarDAL" target="_blank">TarDAL Terms of Use</a>.
* DIC Dataset: The dataset captured by our team (images acronym: DIC) is available under the <a href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank">CC BY-NC 4.0 license Terms of Use</a>.


## 💻 Materials
<p align="justify"> Our proposed dataset is available in the Mendeley data repository: <a href="https://data.mendeley.com/datasets/rng8d63pk3/2" target="_blank">AFAS-YOLOv8</a>. 
Before running the provided examples, ensure that the "original_data" folder is included within this repository. Most of the data is stored in 16-bit format. 
To prepare it for training, please use the "normalize" method in <code>utils.py</code> to convert the data to an 8-bit format. 
</p> 

<p align="justify"> We have uploaded all images and labels for the DIC dataset, while only the labels for the other open-access datasets are included. 
Please refer to the original download links to obtain the respective images for these datasets. 
</p>

## 🔧 Dependencies and Installation 
* Python == 3.10.8
* opencv-python-headless == 4.8.1.78
* numpy == 1.26.1
* matplotlib == 3.7.2
* ultralytics == 8.0.213
  
## 🚀 Code Overview
<p align="justify">
The augmentation process carried out in this work was possible using <a href="https://docs.roboflow.com/datasets/image-augmentation" target="_blank">Roboflow augmentation tool</a>. Please, refer to its documentation to replicate the augmented dataset discussed in the paper if you want to train YOLOv8.

The methods developed for this project are organized in the code folder. Here’s a summary of key files:

* <code>utils.py</code>: This script includes essential utility methods, such as the Intersection over Union (IoU) calculation and preprocessing routines.
* <code>segment_example.py</code>: This example script demonstrates image segmentation. In the resulting segmentation, the color coding is as follows:
    * Red: Towers
    * Blue: Powerlines
    * Green: False Negatives
* <code>SubpixelFireGeneration.py</code>: Utility class to generate the synthetic fire.
* <code>replicate_synth_AFAS.py</code>: Use this script to generate the synthetic AFAS dataset.
</p>

<p align="center" width="100%">
    <img width="100%" src="images/segment_example.svg"> 
</p>

## Citation

> **Galván-Hernández, A.**, Araña-Pulido, V., Cabrera-Almeida, F., & Quintana-Morales, P. (2025).  
> Addressing false alarms from high-voltage structures in subpixel fire detection. *Engineering Applications of Artificial Intelligence*, 158, 111324.  
> [https://doi.org/10.1016/j.engappai.2025.111324](https://doi.org/10.1016/j.engappai.2025.111324)

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.engappai.2025.111324-blue)](https://doi.org/10.1016/j.engappai.2025.111324)

<details>
<summary>BibTeX</summary>

```bibtex
@article{GALVANHERNANDEZ2025111324,
  title   = {Addressing false alarms from high-voltage structures in subpixel fire detection},
  author  = {Galván-Hernández, Antonio and Araña-Pulido, Víctor and Cabrera-Almeida, Francisco and Quintana-Morales, Pedro},
  journal = {Engineering Applications of Artificial Intelligence},
  volume  = {158},
  pages   = {111324},
  year    = {2025},
  issn    = {0952-1976},
  doi     = {10.1016/j.engappai.2025.111324},
  url     = {https://www.sciencedirect.com/science/article/pii/S0952197625013260},
  keywords= {False alarm, High-voltage structure, Subpixel fire, Thermal image, You only look once}
}
</details>

## 📜 License
This project is released under the AGPL-3.0 license.

## 📧 Contact
If you have any questions, please email antonio.galvan@ulpgc.es.
