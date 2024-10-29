# AFAS-YOLOv8
 <p align="justify"> 
Welcome to the official repository for "Addressing False Alarms from High-Voltage Structures in Subpixel Fire Detection" (currently under review). This repository contains all necessary resources to replicate the experiments presented in our study.
</p>

<p align="justify"> 
Note that you will need to <a href="https://docs.ultralytics.com/modes/train/" target="_blank">perform training</a> using a pre-trained YOLOv8 model to run this code effectively.
</p>

## 📂 Dataset Access and Usage

<p align="justify"> 
The datasets used in this work include the FLIR ADAS, $M{^3}FD$, and Powerline Image Dataset (PID) image datasets, which are all open-access for research purposes. Links to download these datasets are available in the paper, and they must be downloaded separately as required by the project. Each image annotation follows its original naming convention, with an identifier added post-name to maintain consistency.
</p>

### Terms and Conditions
* FLIR ADAS Dataset: Refer to the <a href="https://www.flir.com/oem/adas/adas-dataset-agree/" target="_blank">FLIR ADAS Terms of Use</a> for conditions on FLIR ADAS data usage.
* PID: The <a href="https://data.mendeley.com/datasets/n6wrv4ry6v/8" target="_blank">Powerline Image Dataset terms</a>  apply to PID usage.
* TarDAL ($M{^3}FD$) Dataset: Use of the $M{^3}FD$ dataset is subject to the <a href="https://github.com/JinyuanLiu-CV/TarDAL" target="_blank">TarDAL terms</a>.



## 💻 Materials
<p align="justify">
Our proposed dataset is available via our Mendeley data repository: <a href="https://data.mendeley.com/datasets/rng8d63pk3/1" target="_blank">AFAS-YOLOv8</a>. Please ensure the "original_data" folder is located within this repository before running the provided examples. Most data is stored in 16-bit format; to use it for training, apply the "normalize" method in utils.py to convert it to 8-bit.
</p>

## 🔧 Dependencies and Installation 
* Python == 3.10.8
* opencv-python-headless == 4.8.1.78
* numpy == 1.26.1
* matplotlib == 3.7.2
* ultralytics == 8.0.213
  
## 🚀 Code Overview
<p align="justify">
The augmentation process carried out in this work was possible using <a href="https://docs.roboflow.com/datasets/image-augmentation" target="_blank">Roboflow augmentation tool</a>. 

All functions developed for this project are organized in the code folder. Here’s a summary of key files:

* utils.py: This script includes essential utility methods, such as the Intersection over Union (IoU) calculation and preprocessing routines.
* segment_example.py: This example script demonstrates image segmentation. In the resulting segmentation, the color coding is as follows:
    * Red: Towers
    * Blue: Powerlines
    * Green: False Negatives
</p>


<p align="center" width="100%">
    <img width="100%" src="images/segment_example.svg"> 
</p>

##  BibTeX
<!-- @InProceedings{aa,
    author    = {aa},
    title     = {aa},
    date      = {2024}
} -->

## 📜 License
This project is released under the AGPL-3.0 license.

## 📧 Contact
If you have any questions, please email antonio.galvan@ulpgc.es.
