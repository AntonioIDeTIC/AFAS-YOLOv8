# AFAS-YOLOv8
 <p align="justify"> 
This is the official repository of "Addressing false alarms from high-voltage structures in subpixel fire detection" (recently submitted). In this repository, you can find all the resources necessary to replicate our work. Please, take into account that you need to train a pretrained YOLOv8 model to test this code.
</p>

<p align="justify"> 
It should be noted that the FLIR, $M{^3}FD$, and PID images are open access for research purposes and can be downloaded from the links provided in the paper. The annotations made respect the original name of these images, only an identifier has been added to each after the original name.
</p>

<p align="justify"> 
The original terms and conditions of the <a href="https://www.flir.com/oem/adas/adas-dataset-agree/" target="_blank">FLIR ADAS Terms of Use</a> apply to the FLIR ADAS dataset. 
</p>

<p align="justify"> 
The original terms and conditions of the <a href="https://data.mendeley.com/datasets/n6wrv4ry6v/8" target="_blank">Powerline Image Dataset </a> apply to the PID dataset. 
</p>

<p align="justify"> 
The original terms and conditions of the <a href="https://github.com/JinyuanLiu-CV/TarDAL" target="_blank">TarDAL</a> apply to the $M{^3}FD$ dataset. 
</p>


## 💻 Materials
<p align="justify">
Please visit our Mendeley data repository at <a href="https://data.mendeley.com/datasets/rng8d63pk3/1" target="_blank">AFAS-YOLOv8</a> to download the dataset discussed in our work. Take into account that the "original_data" folder must be inside this repository to run the examples we provided. The data is in its mayority 16bit format, use the "normalize" method inside utils.py to convert to 8bit data and train the model.
</p>

## 🔧 Dependencies and Installation 
* Python == 3.10.8
* opencv-python-headless == 4.8.1.78
* numpy == 1.26.1
* matplotlib == 3.7.2
* ultralytics == 8.0.213
  
## 🚀 Code
<p align="justify">
The functions developed in this work can be found in the code folder. The utils.py file implements basic methods such as the IoU calculation used in this work, as well as the preprocessing routine. 
</p>

<p align="justify">
The segment_example.py file shows an example of image segmentation. The result shows with a red color the class corresponding to towers, with a blue color the powerlines, and with a green color the false negatives.
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
