# Adversial-Attacks-on-densely-fused-point-clouds-for-6D-Pose-Estimation

# 11785-CMU---FCOS-with-MLFPN
This project integrates an anchor-free object detector (FCOS) with a multi-level feature pyramid network, which contains richer semantic features for object detection.

## Presentation
Please find the link to our presentation here : https://drive.google.com/file/d/1deIgrSBCUIRI16VfS0DeMjmGBm-jL0vQ/view?usp=sharing

## Background
- FCOS[1] is an anchor-free object detector that uses feature pyramid netword (FPN) to extract semantic features. Compared with anchor-based detectors, which uses a set of predefined bounding boxes to capture objects, anchor-free detectors directly predict bounding boxes around the centres of potential objects. FPN were created solely with the purpose of making detection models scale invariant. It creates a hierarchical structure containing semantic features at various levels, with high level features in charge of bigger objects and small ones in charge of smaller objects.
<div align="center">
  <img src="images/FPN_illustration.png" width="500"/>
</div>

- Multi-Level Feature Pyramid (MLFPN)[2] is a more complicated feature extractor. Instead of using a single FPN, MLFPN feeds the base feature into a block of alternating joint Thinned U-shape Modules (TUM) and Feature Fusion Modules (FFM) to extract more representative, multi-level multi-scale features. Finally feature maps with equivalent scales are gathered up, forming the final feature pyramid for object detection. 

## Introduction
In this project we integrated FCOS feature detector with MLFPN. FCOS has shown remarkable efficiency and adaptability for object detection, while MLFPN demonstrates excellent performance in feature extraction. Thus, we proposed our FCOS-MLFPN architecture in order to boost FCOS's performance.
<div align="center">
  <img src="images/FCOS_MLFPN_Architecture.png" width="900"/>
</div>

## Installation
#### Download project files
```
git clone https://github.com/shayeree96/11785-CMU---FCOS-with-MLFPN
```
#### Set up environment
```
conda env create -f environment.yaml`
pip install -r requirements.txt
```
#### Download VOC dataset
Navigate to the project root directory

Make a data folder:
```
mkdir data
cd data/
```
Download tar files of VOC and extract them:
```
wget http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
tar xvf VOCtrainval_11-May-2012.tar
tar xvf VOCdevkit_18-May-2011.tar
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```
If you encounted this error:
```
Connecting to host.robots.ox.ac.uk (host.robots.ox.ac.uk)|129.67.94.152|:8080... failed: Connection refused.
``` 
Run the following commands:
```
wget http://pjreddie.com/media/files/VOC2012test.tar
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xvf VOC2012test.tar
tar xvf VOCtrainval_11-May-2012.tar
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
```
## Training
Navigate to the project root directory

Run:
```
python3 train_voc.py
```
For specifying parameters:
```
python3 train_voc.py --epochs your_number_of_epochs --batch_size your_batch_size --n_cpu your_number_of_threads --n_gpu you_number_of_gpus
```
By default, 
```
epochs = 30
batch_size = 32
n_cpu = 36
n_gpu = "0,1,2,3,4,5,6,7"
```
## Evaluation
Navigate to the project root directory

Run:
```
python3 eval_voc.py
```
<div align="center">
  <img src="images/22_new.png" width="800"/>
</div>

## Reference
[1] Tian, Z., Shen, C., Chen, H., and He, T., “FCOS: Fully Convolutional One-Stage Object
Detection”, <i>arXiv e-prints</i>, 2019.

[2] Q. Zhao, T. Sheng, Y. Wang, Z. Tang, Y. Chen, L. Cai, and H. Ling. M2det: A single-shot object
detector based on multi-level feature pyramid network. AAAI, 2019.
## Acknowledgements
Our project is based on the following two github repositories:

https://github.com/zhenghao977/FCOS-PyTorch-37.2AP

https://github.com/qijiezhao/M2Det

We would like to thank the contributors for providing us code resources
## Contributors
Shayeree Sarkar [@shayeree96](https://github.com/shayeree96)

Baishali Mullick [@bm-93](https://github.com/bm-93)

Zhenwei Liu [@zhenweil](https://github.com/zhenweil)

Dhruv Vashisht [@dhruvv301292](https://github.com/dhruvv301292)
## License
For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors.
