# Code corresponding to the paper "Patient Identification Based on Deep Metric Learning for Preventing Human Errors in Follow-up X-Ray Examinations"

This repository provides the necessary parts to reproduce the results of our paper. In particular, this repository contains the code used to train with ChestXray8 and evaluate both the patient verification and identification with CheXpert or PadChest.

The figure below illustrates the general problem scenario: DL-based patient verification and re-identification approaches could allow a potential attacker to link sensitive information from a public dataset to an image of interest, highlighting the enormous data security and data privacy issues involved.

# Overview
The overall project is training, two tests (patient verification and patient identification)

* Patient verification (1: 1 comparison for wheather the patient-pair is the same or not)
* Patient identification (1: N comparison for whether the same patient' other image is the top-1-ranked or not)

# Setup
## Requirement libraries
* Python
* PyTorch
* timm
* pandas
* socket
* datetime
* pillow
* shutil
* scilit-learn
* corrections
## ENviroment
Tested in Windows 11 + Intel i9-10900X CPU + Nvidia GeForce 3090Ti with Cuda 11.7 and CuDNN.

## Datasets
* ChestXray8 [https://nihcc.app.box.com/v/ChestXray-NIHCC/]
* CheXpert [https://stanfordmlgroup.github.io/competitions/chexpert/]
* PadChest [https://bimcv.cipf.es/bimcv-projects/padchest/]

## Directory
<pre>
.
├── train.py
├── padchest_verification_identification.py
├── chexpert_verification.py
├── cnn.py
├── sam.py [from (https://github.com/davda54/sam)]
├── metrics.py [from (https://github.com/4uiiurz1/pytorch-adacos)]
├── utility ─── log.py
|           ├── bypass_bn.py [from (https://github.com/davda54/sam)]
|           ├── initialize.py [from (https://github.com/davda54/sam)]
|           └── loading_bar.py [from (https://github.com/davda54/sam)]
└── images  ─── CXR8.csv
            ├── CXR8 ─── 00000001_000.png
            |        ├── 00000001_001.png
            |        ├── 00000001_002.png
            |        ... [from (https://nihcc.app.box.com/v/ChestXray-NIHCC/)]
            ├── PadChest.csv
            ├── PadChest ─── 0 ─── 100069103068753688347522093561206841448_7197k3.png
            |             |    ├── 100081820231385537397079729591266436694_8o3uj2.png
            |             |    ├── 100360992970443012139948853258191567510_orx7ef.png
            |             |    ...
            |            ─── 1 ─── 100035238701184647172015593785663345624_vb6v1o.png
            |                  ... [from (https://bimcv.cipf.es/bimcv-projects/padchest/)]
            ├── CheXpert.txt
            ├── CheXpert ─── Small ─── train ─── patient00001 ─── ...
                                   └── valid ─── patient64541 ─── ...
                                   [from (https://stanfordmlgroup.github.io/competitions/chexpert/)]
</pre>

-- 
   
   
   
   
   
   

# Metric Learning - AdaCos
https://github.com/4uiiurz1/pytorch-adacos
## Requirement .py
* metrics.py

# Optimizer - ASAM
https://github.com/davda54/sam
## Requirement .py
* sam.py
* utility/log.py [corrected in this repository]
* utility/bypass_bn.py
* utility/initialize.py
* utility/loading_bar.py

## Notice
[log.py] have some bugs in our enviroment. So we have been fixed their bugs in our proposed method as follows:
* Line  8 - self.best_each = False
* Line 11 - self.epoch = -1 #(original: self.epoch = initial_epoch)
* Line 12 - self.initial_epoch = initial_epoch
* Line 42 - self.initial_epoch+self.epoch:12d #(original: self.epoch:12d)
* Line 56 - self.best_each = True
* Line 75 - self.initial_epoch+self.epoch:12d #(original: self.epoch:12d)



