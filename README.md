# Code corresponding to the paper "Patient Identification Based on Deep Metric Learning for Preventing Human Errors in Follow-up X-Ray Examinations"

This repository provides the necessary parts to reproduce the results of our paper. In particular, this repository contains the code used to train with ChestXray8 and evaluate both the patient verification and identification with CheXpert or PadChest.

The figure below illustrates the general problem scenario: DL-based patient verification and re-identification approaches could allow a potential attacker to link sensitive information from a public dataset to an image of interest, highlighting the enormous data security and data privacy issues involved.

# Overview
The overall project is training, two tests (patient verification and patient identification)

* Patient verification (1: 1 comparison for wheather the patient-pair is the same or not)
* Patient identification (1: N comparison for whether the same patient' other image is the top-1-ranked or not)

# Optimizer - ASAM
## Notice
[log.py] have some bugs in our enviroment. So we have been fixed their bugs in our proposed method as follows:
* Line 10 - self.epoch = -1 #(original: self.epoch = initial_epoch)
* Line 11 - self.initial_epoch = initial_epoch # add
* Line 42 - self.initial_epoch+self.epoch:12d #(original: self.epoch:12d)
* Line 75 - self.initial_epoch+self.epoch:12d #(original: self.epoch:12d)
