# Clinically-Guided Multi-Phase Fusion Network for Liver Tumor Segmentation  


## Overview

This repository provides the official implementation of **OurMethod**, a clinically-guided multi-phase fusion network designed for accurate liver tumor segmentation from contrast-enhanced CT (CECT).  
Unlike existing fusion strategies that treat phases equally, OurMethod leverages the **clinical dominance of the portal venous (PV) phase** and explicitly models phase-specific propagation orders to achieve more reliable multi-phase feature integration.  

Extensive experiments on **PLC-CECT** and **MPLL** demonstrate that OurMethod consistently outperforms prior approaches (e.g., MW-UNet, MCDA-Net), setting a new state-of-the-art in both segmentation accuracy and boundary precision.

## Datasets  

| Dataset                                          | Phases                |                     Disease types |
| -------------------------------------------------| --------------------- | --------------------------------- | 
| [PLC-CECT](https://github.com/ljwa2323/PLC_CECT) | Multi (NC/ART/PV/DL)  | 278 patients / 152,965 slices     | 
| MPLL                                             | Multi (ART/PV/DL)     | 141 patients / 952,601 slices     | 

<img src="" width="500">

## 👉 Why Multi-Phase CECT?

Contrast-enhanced CT captures dynamic enhancement patterns via multiple phases:  
- **NC (non-contrast):** baseline anatomy  
- **ART (arterial):** highlights early vascular supply  
- **PV (portal venous):** best lesion–parenchyma contrast  
- **DL (delayed):** reveals washout and boundary refinement 

These phases are complementary, making multi-phase fusion a powerful strategy for robust lesion segmentation.

## Limitations of Existing Fusion Methods
- **Input-level fusion:** simple concatenation, ignores phase importance  
- **Feature-level fusion:** self-attention, but equal weighting across phases  
- **Decision-level fusion:** ensemble of outputs, but lacks inter-phase guidance  

➡️ A key drawback: they treat all phases equally, ignoring **clinical hierarchy (PV > ART > DL)**.  

## Our Contributions
- **Systematic single-phase analysis:** On MPLL, PV phase delivers the strongest segmentation performance, consistent with its clinical reliability.  
- **Clinically-guided propagation order:** We design a Multi-Phase Cross-Query Sequential (MCQS) branch. Ablation (Table 3) shows that **PV→ART→DL** order consistently achieves the best performance (76.29% DSC, 61.67% Jaccard, lowest HD$_{95}$ and ASSD).  
- **Proposed network:** A fusion architecture that integrates ART, PV, and DL with guided feature interaction and deep refinement.  
- **Novel loss function (BED-Loss):** Incorporates boundary-aware supervision for sharper tumor delineation.

## Results
- **On PLC-CECT:** OurMethod achieves 76.26% DSC (+1.09%), 62.44% Jaccard (+2.22%), 24.63 HD$_{95}$ (↓4.59), and 14.67 ASSD (↓2.08), surpassing MCDA-Net and MW-UNet.  
- **On MPLL:** OurMethod sets new SOTA with 76.29% DSC and 61.67% Jaccard, while also reducing boundary errors.  
- **Qualitative analysis:** Compared to MCDA-Net, OurMethod yields sharper boundaries (e.g., case 1) and more accurate detection of small nodules (e.g., case 3).  

Extensive experiments on two benchmark datasets, LiTS2017 and MPLL, demonstrate the superiority of our proposed method, which significantly outperforms existing state-of-the-art approaches.

## Usage

##  一、Multi-Phase Experiments

### Pre-trained Weights  

The weights of the pre-trained MADF-Net in 1P、2P、3P comparative analysis could be downloaded [Here](https://drive.google.com/drive/folders/1FSgOOqEkdjfBTvYudSf9NAxIwG3CxWxW?usp=drive_link)  

### Before Experiments：Create your conda environment

1、environments:Linux 5.4.0

2、Create a virtual environment: conda create -n environment_name python=3.8 -y and conda activate environment_name.

3、Install Pytorch : pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117

4、Requirements:
numpy==1.14.2
torch==1.0.1.post2
visdom==0.1.8.8
pandas==0.23.3
scipy==1.0.0
tqdm==4.40.2
scikit-image==0.13.1
SimpleITK==1.0.1
pydensecrf==1.0rc3

### 1、Pre-process 

1.1  First run ./data_prepare/split.py for Data partition.

1.2  run ./data_prepare/generate_2D_train.py and data_prepare/generate_2D_test.py for period data processing, then you can see the result in ./processed/train and ./processed/test

### 2、Generate distance map

- 2.1  Run boundary_map/liver_distance_map.py and boundary_map/tumor_distance_map.py to generate the boundary maps for liver and tumor, respectively.

- 2.2  Using the ./dataset/dataset_multiphase.py loader if you want to train without loading the distance map, or the dataset/dataset_multiphase_boundarymap.py loader if you want to load the distance map during training.

### 3、Training Process

3.1  The model is trained by running ./bash/train_multiphase.sh (You can modify the hyperparameters as prompted.), and the weights of its runs are stored in the model_out folder. If using BED-Loss training, the initial weights of BED-Loss in Eq. (8) are set to \(\alpha\)=0.49, \(\beta \)=0.49 and \(\gamma\)=0.02. If the training loss plateaus for 10 epochs, the weights are dynamically adjusted to a 4:4:2 ratio, and you can download the model weights from the Google Drive link above, and if the link is broken, you can contact the corresponding author to obtain and update the URL.

### 4、Evalution

4.1  Run ./bash/evaluate.sh, replacing the training weights and test data addresses in evaluate.sh. The test results will be saved in the model_out folder for viewing.

##  二、Singe-Phase Experiments

### 1、Data-Preparation & Pre-process 

You can jump to the download link of the LiTS2017 dataset according to the link in the dataset introduction, after downloading the datasets, then put the original image and mask in ./LiTS2017/data/ct and ./LiTS2017/data/label. Then you can divide it according to a certain ratio, run ./LiTS2017/data_prepare/preprocess_lits2017_png.py to convert .nii files into .png files for training. The file structure is as follows

- './LiTS2017/data_prepare/'
  - preprocess_lits2017_png.py
- './LiTS2017/data/'
  - LITS2017
    - ct
      - .nii
    - label
      - .nii
  - trainImage_lits2017_png
    - .png
  - trainMask_lits2017_png
    - .png

### 2、Generate distance map

- 2.1  Run boundary_map/liver_distance_map.py and boundary_map/tumor_distance_map.py to generate the boundary maps for liver and tumor, respectively.

- 2.2  You can modify the ./LiTS2017/dataset/dataset.py data loader to decide whether to add liver or tumor distance map.

### 3、Training Process

3.1  The model is trained by running ./LiTS2017/train/train.py (You can modify the hyperparameters as prompted.), and the weights of its runs are stored. If using BED-Loss training, the initial weights of BED-Loss in Eq. (8) are set to \(\alpha\)=0.49, \(\beta \)=0.49 and \(\gamma\)=0.02. If the training loss plateaus for 10 epochs, the weights are dynamically adjusted to a 4:4:2 ratio, and you can download the model weights from the Google Drive link above, and if the link is broken, you can contact the corresponding author to obtain and update the URL.

### 4、Evalution

4.1  Run ./LiTS2017/test/test.py, replacing the training weights and test data addresses in evaluate.sh. The test results will be saved.

