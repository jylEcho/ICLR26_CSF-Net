# MADF-Net: Multi-phase Attentional Deep Fusion Network for Liver Tumor Segmentation  

### Pre-trained Weights  

The weights of the pre-trained MADF-Net in 1P、2P、3P comparative analysis could be downloaded [Here](https://drive.google.com/drive/folders/1FSgOOqEkdjfBTvYudSf9NAxIwG3CxWxW?usp=drive_link)  

## Datasets  

| Dataset                                                      | Phases            | Samples | Disease types |
| ------------------------------------------------------------ | ----------------- | ------- | ------------------ | ------------------- |
| [PLC-CECT](https://github.com/ljwa2323/PLC_CECT) | Multi (NC/ART/PV/DL)  | 278 patients / 152,965 slices     |  |
| MPLL                                             | Multi (ART/PV/DL)     | 141 patients / 952,601 slices     | |

<img src="https://github.com/jylEcho/MADF-Net/blob/main/image/Dataset.png?raw=true" width="500">

## Contrast-Enhanced CT (CECT)： 

Contrast-enhanced CT (CECT), which captures dynamic tissue attenuation changes through contrast agent administration at different time points, offers a more informative alternative. It typically includes non-contrast (NC), arterial (ART), portal venous (PV), and delayed (DL) phases. These phases provide complementary information such as early vascular features, clear hepatic parenchyma structure, hyper-perfused regions, and delayed enhancement or washout effects, all of which help delineate lesion boundaries and improve segmentation accuracy. 

The complementary nature of these phases presents a valuable opportunity to improve segmentation performance through multi-phase fusion. 

## Existing Fusion Method： 

- **Input-Level Fusion: Concatenates arterial (ART), portal venous (PV), and delayed (DL) phase CT images.**

- **Feature-Level Fusion: Employs self-attention to dynamically weight phase-specific features.**

- **Decision-Level Fusion: Fuses predictions from individual phases and the fusion branch.**

## The existing fusion methods's shortcoming：

Treating each phase equally during fusion, failing to account for their clinical significance and complementary properties. This results in suboptimal performance, especially in scenarios with blurred lesion boundaries or small tumors.

## We achieved:

- **Conducts a systematic analysis of the segmentation performance of different contrast-enhanced CT phases using a clinical multi-phase liver tumor dataset (MPLL) collected from the First Affiliated Hospital of USTC. The results show that the PV phase contributes most significantly to segmentation performance, underscoring its importance both clinically and empirically.**

- **Based on these findings, we propose MADF-Net, a multi-phase attention-based fusion network that integrates features from the ART, PV, and DL phases, enabling deep inter-phase feature interaction across multiple stages to enhance segmentation performance.**

- **We design a novel dynamically weighted loss function, BED-Loss, which integrates regional and boundary information to improve the model’s sensitivity to tumor contours.**

Extensive experiments on two benchmark datasets, LiTS2017 and MPLL, demonstrate the superiority of our proposed method, which significantly outperforms existing state-of-the-art approaches.

## Experiments：Single-Phase & Multi-Phase

- **一、 Multi-Phase Experiments：In the MPLL folder**
- **二、Single-Phase Experiments：In the LiTS2017 folder**

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

