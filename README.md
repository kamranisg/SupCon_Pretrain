# Pretrain_Chest_X-rays
Pre-training for localised tasks on Medical Chest-rays

## About

Self-supervised learning has emerged as a powerful technique to understand non-curated datasets (i.e., without labels). In particular, contrastive learning has shown tremendous transfer learning capability in several downstream tasks such as Classification, segmentation and detection. In addition, we exploit the pre-training downstream. In this report, we evaluate three different scenarios involving pre-training and transfer learning to improve the localisation on chest X-rays in terms of bounding boxes.


## Architecture

### Stage-1 : Pre-training 
<img src = "https://github.com/kamranisg/pretrain_xrays/blob/main/Stage-1.png" height="250" width="600">

### Stage-2 : Downstream Training and Evaluation 
<img src = "https://github.com/kamranisg/pretrain_xrays/blob/main/Stage-2.png" height="250" width="600">

## Setup

### 1. Enviroment
Use the conda packages

``` terminal
conda env create -f gr2.yaml 
```


### 2. Datasets

#### MIMIC-CXR
 - Download the dataset MIMIC-CXR from https://physionet.org/content/mimic-cxr-jpg/2.0.0/ into your folder ```<path-to-datasets>/MIMIC-CXR```

#### RSNA-PNEUMONIA
 - Download the dataset RSNA from this kaggle repository
   https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data into your folder ```<path-to-datasets>/RSNA```

- Pre-process the dataset 

  ``` terminal 
  python src/data/datasets/mimic_cxr/mimic_cxr_dataset.py create_image_list --path <path_to_datasets>/MIMIC-CXR/mimic-cxr_ap-pa_dataset 
  ```
  
### 3. Stage-1 (Pre-training)

#### Backbone - 1: ImageNet
- For Backbone - 1 we do not need to pre-train our ResNet architecture, since we are using ResNet pre-trained on ImageNet weights. Skip to Stage-2 training.

#### Backbone -2: Cross Entropy
- Run the following script in the terminal
  
  ``` terminal 
  python main_ce.py 
  --batch_size=64 
  --epochs=100 
  --learning_rate=5e-4
  --size=224
  ```
  
#### Backbone -3: Supervised Contrastive 
- Run the following script in the terminal
  
  ``` terminal 
  python main_supcon.py 
  --batch_size=64 
  --epochs=100 
  --learning_rate=3e-5 
  ```

### 4. Stage-2 (Downstream training and Evalaution)

- Run the following script in the terminal
  
  ``` terminal 
  python RSNA_frozen.py 
  --batch_size=64 
  --epochs=100 
  --learning_rate=1e-3
  --backbone="ImageNet"
  --train_folder=<path_to_datasets>/RSNA/CSV/train.csv
  --val_folder=<path_to_datasets>/RSNA/CSV/validation.csv
  ```
  
  Additional Hyperparameters:
  If you wish to use 
  
- Validation on Only Bounding Boxes 
  
  ``` terminal 
  --val_folder=<path_to_datasets>/RSNA/CSV/dummy.csv
  ``` 
  
- Backbone trained using Cross Entropy 
  
  First, pick the best model from visualizing the loss functions in tensorboard. 
  
  Try different learning rates, batch sizes, and choose the best epoch. Usually the last epoch works best with lowest validation loss
  Go to `~/save/SupCon/mimic_cxr_model/<path-to-your-best-checkpoint>`
  
  Next, add the following command to the script (example)
  ``` terminal 
   --checkpoint="~/save/SupCon/mimic_cxr_model/SupCE_mimic_cxr_resnet50_lr_0.0005_decay_1e-06_bsz_64_trial_0/last.pth"
   --backbone="SupCE"
  ``` 
  
- Backbone trained using Supervised Contrastive loss
  
  Go to `~/save/SupCon/mimic_cxr_model/<path-to-your-best-checkpoint>`
  
  ``` terminal 
  --checkpoint="~/save/SupCon/mimic_cxr_model/SupCon_mimic_cxr_resnet50_lr_3e-05_decay_1e-06_bsz_64_temp_0.07_trial_0/last.pth"
  --backbone="SupCon"
  ``` 
  

## Configurations

Stage-1 (Pre_training) : We have used trained ResNet-50 on 20% training data of MIMIC_CXR

Stage-2 (Downstream Traning and Evaluation): 
  
| RSNA_Training_Set = 10%       | RSNA_Training_Set = 50%   |
| ------------- |:-------------:| 
| ImageNet + Validation = 20%      | ImageNet + Validation = 20% 
| ImageNet + Validation = Only Boxes      |  ImageNet + Validation = Only Boxes  |   
| SupCon + Validation = 20% | SupCon + Validation = 20%      |   
| SupCon + Validation = Only Boxes      |  SupCon + Validation = Only Boxes      | 
| Cross_Entropy + Validation = 20%      | Cross_Entropy + Validation = 20%  |   
| Cross_Entropy + Validation = Only_Boxes |    Cross_Entropy + Validation = Only_Boxes   |   
  
 


## Results
<img src = "https://github.com/kamranisg/pretrain_xrays/blob/main/10Per.png" height="350" width="600">
<img src = "https://github.com/kamranisg/pretrain_xrays/blob/main/50per.png" height="350" width="600">

