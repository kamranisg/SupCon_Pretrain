# Pretrain_Chest_X-rays
Pre-training for localised tasks on Medical Chest-rays

## About

Self-supervised learning has emerged as a powerful technique to understand non-curated datasets (i.e., without labels). In particular, contrastive learning has shown tremendous transfer learning capability in several downstream tasks such as Classification, segmentation and detection. In addition, we exploit the pre-training downstream. In this report, we evaluate three different scenarios involving pre-training and transfer learning to improve the localisation on chest X-rays in terms of bounding boxes.


## Architecture

### Stage-1 : Pre-training 
![Shot](/Stage-1.png)
### Stage-2 : Downstream Training and Evaluation 
![Shot](/Stage-2.png)

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
  ```


## Results
