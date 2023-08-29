# Masking Strategies for Background Bias Removal in Computer Vision Models

Official implementation of the paper: Masking Strategies for Background Bias Removal in Computer Vision Models, accepted at ICCVW-OODCV 2023.

[[`Arxiv`]](https://arxiv.org/abs/2308.12127)

In our research, we focus on fine-grained image classification, such as identifying bird species from images. We identify a common issue where computer vision models learn to associate the species to their habitats (the image background), introducing bias. We propose some simple masking strategies to remove this bias.

![bg_problem drawio](https://github.com/ananthu-aniraj/masking_strategies_bias_removal/assets/50333505/feb600d7-2450-4903-a494-c7035affe095)


## Abstract

Models for fine-grained image classification tasks, where the difference between some classes can be extremely subtle and the number of samples per class tends to be low, are particularly prone to picking up background-related biases and demand robust methods to handle potential examples with out-of-distribution (OOD) backgrounds. To gain deeper insights into this critical problem, our research investigates the impact of background-induced bias on fine-grained image classification, evaluating standard backbone models such as Convolutional Neural Network (CNN) and Vision Transformers (ViT). We explore two masking strategies to mitigate background-induced bias: Early masking, which removes background information at the (input) image level, and late masking, which selectively masks high-level spatial features corresponding to the background. Extensive experiments assess the behavior of CNN and ViT models under different masking strategies, with a focus on their generalization to OOD backgrounds. The obtained findings demonstrate that both proposed strategies enhance OOD performance compared to the baseline models, with early masking consistently exhibiting the best OOD performance. Notably, a ViT variant employing GAP-Pooled Patch token-based classification combined with early masking achieves the highest OOD robustness.

![LM_EM_alternate drawio](https://github.com/ananthu-aniraj/masking_strategies_bias_removal/assets/50333505/462a5653-0e43-443f-836f-6fe6db09a723)


## Setup
Each folder essentially represents a subproject containing the Python code to run the experiments. The code is written in Python 3.11 and PyTorch 2.0.0 (newer versions will also work, please raise an issue if it doesn't work).

Use the "environment.yml" file to create a conda environment with all the required dependencies.
In case this doesn't work, please create a new issue. 

Alternatively, install PyTorch using instructions from [here](https://pytorch.org/get-started/locally/) and install MMSegmentation from [here - Case a](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation).

- The "early_masking_and_baseline" folder contains the code for the early masking and baseline experiments. 
- The "late_masking" folder contains the code for training and evaluating the late masking models from the paper.
- The "mmsegmentation" folder contains the code for training and evaluating the segmentation models from the paper.



## Datasets

### CUB-200-2011
We use the CUB-200-2011 dataset for training the models. The dataset can be downloaded from [here](https://www.vision.caltech.edu/datasets/cub_200_2011/). 

NOTE: Download "segmentations" if you intend to train/evaluate the segmentation models. Otherwise, just download the images and annotations.

For evaluation, we also use the out-of-distribution Waterbirds dataset, which can be downloaded from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz).

After downloading the datasets, the folder structure should look like this:

```
CUB_200_2011
├── attributes
├── bounding_boxes.txt
├── classes.txt
├── images
├── image_class_labels.txt
├── images.txt
├── parts
├── README
├── segmentations (optional)
├── train_test_split.txt
└── waterbird_complete95_forest2water2
```


The segmentation model that we used for early and late masking can be downloaded from [here](https://drive.google.com/file/d/1m1gTCRte-sZ5uD86SeZTh9C-gSlGdVul/view?usp=sharing). 
This can be used to reproduce the masked datasets used in our experiments (along with training the late masked models).

### CUB Binary Segmentation Dataset

The CUB Binary Segmentation Dataset can be generated from ["segmentations"](https://www.vision.caltech.edu/datasets/cub_200_2011/) provided by the CUB-200-2011 dataset.
Please note that you need to apply a threshold to the segmentation masks to generate the binary segmentation masks. We used a threshold of (> 0) to generate the binary segmentation masks.

For training and evaluating the segmentation model, please refer to the instructions in the "mmsegmentation" folder.

The dataset should be stored in the following format:

```
cub_binary_segmentation
├── annotations
│   ├── test
│   └── train
└── images
    ├── test
    ├── train
    └── waterbird_test
```

## Models

Trained models from our experiments will be released soon, however it is already possible to train the models using the code provided in this repository (plus the segmentation model).

Please raise an issue if you face any issues with the code or the datasets.
