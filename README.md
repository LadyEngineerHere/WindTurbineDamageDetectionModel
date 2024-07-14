

# Wind Turbine Damage Detection Project

![Project Title Image](https://raw.githubusercontent.com/LadyEngineerHere/ladyengineerhere-image-resources/main/20201023-title.png)


## Overview

This project aims to develop a deep learning model for detecting various types of damage on wind turbine images. The types of damage include cracks, erosion, lightening, and VG panel damage. The project involves creating a dataset, training a convolutional neural network (CNN) model using PyTorch, and evaluating its performance.

## Project Structure

The project is organized into the following structure:

```
WindTurbineDamageDetection/
│
├── annotations/
│ ├── cracks/
│ ├── LE Erosion/
│ ├── Lightening/
│ └── VG Panel/
│
├── annotations.json
├── combinejsonfiles.py
├── Data images/
│ ├── cracks/
│ ├── LE Erosion/
│ ├── Lightening/
│ └── VG Panel/
│
├── dataset.py
└── train.py
```





* annotations/: Contains JSON annotation files for each damage type.
* annotations.json: Combined JSON file generated from individual annotation files using combinejsonfiles.py.
* combinejsonfiles.py: Python script to combine multiple JSON annotation files into a single annotations.json.
* Data images/: Contains subfolders with images categorized by damage type.
* dataset.py: Python script defining the custom dataset class (CustomDataset) for loading images and annotations.
* train.py: Main script for training the CNN model using PyTorch.

# Organizing the Data

### Gathering Images

Images for training were sourced from online repositories and search engines, focusing on wind turbine damage scenarios. These images were manually categorized into folders under **Data images/** based on the type of damage depicted.

### Annotating Images

To prepare the dataset for training, the LabelIMG repository was cloned and utilized. This tool allows for the annotation of images by drawing bounding boxes around areas of damage. Annotations were saved in JSON format corresponding to each image category (cracks, LE Erosion, Lightening, VG Panel) within the **annotations/** folder.

To combine these individual JSON files into a unified dataset, the `combinejsonfiles.py` script was executed. This script merged annotations from all categories into a single `annotations.json` file, assigning unique IDs to each image for consistency during training.

## Usage

### 1. Combining Annotations
Before training, run combinejsonfiles.py to combine individual JSON annotation files from annotations/ into annotations.json. This script merges annotations and assigns unique IDs to each image.

```
python combinejsonfiles.py
```

### 2. Training the Model
Edit train.py to adjust hyperparameters, such as learning rate, batch size, and number of epochs. The script loads the dataset, initializes the CNN model (GoogleNet in this case), defines loss and optimizer functions, and trains the model using GPU acceleration if available.

```
python train.py
```
### 3. Custom Dataset
dataset.py contains the CustomDataset class, which loads images and annotations from Data images/ and annotations.json. Ensure all image paths and annotations are correctly formatted before training.

## Notes

Adjust paths (dataset_root, annotation_file) in train.py and dataset.py to match your local directory structure.
Ensure all dependencies (PyTorch, torchvision, PIL) are installed and compatible with your Python environment.

