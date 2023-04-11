<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/GeekLad/autonomoeye">
    <img src="images/autonomoeye_1.png" alt="Logo" width="200" height="100">
  </a>

  <h3 align="center">Autonomoeye</h3>

  <p align="center">
    SIADS 699 Capstone Project
    <br />
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-overview">Project Overview</a>
    </li>
    <li>
      <a href="#waymo-dataset">Waymo Dataset</a>
      <ul>
        <li><a href="#images">Images</a></li>
        <li><a href="#annotations">Annotations</a></li>
        <li><a href="#metadata">Metadata</a></li>
      </ul>
    </li>
    <li>
      <a href="#architecture">Architecture</a>
      <ul>
        <li><a href="#system_view">System View</a></li>
        <li><a href="#technology_view">Technology View</a></li>
      </ul>
    </li>
    <li><a href="#data_processing">Data Processing</a></li>
    <li><a href="#faster_r_cnn_model">Faster R CNN Model</a></li>
    <li><a href="#model_training">Model Training</a></li>
    <li><a href="#model_evaluation">Model Evaluation</a></li>
    <li><a href="#user_interface">User Interface</a></li>
    <li><a href="#results">Results</a></li>
  </ol>
</details>

<br />
<br />
<br />



## Project Overview
The goal of this project is to develop an object detection model using the Waymo Open Dataset and create an interactive user interface (UI) using Streamlit and Flask. The UI will allow users to upload images, visualize the object detection results, and adjust the model's parameters in real time.



## Waymo Dataset
Waymo Open Dataset is the largest and most diverse multimodal autonomous driving dataset to date, comprising of images recorded by multiple high-resolution cameras and sensor readings from LiDAR scanners. The data has been recorded with large geographical coverage within multiple cities like San Francisco, Phoenix and many more.

Since our project focuses on 2D detection, the only data we are using are the camera images. The dataset contains around 10 million camera box annotations, all manually annotated for 1150 scenes or segments. 

[![waymo][waymo-screenshot]](https://waymo.com/open/)

### Images
We have five different views for a given frame/scene in the segments

- FRONT
- FRONT_LEFT
- FRONT_RIGHT
- SIDE_LEFT
- SIDE_RIGHT

### Annotations
Each object (vehicles, pedestrians, cyclists and signs) are annotated with a tightly fitting 4-DOF image axis-aligned 2D bounding box. The label is encoded as (cx, cy, l, w) with a unique tracking ID, where cx and cy represent the center pixel of the box, l represents the length of the box along the horizontal (x) axis in the image frame, and w represents the width of the box along the vertical (y) axis in the image frame.

### Metadata
We also have metadata related to the scene and image capture. 

- Segment Name
- Date
- Time of the day
- Location
- Weather


## Architecture
We have structured our architecture components of the project in modules so that the components can be reused and modified. This section provides details on our system view and technology view.

### System View
Below is our logical system architecture with all the components used to develop the solution.

![system][system-screenshot]


### Technology View
Here are the major frameworks/libraries used in our project.

![tech][tech-screenshot]


## Data Processing
The [Waymo Open Dataset](https://waymo.com/open/) is stored in a [Google Cloud Platform (GCP)](https://cloud.google.com/) [Storage Bucket](https://cloud.google.com/storage/), and is split into hundreds of [TensorFlow TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) segments.  Each segment contains approximately 1,000 images and related data, including labelled annotations for vehicles, pedestrians, signs, and cyclists.  Below is an overview of how the data was processed

![data][data-processing]

We processed the data in two stages:

### Extraction from the GCP Bucket
First the data needed to be parsed from the TFRecordset files.  We developed a Python script to load each segment, iterate through each frame, download each of the images, and gather all the annotations into a single file (per segment).  This process checked for existing annotations and image files so that they could be resumed if interrupted.  It also alowed for multiple instances to be run simultaneously, which helped to speed up processing.  

We utilized GCP VMs and personal computers to perform this processing step.  The data was stored in another Google Cloud Storage bucket that we created for the project. This was so that we could ensure we could store the data in a centralized repository without storage space limitations.

### Placement into Training Environment
The next step in the process was to place the data into the training environment, and perform some necessary transformations.  In our project storage bucket, there was one `.json` annotation file per segment.  For simplification of training, when downloading from the project bucket to the training environment, we combined them into a single `.json` annotation file.

In addition to combining the annotations, we also had to resize the images.  That also resulted in having to make updates to the annotations, due to the changes in the bounding boxes for the detected objects in the images.  These were also placed into a single file to facilitate the training process.

## Executing Data Processing

### Requirements

- Sign up for free access to the [Waymo Open Dataset](https://waymo.com/intl/en_us/open/licensing/)
- [Install the Google Cloud CLI tools](https://cloud.google.com/sdk/docs/install)
- [Authenticate with `gcloud`](https://cloud.google.com/sdk/docs/authorizing) using the same email address you used to access the Waymo Open Dataset
- [Dowload and install Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Set up and activate a conda environment with `conda create --name autonomoeye python=3.8.10 && conda activate autonomoeye`
- Install the required Python packages with `pip install -r requirements.txt`

### Extract from GCP Bucket

The [`dataprocessing/process_coco.py`](dataprocessing/process_coco.py) script should be run from the project directory as follows:

```shell
export PYTHONPATH=$PWD/..:$PYTHONPATH
python3 dataprocessing/process_coco.py -b BUCKET_NAME -t DATA_TYPE
```

`DATA_TYPE` should be set to either `train` or `validation`.  The files will be downloaded and organized in the `BUCKET_NAME`.

### Download Data to Training Environment

The [`utils/gcp.py`](utils/gcp.py.py) script should be run from the project directory as follows:

```shell
export PYTHONPATH=$PWD/..:$PYTHONPATH
python3 utils/gcp.py -b BUCKET_NAME -t DATA_TYPE -d DATA_DIRECTORY
```

`DATA_TYPE` should be set to either `train` or `validation`.  The files will be downloaded locally and stored in `DATA_DIRECTORY`.

## Faster R CNN Model
For object detection we need to build a model and teach it to learn to both recognize and localize objects in the image. The Faster R-CNN model takes the following approach: The Image first passes through the backbone network to get an output feature map, and the ground truth bounding boxes of the image get projected onto the feature map. The backbone network is usually a dense convolutional network like ResNet or MobileNet. The output feature map is a spatially dense Tensor that represents the learned features of the image. Next, we treat each point on this feature map as an anchor. For each anchor, we generate multiple boxes of different sizes and shapes. The purpose of these anchor boxes is to capture objects in the image.

We used a 1x1 convolutional network to predict the category and the offsets of all the anchor boxes. During training, we sample the anchor boxes that overlap the most with the projected ground truth boxes. These are called positive or activated anchor boxes. We also sample negative anchor boxes which have little to no overlap with the ground truth boxes. The positive anchor boxes are assigned the category object, while the negative boxes are assigned background. The network learns to classify anchor boxes using binary cross-entropy loss. Now, the positive anchor boxes may not exactly align with the projected ground truth boxes. So we train a similar 1x1 convolutional network to learn to predict offsets from ground truth boxes. These offsets when applied to the anchor boxes bring them closer to the ground truth boxes. We use L2 regression loss to learn the offsets. The anchor boxes are transformed using the predicted offsets and are called region proposals, and the network described above is called the region proposal network.

We predicted the category of the object in the region proposal using a simple convolutional network. The raw region proposals are of different sizes, so we use a technique called ROI pooling to resize them before passing through the network. This network learns to predict multiple categories using cross-entropy loss. We used another network to predict offsets of region proposals from ground truth boxes. This network further tries to align region proposals with ground truth boxes. This uses L2 regression loss. Finally we take a weighed combination of both the losses to compute the final loss.

![frcnn][frcnn-screenshot]


## Model Training
In PyTorch, it’s considered a best practice to create a class that inherits from PyTorch’s Dataset class to load the data. This will give us more control over the data and helps keep the code modular. We created a PyTorch DataLoader from the dataset instance which automatically take care of batching, shuffling, and sampling the data.

We used ResNet 50 and Mobilenet as the backbone networks. A single block in ResNet 50 is composed of stacks of bottleneck layers. The Image gets reduced in half after each block along the spatial dimension while the number of channels get doubled. A bottleneck layer is composed of three convolutional layers along with a skip connection. Once an image passes through the backbone network, it gets downsampled along the spatial dimension. 

![train][train]

## Model Evaluation
We used PyTorch DataLoader to create the validation dataload as well. Model evaluation provides the Predicted Bounding Boxes based on the ground truth bunding boxes. Then we performed NMS - Non maximum Supression before calculating the IOU to get the precision and recall.

![eval][eval]

## User Interface
Streamlit is an open-source Python library that can build a UI for various purposes. It has simple but effective components for user interaction and layout which let us build effective and attractive data science and dashboard applications.

We have simple UI design to display the object detection of images with different location, time and weather.

Also, we have an option for users to upload the image and run our object detection model to display the results.

## Results
The model is being trained using Faster R-CNN which consists of several losses. Here are the details on each of the losses:

loss_classifier: This loss is the cross-entropy loss for the predicted class labels. It calculates the difference between predicted class probabilities and true class probabilities for each object.
loss_box_reg: This loss measures the difference between the predicted bounding boxes and the ground truth bounding boxes.
loss_objectness: This loss measures the difference between the predicted objectness scores and the ground truth objectness scores. Objectness is a binary value indicating whether an object is present or not.
loss_rpn_box_reg: This loss measures the difference between the predicted bounding boxes and the ground truth bounding boxes generated by the region proposal network (RPN).

During training, all the above losses are calculated, and their sum is used to compute the gradients and update the weights of the model. The aim is to minimize the total loss and improve the performance of the model. Finally, we used Weights and Biases (wandb) to track training metrics such as loss, classifier loss, box regression loss, objectness loss, and RPN loss. The wandb_config specifies hyperparameters for the model such as the number of epochs, learning rate, and batch size. 

![results][results]



[waymo-screenshot]: images/waymo.png
[system-screenshot]: images/system_view.png
[tech-screenshot]: images/tech_view.png
[data-processing]: images/data_processing.png
[frcnn-screenshot]: images/fastrcnn.png
[train]: images/train.png
[eval]: images/eval.png
[results]: images/results.png



