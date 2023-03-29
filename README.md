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

We have strcutured our architecture components of the project in modules so that the components can be reused and modified. This section provides details on our system view and technology view.

### System View
Below is our logical system architecture with all the components used to develop the solution.

![system][system-screenshot]


### Technology View

Here are the major frameworks/libraries used in our project.

![tech][tech-screenshot]







[waymo-screenshot]: images/waymo.png
[system-screenshot]: images/system_view.png
[tech-screenshot]: images/tech_view.png