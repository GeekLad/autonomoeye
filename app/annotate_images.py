# import streamlit as st

from PIL import Image

import requests
import json
import numpy as np
import os
import time

from glob import glob

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches


# Create figure and axes
fig, ax = plt.subplots(figsize=(200, 200))


def plot_annotations(img, bbox, labels, scores, confidence_threshold,
                     save_fig_path='predicted_img.jpeg', show=False, save_fig=True):
    """
    This function plots bounding boxes over image with text labels and saves the image to a particualr location.
    """

    # Default colors and mappings
    colors_map = {'1': '#5E81AC', '2': '#A3BE8C', '3': '#B48EAD'}
    labels_map = {'1': 'Vehicle', '2': 'Person', '3': 'Cyclist'}

    # Clear the figure
    ax.clear()

    # Display the image
    ax.imshow(img)
    i = 0
    # Filter for scores greater than certain threshold
    scores_ind = [idx for idx, x in enumerate(
        scores) if x > confidence_threshold]
    for idx, entry in enumerate(bbox):
        if idx in scores_ind:
            h = entry[2] - entry[0]
            w = entry[3] - entry[1]

            # Create a Rectangle patch
            rect = patches.Rectangle((entry[0], entry[1]), h, w,
                                     linewidth=60,
                                     edgecolor=colors_map[str(labels[idx])],
                                     facecolor='none')

            # Add classification category
            plt.text(entry[0], entry[1], s=labels_map[str(labels[idx])],
                     color='white', verticalalignment='top',
                     bbox={'color': colors_map[str(labels[idx])], 'pad': 0},
                     font={'size': 150})

            # Add the patch to the Axes
            ax.add_patch(rect)
        i += 1

    if show == True:
        plt.show()

    plt.savefig(save_fig_path,
                bbox_inches='tight',
                pad_inches=0,
                dpi=5)

    return save_fig_path


def generate_prediction_user_image(imgfile, rest_api, image_path, nms):
    """
    This function requests prediction from rest_api and 
    plots bounding boxes of result over image.
    """
    # Make request to prediction api
    files = {'image': open(imgfile, 'rb')}
    response = requests.post(rest_api, files=files)

    # Parse response
    annotations = json.loads(response.content)
    boxes = [np.array(x).astype(float) for x in annotations['boxes']]
    labels = np.array(annotations['labels']).astype('int')
    scores = np.array(annotations['scores']).astype('float')

    # Create new figure
    img = Image.open(imgfile)
    pred_fig = plot_annotations(img, boxes, labels, scores,
                                nms, save_fig_path=image_path)

    return pred_fig


def save_annotation(location, tod, weather, camera):
    path = "{}/{}/{}/{}".format(
        location_map[location], tod_map[tod], weather_map[weather], segment)
    src_path = f'data/{path}/{segment}_0_{camera}.jpeg'
    target_path = f'out/{path}/{segment}_0_{camera}.jpeg'
    if not os.path.exists(f'out/{path}'):
        os.makedirs(f'out/{path}')
    elif os.path.isfile(target_path):
        print(f'{target_path} already exists')
        return

    print(f"Processing {src_path}")
    try:
        generate_prediction_user_image(
            src_path, rest_api, target_path, 0.1)
    except:
        print(f"Error processing {src_path}")


rest_api = os.getenv("FLASK_API_URL", "http://localhost:5000/predict")

# Dropdown
location_map = {'San Francisco': 'location_sf',
                'Phoenix': 'location_phx', 'Other': 'location_other'}
tod_map = {'Day': 'day', 'Dawn/Dusk': 'dawn_dusk', 'Night': 'night'}
weather_map = {'Clear': 'clear', 'Rain': 'rain'}

for location in location_map:
    for tod in tod_map:
        for weather in weather_map:
            try:
                segments = [x.split('/')[-2] for x in glob('data/{}/{}/{}/*/'.format(
                    location_map[location], tod_map[tod], weather_map[weather]))]
                segment = segments[np.random.randint(0, len(segments))]
            except:
                print(f"No images for {location}, {tod}, {weather}")
                continue
            print(f"Processing {location}, {tod}, {weather}")
            save_annotation(location, tod, weather, "FRONT_LEFT")
            save_annotation(location, tod, weather, "FRONT")
            save_annotation(location, tod, weather, "FRONT_RIGHT")
            save_annotation(location, tod, weather, "SIDE_LEFT")
            save_annotation(location, tod, weather, "SIDE_RIGHT")
