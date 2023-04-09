import streamlit as st

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


def plot_annotations(img, bbox, labels, scores, confidence_threshold,
                     save_fig_path='predicted_img.jpeg', show=False, save_fig=True):
    """
    This function plots bounding boxes over image with text labels and saves the image to a particualr location.
    """

    # Default colors and mappings
    colors_map = {'1': '#5E81AC', '2': '#A3BE8C', '3': '#B48EAD'}
    labels_map = {'1': 'Vehicle', '2': 'Person', '3': 'Cyclist'}

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(200, 200))

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


def process_uploaded_image():
    global processing
    global prior_image
    global prior_nms
    try:
        if processing == True:
            return
    except:
        processing = False
        prior_image = None
        prior_nms = 0
    if uploaded_img != prior_image and nms != prior_nms and processing == False:
        processing = True
        prior_image = uploaded_img
        prior_nms = nms
        placeholder.empty()
        image = Image.open(uploaded_img)
        image.save('userupload/tmpImgFile.jpg')
        with placeholder.container():
            st.write("Our Model is running its prediction...")
            st.write("&nbsp;")
            st.image(image, caption='Uploaded Image.',
                     use_column_width=False, width=776)
        # Call Model
        pred_img = generate_prediction_user_image(
            'userupload/tmpImgFile.jpg', rest_api, '/tmp/tmp_usr.jpeg', st.session_state.nms)
        placeholder.empty()
        time.sleep(0.01)
        with placeholder.container():
            st.image(pred_img)
        processing = False


rest_api = os.getenv("FLASK_API_URL", "http://localhost:5000/predict")

# st.title("AutonomoEye")
# Page Config
st.set_page_config(page_title="autonomoeye",
                   page_icon="eye",
                   layout="wide")

col1, col2 = st.columns((2, 3))
with col2:
    # Header
    header_img = Image.open('assets/autonomoeye_1.png')
    st.image(header_img, width=300)


row1_1, row1_2 = st.columns((2, 3))

# Title
with row1_2:
    st.title("Object Detector")

st.markdown("""---""")

# st.markdown("#")
# User Upload
st.markdown('''
    ## Upload an Image For Vehicle and Pedestrian Detection
    Lower values for detection sensitivity will detect more objects, but potentially have more false positives.  Higher values should result in fewer false positives, but also fewer objects detected overall.    
''')
nms = st.slider("Detection Sensitivity", 0.1, 0.9, 0.1, 0.01, key="nms",
                on_change=process_uploaded_image)
uploaded_img = st.file_uploader("Upload an image...", type=["jpg", "jpeg"])
placeholder = st.empty()
process_uploaded_image()

st.markdown("""---""")

# Dropdown
row2_1, row2_2, row2_3 = st.columns((1.5, 1.5, 1.5))
location_map = {'San Francisco': 'location_sf',
                'Phoenix': 'location_phx', 'Other': 'location_other'}
tod_map = {'Day': 'day', 'Dawn/Dusk': 'dawn_dusk', 'Night': 'night'}
weather_map = {'Sunny': 'sunny', 'Rain': 'rain'}

with row2_1:
    location = st.selectbox('Location', ('San Francisco', 'Phoenix', 'Other'))
with row2_2:
    tod = st.selectbox('Time of Day', ('Day', 'Dawn/Dusk', 'Night'))
with row2_3:
    weather = st.selectbox('Weather', ('Sunny', 'Rain'))

# Slider
frame = st.slider("Frame", 0, 180, 0)

#segments = []
#base = '/Users/ananshu/Documents/Aravindh/MADS/VSCODE/Projects/autonomoeye_1/app/'
#print(glob('/data/{}/{}/{}/*/'.format(location_map[location],tod_map[tod], weather_map[weather])).split('/')[-2])

segments = [x.split('/')[-2] for x in glob('data/{}/{}/{}/*/'.format(
    location_map[location], tod_map[tod], weather_map[weather]))]
segment = segments[np.random.randint(0, len(segments))]

print(segments)
print(segment)


row3_1, row3_2, row3_3 = st.columns((1.5, 1.5, 1.5))
with row3_1:
    try:
        img_fl = Image.open('data/{}/{}/{}/{}/{}_{}_FRONT_LEFT.jpeg'.format(
            location_map[location], tod_map[tod], weather_map[weather], segment, segment, frame))
        #print('data/{}/{}/{}/{}/{}_{}_FRONT_LEFT.jpeg'.format(location_map[location],tod_map[tod], weather_map[weather], segment, segment,frame))
        st.markdown("## Front Left Camera")

        st.image(img_fl)
    except:
        st.markdown("")

with row3_2:
    try:
        img_f = Image.open('data/{}/{}/{}/{}/{}_{}_FRONT.jpeg'.format(
            location_map[location], tod_map[tod], weather_map[weather], segment, segment, frame))
        st.markdown("## Front Center Camera")
        # Call Model
        st.image(img_f)
    except:
        st.markdown("")

with row3_3:
    try:
        img_sl = Image.open('data/{}/{}/{}/{}/{}_{}_SIDE_LEFT.jpeg'.format(
            location_map[location], tod_map[tod], weather_map[weather], segment, segment, frame))
        st.markdown("## Front Right Camera")
        st.image(img_sl)
    except:
        st.markdown("")


row4_1, row4_2, row4_3 = st.columns((1.5, 1.5, 1.5))
with row4_1:
    try:
        img_l = Image.open('data/{}/{}/{}/{}/{}_{}_FRONT_RIGHT.jpeg'.format(
            location_map[location], tod_map[tod], weather_map[weather], segment, segment, frame))
        st.markdown("## Left Camera")
        st.image(img_l)
    except:
        st.markdown("")

with row4_2:
    st.markdown("")

with row4_3:
    try:
        img_r = Image.open('data/{}/{}/{}/{}/{}_{}_SIDE_RIGHT.jpeg'.format(
            location_map[location], tod_map[tod], weather_map[weather], segment, segment, frame))
        st.markdown("## Right Camera")
        st.image(img_r)
    except:
        st.markdown("")

st.markdown("""---""")
