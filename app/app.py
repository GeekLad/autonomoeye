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


def show_upload_ui():
    prior_nms_key = f"nms_{st.session_state['upload_count']}"
    st.session_state['upload_count'] = st.session_state['upload_count'] + 1
    default_sensitivity = st.session_state[prior_nms_key] if prior_nms_key in st.session_state else 0.1
    nms_key = f"nms_{st.session_state['upload_count']}"
    upload_key = f"upload_{st.session_state['upload_count']}"
    st.markdown("Lower values for detection sensitivity will detect more objects, but potentially have more false positives.  Higher values should result in fewer false positives, but also fewer objects detected overall.")
    st.slider("Detection Sensitivity", 0.1, 0.9, default_sensitivity, 0.01, key=nms_key,
              on_change=process_uploaded_image)
    st.file_uploader(
        "Upload an image...", key=upload_key, type=["jpg", "jpeg"], on_change=process_uploaded_image)


def process_uploaded_image():
    global processing
    global prior_image
    try:
        if processing == True:
            return
    except:
        processing = False
    nms_key = f"nms_{st.session_state['upload_count']}"
    upload_key = f"upload_{st.session_state['upload_count']}"
    uploaded_img = st.session_state[upload_key]
    if uploaded_img == None and "current_image" in st.session_state:
        uploaded_img = st.session_state["current_image"]
    nms = st.session_state[nms_key]
    if uploaded_img != None and processing == False:
        processing = True
        st.session_state["current_image"] = uploaded_img
        placeholder.empty()
        image = Image.open(uploaded_img)
        image.save('userupload/tmpImgFile.jpg')
        with placeholder.container():
            st.write("Our Model is running its prediction, please wait...")
            st.image(image, caption='Uploaded Image',
                     use_column_width=False, width=776)
        # Call Model
        pred_img = generate_prediction_user_image(
            'userupload/tmpImgFile.jpg', rest_api, '/tmp/tmp_usr.jpeg', nms)
        placeholder.empty()
        with placeholder.container():
            time.sleep(0.01)
            show_upload_ui()
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
st.markdown('''## Upload an Image For Vehicle and Pedestrian Detection''')
placeholder = st.empty()
if "upload_count" not in st.session_state:
    st.session_state["upload_count"] = -1
    uploaded_img = None
    nms = -1
with placeholder.container():
    show_upload_ui()
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
