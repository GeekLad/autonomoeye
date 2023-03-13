import streamlit as st

from PIL import Image

import requests
import multiprocessing
from itertools import product
import json
import numpy as np

from glob import glob

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches

import os 


#st.title("AutonomoEye")
#Page Config
st.set_page_config(page_title="autonomoeye",  
                    page_icon="eye",
                    layout="wide")

col1, col2 = st.columns((2,3)) 
with col2:
    #Header
    header_img = Image.open('assets/autonomoeye_1.png')
    st.image(header_img, width=300)


row1_1, row1_2 = st.columns((2,3)) 

#Title
with row1_2:
    st.title("Object Detector")

st.markdown("""---""")

#Dropdown
row2_1, row2_2, row2_3 = st.columns((1.5,1.5,1.5))
location_map = {'San Francisco':'location_sf','Phoenix':'location_phx','Other':'location_other'} 
tod_map = {'Day':'day','Dawn/Dusk':'dawn_dusk','Night':'night'} 
weather_map = {'Sunny':'sunny', 'Rain':'rain'}

with row2_1:
    location = st.selectbox('Location', ('San Francisco', 'Phoenix', 'Other'))
with row2_2:
    tod = st.selectbox('Time of Day', ('Day', 'Dawn/Dusk', 'Night'))
with row2_3:
    weather = st.selectbox('Weather', ('Sunny', 'Rain'))

# Slider
frame = st.slider("Frame",0,180,0)
st.markdown("#")

#segments = []
#base = '/Users/ananshu/Documents/Aravindh/MADS/VSCODE/Projects/autonomoeye_1/app/'
#print(glob('/data/{}/{}/{}/*/'.format(location_map[location],tod_map[tod], weather_map[weather])).split('/')[-2])

segments = [x.split('/')[-2] for x in glob('data/{}/{}/{}/*/'.format(location_map[location],tod_map[tod], weather_map[weather]))]
segment = segments[np.random.randint(0,len(segments))]

print(segments)
print(segment)



row3_1, row3_2, row3_3 = st.columns((1.5,1.5,1.5))
with row3_1:
    try:
        img_fl = Image.open('data/{}/{}/{}/{}/{}_{}_FRONT_LEFT.jpeg'.format(location_map[location],tod_map[tod], weather_map[weather], segment, segment,frame))
        #print('data/{}/{}/{}/{}/{}_{}_FRONT_LEFT.jpeg'.format(location_map[location],tod_map[tod], weather_map[weather], segment, segment,frame))
        st.markdown("## Front Left Camera")
        st.image(img_fl)
    except:
        st.markdown("")

with row3_2:
    try:
        img_f = Image.open('data/{}/{}/{}/{}/{}_{}_FRONT.jpeg'.format(location_map[location],tod_map[tod], weather_map[weather], segment, segment,frame))
        st.markdown("## Front Center Camera")
        # Call Model
        st.image(img_f)
    except:
        st.markdown("")

with row3_3:
    try:
        img_sl = Image.open('data/{}/{}/{}/{}/{}_{}_SIDE_LEFT.jpeg'.format(location_map[location],tod_map[tod], weather_map[weather], segment, segment,frame))
        st.markdown("## Front Right Camera")
        st.image(img_sl)
    except:
        st.markdown("")


row4_1, row4_2, row4_3 = st.columns((1.5, 1.5,1.5))
with row4_1:
    try:
        img_l = Image.open('data/{}/{}/{}/{}/{}_{}_FRONT_RIGHT.jpeg'.format(location_map[location],tod_map[tod], weather_map[weather], segment, segment,frame))
        st.markdown("## Left Camera")
        st.image(img_l)
    except:
        st.markdown("")

with row4_2:
    st.markdown("")
    
with row4_3:
    try:
        img_r = Image.open('data/{}/{}/{}/{}/{}_{}_SIDE_RIGHT.jpeg'.format(location_map[location],tod_map[tod], weather_map[weather], segment, segment,frame))
        st.markdown("## Right Camera")
        st.image(img_r)
    except:
        st.markdown("")

st.markdown("""---""")

st.markdown("#")
#User Upload
st.markdown('''## You want to try now?''')
uploaded_img = st.file_uploader("Upload an image...", type="jpeg")
if uploaded_img is not None:
    image = Image.open(uploaded_img)
    image.save('userupload/tmpImgFile.jpg')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Our Model is running its prediction...")
    # Call Model

st.markdown("""---""")