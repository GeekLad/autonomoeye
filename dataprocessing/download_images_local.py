import os
import sys
import json

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import argparse
from multiprocessing import Manager, Pool

import torch
import torch.utils.data as data

from google.cloud import storage

from autonomoeye.utils.gcp import download_blob, upload_blob


def download_images_to_local(gcp_bucket, gcp_annotation, local_path):

    gcp_annotations_path = gcp_annotation
    dataset_name =  gcp_annotations_path.split('/')[-1].replace('.json','')
    segment_date = gcp_annotations_path.split('/')[-2]
    dataset_path =  local_path +'/'.join(gcp_annotations_path.split('/')[:-2]) 
    path_to_images = dataset_path.replace('annotations','') +'images/' + segment_date + '/' + dataset_name + '/'
    path_to_annotations = dataset_path + '/{}.json'.format(dataset_name)


    if os.path.exists(dataset_path)==False:
        os.mkdir(dataset_path)
    if os.path.exists(path_to_images)==False:
        os.mkdir(path_to_images)

    # read in annotations
    client = storage.Client()
    bucket = client.get_bucket(gcp_bucket)
    
    print("Downloading Annotation")
    download_blob(gcp_bucket,gcp_annotations_path,path_to_annotations)

    print("Reading Annotation")
    f = open(path_to_annotations,'r')
    annotations = json.load(f)
    f.close()

    # determine segment paths
    segment_paths = []
    for image in annotations['images']:
            uri = image['gcp_url']
            segment = '/'.join(uri.split('/')[3:7])+'/'
            if segment not in segment_paths:
                segment_paths.append(segment)
    
    # Download images for segments to local folder
    for segment in segment_paths:
            blobs = bucket.list_blobs(prefix=segment, delimiter='/')
            for blob in tqdm(list(blobs)):
                filename=blob.name.replace(segment,'')
                blob.download_to_filename(path_to_images + '/{}'.format(filename))
                #print(filename)
    print("Image download - done")