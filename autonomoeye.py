'''
Contains various helper functions for Waymo dataset processing
'''

# Required imports
import subprocess
import os
from datetime import datetime
import re
import numpy as np
import json
import pandas as pd
from waymo_open_dataset import dataset_pb2 as open_dataset
from PIL import Image
import tensorflow as tf
from tensorflow.python.lib.io.file_io import FileIO
from google.cloud import storage


def get_uris(gs_uri):
  '''
  Lists all the files under a Google Storage bucket URI.
  This function requires the Google Cloud CLI to be installed.
  '''
  files = subprocess.check_output(f"gsutil ls {gs_uri}", shell=True)
  return files.decode("utf-8").split("\n")


def get_annotation_uris():
  '''Obtains all of the annotation URIs from the capstone project bucket'''
  return get_uris("gs://waymo-processed-images/train/annotations/**/*.json")

def get_all_waymo_training_segment_uris():
  return get_uris("gs://waymo_open_dataset_v_1_4_1/individual_files/training/*.tfrecord")

def get_segment_name(gs_uri):
  '''Parses the Waymo dataset segment name URI/path/filename'''
  matches = re.findall(r"(\d+(?:_\d+){4})", gs_uri)
  if len(matches) == 1:
    return matches[0]
  return ""


def initialize_annotations_dict():
  annotations = {
      "info": {"description": "Waymo Open Data"},
      "licenses": {},
      "images": [],
      "annotations": [],
      "categories": [
          {
              "id": 1,
              "name": "TYPE_VEHICLE"
          },
          {
              "id": 2,
              "name": "TYPE_PEDESTRIAN"
          },
          {
              "id": 4,
              "name": "TYPE_CYCLIST"
          }
      ]
  }
  return annotations


def upload_blob(bucket_name, source_file_name, destination_blob_name):
  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(destination_blob_name)
  blob.upload_from_filename(source_file_name)

def get_metadata(processed_bucket, datatype, frame):
  segment_metadata = frame.context.stats
  segment_name = frame.context.name
  timestamp = frame.timestamp_micros/1000000
  dt_object = datetime.fromtimestamp(timestamp)
  date = dt_object.strftime("%Y-%m-%d")
  time_of_day = segment_metadata.time_of_day
  location = segment_metadata.location
  weather = segment_metadata.weather
  gcp_url = "gs://{processed_bucket}/{datatype}/annotations/{date}/{segment_name}.json"

  return [segment_name, date, time_of_day, location, weather, gcp_url]

def process_segment(dataset, annotations, processed_bucket, datatype):
  for idx, data in enumerate(dataset):
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    # get segment name and date for constructing filepaths
    segment_name = frame.context.name
    timestamp = frame.timestamp_micros/1000000
    dt_object = datetime.fromtimestamp(timestamp)
    date = dt_object.strftime("%Y-%m-%d")
    # Get segment metadata
    if idx == 1:
      seg_metadata = get_metadata(processed_bucket, datatype, frame)

    # Write images to file locations partitioned by day
    for camera_id, image in enumerate(frame.images):
      camera = open_dataset.CameraName.Name.Name(
          camera_id+1)  # Convert to camera name
      img_array = np.array(tf.image.decode_jpeg(image.image))
      img = Image.fromarray(img_array)
      file_name = f"{segment_name}_{idx}_{camera}.jpg"
      gcp_url = f"gs://{processed_bucket}/{datatype}/images/{date}/{segment_name}/{file_name}"
      with FileIO(gcp_url, 'wb') as f:
        img.save(f, "PNG")
      annotations["images"].append({
          "id": f"{segment_name}_{idx}_{camera}",
          "gcp_url": gcp_url,
          "file_name": file_name
      })

      for camera_labels in frame.camera_labels:
        # Ignore camera labels that do not correspond to this camera.
        if camera_labels.name != image.name:
          continue
        else:                     # Iterate over the individual labels.
          for label in camera_labels.labels:
            bbox = [label.box.center_x - (0.5*label.box.length),
                    label.box.center_y - (0.5*label.box.width),
                    label.box.width, label.box.length]
            annotations["annotations"].append({
                "id": label.id,
                "category_id": label.type,
                "image_id": "{}_{}_{}".format(segment_name, idx, camera),
                "area": label.box.length*label.box.width,
                "bbox": bbox
            })

  # Save the annotations
  with open(f"{segment_name}.json", "w") as f:
    json.dump(annotations, f)
  upload_blob(processed_bucket, "f{segment_name}.json",
    f"{datatype}/annotations/{date}/{segment_name}.json")
  os.remove(f"{segment_name}.json")

  # Add the processed info to the metadata csv
  metadata_path = f"gs://{processed_bucket}/{datatype}/metadata/metadata.csv"
  metadata = pd.read_csv(metadata_path)
  metadata.loc[len(metadata)] = seg_metadata
  metadata.drop_duplicates().to_csv(metadata_path, index=False)

  return seg_metadata