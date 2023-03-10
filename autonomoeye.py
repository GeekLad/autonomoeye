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


def get_annotation_uris(datatype="train"):
    '''Obtains all of the annotation URIs from the capstone project bucket'''
    try:
        return get_uris(f"gs://waymo-processed-images/{datatype}/annotations/**/*.json")
    except:
        return []


def get_all_waymo_segment_uris(datatype="train"):
    if datatype == "train":
        datatype = "training"
    return get_uris(f"gs://waymo_open_dataset_v_1_4_1/individual_files/{datatype}/*.tfrecord")


def get_segments_with_images(datatype):
    image_uris = get_uris(
        f"gs://waymo-processed-images/{datatype}/images/**/*.jpeg")
    segments_with_images = {get_segment_name(uri) for uri in image_uris}
    segments_with_images.remove("")
    return list(segments_with_images)


def get_segments_missing_annotations(datatype):
    annotation_uris = get_annotation_uris(datatype)
    uris = get_all_waymo_segment_uris(datatype)
    processed_segments = [get_segment_name(uri) for uri in annotation_uris]
    all_segments = [get_segment_name(uri) for uri in uris]
    return [uri for uri in all_segments if uri not in processed_segments]


def get_missing_segments(datatype):
    segments_with_images = get_segments_with_images(datatype)
    all_segments = [get_segment_name(uri)
                    for uri in get_all_waymo_segment_uris(datatype)]
    return [uri for uri in all_segments if uri not in segments_with_images]


def get_segment_name(gs_uri):
    '''Parses the Waymo dataset segment name URI/path/filename'''
    matches = re.findall(r"(\d+(?:_\d+){4})", gs_uri)
    if len(matches) > 0:
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
    gcp_url = f"gs://{processed_bucket}/{datatype}/annotations/{date}/{segment_name}.json"

    return [segment_name, date, time_of_day, location, weather, gcp_url]


def fetch_segment(bucket, datatype, segment, temp_directory="tmp"):
    # Import the Segment
    remote_datatype = datatype if datatype != "train" else "training"
    uri = f"gs://waymo_open_dataset_v_1_4_1/individual_files/{remote_datatype}/segment-{segment}_with_camera_labels.tfrecord"
    dataset = tf.data.TFRecordDataset(uri, compression_type='')

    # initialize annnotations dictionary
    annotations = initialize_annotations_dict()

    # process images and annotations in frame and return metadata
    return process_segment(dataset, annotations, bucket, datatype, temp_directory)


def process_segment(dataset, annotations, processed_bucket, datatype, temp_directory):
    temp_directory = re.sub(r"\/$", "", temp_directory)
    if not os.path.exists(temp_directory):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(temp_directory)

    for idx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        # get segment name and date for constructing filepaths
        segment_name = frame.context.name
        timestamp = frame.timestamp_micros/1000000
        dt_object = datetime.fromtimestamp(timestamp)
        date = dt_object.strftime("%Y-%m-%d")
        # Get segment metadata
        if idx == 0:
            seg_metadata = get_metadata(processed_bucket, datatype, frame)

        # Write images to file locations partitioned by day
        for camera_id, image in enumerate(frame.images):
            camera = open_dataset.CameraName.Name.Name(
                camera_id+1)  # Convert to camera name
            img_array = np.array(tf.image.decode_jpeg(image.image))
            img = Image.fromarray(img_array)
            image_id = f"{segment_name}_{idx}_{camera}"
            file_name = f"{image_id}.jpeg"
            temp_file = f"{temp_directory}/{file_name}"
            gcp_path = f"{datatype}/images/{date}/{segment_name}/{file_name}"
            img.save(temp_file)
            upload_blob(processed_bucket, temp_file, gcp_path)
            os.remove(temp_file)
            annotations["images"].append({
                "id": f"{segment_name}_{idx}_{camera}",
                "gcp_url": f"gs://{processed_bucket}/{gcp_path}",
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
                            "image_id": file_name,
                            "area": label.box.length*label.box.width,
                            "bbox": bbox
                        })

    # Save the annotations
    with open(f"{temp_directory}/{segment_name}.json", "w") as f:
        json.dump(annotations, f)
    upload_blob(processed_bucket, f"{temp_directory}/{segment_name}.json",
                f"{datatype}/annotations/{date}/{segment_name}.json")
    os.remove(f"{temp_directory}/{segment_name}.json")

    # Add the processed info to the metadata csv
    metadata_path = f"gs://{processed_bucket}/{datatype}/metadata/metadata.csv"
    try:
        metadata = pd.read_csv(metadata_path)
        metadata.loc[len(metadata)] = seg_metadata
        metadata.drop_duplicates().to_csv(metadata_path, index=False)
    except:
        metadata = pd.DataFrame([seg_metadata], columns=[
                                segment_name, "date", "time_of_day", "location", "weather", "gcp_url"])
        metadata.to_csv(metadata_path, index=False)
    return seg_metadata
