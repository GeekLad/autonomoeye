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
import struct
import imghdr
from tqdm import tqdm
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


def get_partial_segments(datatype):
    image_segments = get_segments_with_images(datatype)
    annotation_segments = [get_segment_name(
        uri) for uri in get_annotation_uris(datatype)]
    return [segment for segment in image_segments if segment not in annotation_segments]


def get_last_index(bucket, datatype, segment):
    try:
        image_uris = get_uris(
            f"gs://{bucket}/{datatype}/images/*/{segment}/*.jpeg")
        indicies = [re.findall(
            r"\d+(?:_\d+){4}_(\d+)_", uri) for uri in image_uris]
        indicies = [int(index[0]) for index in indicies if len(index) > 0]
        return max(indicies)
    except:
        return 0


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

    last_index = None
    for idx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        # get segment name and date for constructing filepaths
        segment_name = frame.context.name
        timestamp = frame.timestamp_micros/1000000
        dt_object = datetime.fromtimestamp(timestamp)
        date = dt_object.strftime("%Y-%m-%d")
        if last_index == None:
            last_index = get_last_index(
                processed_bucket, datatype, segment_name)

        # Get segment metadata
        if idx == 0:
            seg_metadata = get_metadata(processed_bucket, datatype, frame)

        # Write images to file locations partitioned by day
        for camera_id, image in enumerate(frame.images):
            camera = open_dataset.CameraName.Name.Name(
                camera_id+1)  # Convert to camera name
            image_id = f"{segment_name}_{idx}_{camera}"
            file_name = f"{image_id}.jpeg"
            temp_file = f"{temp_directory}/{file_name}"
            gcp_path = f"{datatype}/images/{date}/{segment_name}/{file_name}"
            annotations["images"].append({
                "id": f"{segment_name}_{idx}_{camera}",
                "gcp_url": f"gs://{processed_bucket}/{gcp_path}",
                "file_name": file_name
            })
            if idx >= last_index:
                img_array = np.array(tf.image.decode_jpeg(image.image))
                img = Image.fromarray(img_array)
                img.save(temp_file)
                upload_blob(processed_bucket, temp_file, gcp_path)
                os.remove(temp_file)
            else:
                print(
                    f"Image at index {idx} already downloaded, skipping image download")

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

# Obtained this snippet from https://stackoverflow.com/a/20380514


def get_image_dimensions_and_resize(fname, target_size):
    img = Image.open(fname)
    width, height = img.size
    new_img = img.resize((target_size, target_size))
    new_img.save(fname, "JPEG", optimize=True)
    return width, height


def coco_to_yolo_annotations(annotations_json, labels_directory, images_directory, target_size=640, round_digits=12):
    # Category ID translations
    id_translations = {1: 0, 2: 1, 4: 2}
    # Create the directory if it doesn't exist
    if not os.path.exists(labels_directory):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(labels_directory)

    # Load the COCO annotation file
    with open(annotations_json, 'r') as f:
        coco_data = json.load(f)

    # Re-map the image array into a dictionary with keys
    image_dict = {el["id"]: {"file_name": el["file_name"]}
                  for el in coco_data["images"]}

    # Iterate through each annotation and convert it to YOLOv7 format
    img_dimensions = {}
    for annotation in tqdm(coco_data['annotations']):
        image_id = re.sub("\.jpeg$", "", annotation['image_id'])
        bbox = annotation['bbox']

        # Get the image filename without the extension
        image_filename = image_dict[image_id]['file_name'].split('.')[0]

        # If the file has been deleted, skip it
        if not os.path.exists(f"{images_directory}/{image_filename}.jpeg"):
            continue

        # Get the image dimensions and resize to the target size
        if not image_id in img_dimensions:
            img_x, img_y = get_image_dimensions_and_resize(
                f"{images_directory}/{image_filename}.jpeg", target_size)
            img_dimensions[image_id] = (img_x, img_y)
        else:
            img_x, img_y = img_dimensions[image_id]

        # Yolo format uses a the center of the bounding box as the anchor, and normalized values
        # Calculate the normalized (0-1) center coordinates and width/height of the bounding box
        x_center = round((bbox[0] + bbox[2] / 2)/img_x, round_digits)
        y_center = round((bbox[1] + bbox[3] / 2)/img_y, round_digits)
        width = round(bbox[2]/img_x, round_digits)
        height = round(bbox[3]/img_y, round_digits)

        # Get the translated category ID
        category_id = id_translations[annotation["category_id"]]

        # Write the annotation to a YOLOv7-style text file
        with open(f"{labels_directory}/{image_filename}.txt", 'a') as out_file:
            out_file.write(
                f"{category_id} {x_center} {y_center} {width} {height}\n")


def chunks(lst, n):
    out = []
    for i in range(0, len(lst), n):
        out.append(lst[i:i + n])
    return out


def download_images(annotations_json, images_directory, chunk_size=25):
    # Create the directory if it doesn't exist
    if not os.path.exists(images_directory):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(images_directory)

    # Load the COCO annotation file
    with open(annotations_json, 'r') as f:
        coco_data = json.load(f)

    DEVNULL = open(os.devnull, 'w')
    for images in tqdm(chunks(coco_data["images"], chunk_size)):
        uri_string = " ".join(['"'+img["gcp_url"]+'"' for img in images])
        command = f"gsutil -m cp {uri_string} {images_directory}"
        subprocess.call(command, shell=True, stdout=DEVNULL, stderr=DEVNULL)
