import os
import subprocess
import re
import tensorflow as tf
from google.cloud import storage
from tqdm import tqdm
import json
import shutil
import argparse

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def fetch_segment(segment, datatype):
    # Import the Segment
    remote_datatype = datatype if datatype != "train" else "training"
    uri = f"gs://waymo_open_dataset_v_1_4_1/individual_files/{remote_datatype}/segment-{segment}_with_camera_labels.tfrecord"
    return tf.data.TFRecordDataset(uri, compression_type='')


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


def chunks(lst, n):
    out = []
    for i in range(0, len(lst), n):
        out.append(lst[i:i + n])
    return out

def create_missing_dir(dir):
    # Create the directory if it doesn't exist
    if not os.path.exists(dir):
        os.makedirs(dir)

def download_images(annotations_json_file, images_directory, chunk_size=25, nth_frame=10):
    '''Downloads the images from a JSON annotations file'''
    # Create the directory if it doesn't exist
    create_missing_dir(images_directory)

    # Load the COCO annotation file
    with open(annotations_json_file, 'r') as f:
        coco_data = json.load(f)

    nth_images = []
    for i, image in enumerate(coco_data["images"]):
        idx = int(re.findall(r"\d+(?:_\d+){4}_(\d+)_", image['id'])[0])
        if idx % nth_frame == 0:
            nth_images.append(image)

    DEVNULL = open(os.devnull, 'w')
    for images in tqdm(chunks(nth_images, chunk_size)):
        uri_string = " ".join(['"' + img["gcp_url"] + '"' for img in images])
        command = f"gsutil -m cp {uri_string} {images_directory}"
        subprocess.call(command, shell=True, stdout=DEVNULL, stderr=DEVNULL)

def download_annotations_and_images(annotations_directory, images_directory, datatype="train", chunk_size=25, nth_frame=10):
    # Strip any trailing slashes
    annotations_directory = re.sub(r"\/$", "", annotations_directory)

    # Create the directory if it doesn't exist
    create_missing_dir(annotations_directory)
    annotation_uris = get_annotation_uris(datatype)
    DEVNULL = open(os.devnull, 'w')

    print("Downloading annotations")
    for annotations in tqdm(chunks(annotation_uris, chunk_size)):
        uri_string = " ".join(['"' + annotation + '"' for annotation in annotations])
        command = f"gsutil -m cp {uri_string} {annotations_directory}"
        subprocess.call(command, shell=True, stdout=DEVNULL, stderr=DEVNULL)

    print("Downloading images")
    # from os import listdir
    # from os.path import isfile, join
    annotation_files = [f for f in os.listdir(annotations_directory) if os.path.isfile(os.path.join(annotations_directory, f))]
    for annotation_file in tqdm(annotation_files):
        download_images(f"{annotations_directory}/{annotation_file}", images_directory, chunk_size, nth_frame)

if __name__ == '__main__':
    # Read in script arguments
    parser = argparse.ArgumentParser(
        description='Download images and annotations from the project bucket')
    parser.add_argument('-t', '--datatype', help='Datatype to load',
                        dest='datatype', default='train', choices=['train', 'validation'])
    parser.add_argument('-a', '--annotations_directory', help='Annotations directory',
                        dest='annotations_directory', default='data/train/annotations')
    parser.add_argument('-i', '--images_directory', help='Images directory',
                        dest='images_directory', default='data/train/images')
    parser.add_argument('-c', '--chunk_size', help='How many files to download at one time',
                        dest='chunk_size', default=25)
    parser.add_argument('-n', '--nth_frame', help='How many frames to skip (i.e. every 10th frame)',
                        dest='nth_frame', default=10)
    args = parser.parse_args()

    download_annotations_and_images(args.annotations_directory, args.images_directory, args.datatype, int(args.chunk_size), int(args.nth_frame))
