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

def get_image_index(image_path):
    try:
        return int(re.findall(r"\d+(?:_\d+){4}_(\d+)_", image_path)[0])
    except:
        return -1

def download_images(annotations_json_file, images_directory, chunk_size=25):
    '''Downloads the images from a JSON annotations file'''
    # Create the directory if it doesn't exist
    create_missing_dir(images_directory)

    # Get the existing images
    existing_images = list_files(images_directory)

    # Load the COCO annotation file
    with open(annotations_json_file, 'r') as f:
        coco_data = json.load(f)

    image_data = [img for img in coco_data["images"] if img["id"]+".jpeg" not in existing_images]

    DEVNULL = open(os.devnull, 'w')
    for images in tqdm(chunks(image_data, chunk_size), unit_scale=chunk_size):
        uri_string = " ".join(['"' + img["gcp_url"] + '"' for img in images])
        command = f"gsutil -m cp {uri_string} {images_directory}"
        subprocess.call(command, shell=True, stdout=DEVNULL, stderr=DEVNULL)

def delete_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))    


def combine_annotations(annotations_directory, nth_frame=25):
    combined_annotations = {}
    image_ids = set()

    annotation_files = [f for f in os.listdir(annotations_directory) if os.path.isfile(os.path.join(annotations_directory, f))]
    for annotation_file in tqdm(annotation_files):
        try:
            with open(f"{annotations_directory}/{annotation_file}", 'r') as f:
                coco_data = json.load(f)

            if "licenses" not in combined_annotations:
                combined_annotations["info"] = coco_data["info"]
                combined_annotations["licenses"] = coco_data["licenses"]
                combined_annotations["categories"] = coco_data["categories"]
                combined_annotations["images"] = []
                combined_annotations["annotations"] = []
            
            for image in coco_data["images"]:
                idx = get_image_index(image['id'])
                if idx % nth_frame == 0 and image["id"] not in image_ids:
                    combined_annotations["images"].append(image)
                    image_ids.add(image["id"])

            for annotation in coco_data["annotations"]:
                if annotation["image_id"] in image_ids:
                    combined_annotations["annotations"].append(annotation)
        except:
            print(f"Warning: {annotations_directory}/{annotation_file} is not a valid JSON file")

    delete_files(annotations_directory)

    print("Saving combined annotations file (this may take a minute)")
    with open(f"{annotations_directory}/combined_annotations.json", "w") as f:
        json.dump(combined_annotations, f)

def list_files(path):
    if not os.path.exists(path):
        return []
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def download_annotations_and_images(annotations_directory, images_directory, datatype="train", chunk_size=25, nth_frame=10):
    # Strip any trailing slashes
    annotations_directory = re.sub(r"\/$", "", annotations_directory)

    # Create the directory if it doesn't exist
    create_missing_dir(annotations_directory)
    annotation_uris = get_annotation_uris(datatype)
    DEVNULL = open(os.devnull, 'w')

    # Get existing annotations
    existing_annotations = list_files(annotations_directory)
    existing_annotations = [get_segment_name(annotation) for annotation in existing_annotations]

    if len(existing_annotations) > 0 and f"combined_annotations.json" in existing_annotations:
        print("Annotations already downloaded and combined")
    else:
        print("Downloading annotations")
        annotation_uris = [annotation for annotation in annotation_uris if get_segment_name(annotation) not in existing_annotations]
        for annotations in tqdm(chunks(annotation_uris, chunk_size), unit_scale=chunk_size):
            uri_string = " ".join(['"' + annotation + '"' for annotation in annotations])
            command = f"gsutil -m cp {uri_string} {annotations_directory}"
            subprocess.call(command, shell=True, stdout=DEVNULL, stderr=DEVNULL)

        print("Combining annotations")
        combine_annotations(annotations_directory, nth_frame)

    print("Downloading images")
    download_images(f"{annotations_directory}/combined_annotations.json", images_directory, chunk_size)

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
