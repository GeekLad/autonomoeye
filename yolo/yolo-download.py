# Sample usage:
#     python3 yolo-download.py -t train_annotations_2017-10-02_10793018113277660068_2714_540_2734_540.json -v validation_annotations_2017-10-01_10689101165701914459_2072_300_2092_300.json

# Imports
import autonomoeye
import google.auth
import tensorflow as tf
import argparse
import autonomoeye
import os
import subprocess

# Setup credentials
credentials, _ = google.auth.default()

# Parse the args
parser = argparse.ArgumentParser()
parser.add_argument('-s', dest='segment_name', required=True)
parser.add_argument('-t', dest='datatype', required=True)
args = parser.parse_args()

# Fetch & process
json = f"gs://waymo-processed-images/{args.datatype}/annotations/*/{args.segment_name}.json"
images = f"data/{args.datatype}/images"
annotations = f"data/{args.datatype}/labels"

DEVNULL = open(os.devnull, 'w')
command = f'gsutil -m cp "{json}" .'
subprocess.call(command, shell=True)
print("Downloading images")
autonomoeye.download_images(f"{args.segment_name}.json", images)
print("Building image labels")
autonomoeye.coco_to_yolo_annotations(
    f"{args.segment_name}.json", annotations, images)
os.remove(f"{args.segment_name}.json")
