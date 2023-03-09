# Sample usage:
#     python3 load-segment.py -s 967082162553397800_5102_900_5122_900 -b waymo-processed-images -d validation

# Imports
import autonomoeye
import google.auth
import tensorflow as tf
import argparse

# Setup credentials
credentials, _ = google.auth.default()

# Parse the args
parser = argparse.ArgumentParser()
parser.add_argument('-b', dest='bucket')
parser.add_argument('-s', dest='segment')
parser.add_argument('-d', dest='datatype')
args=parser.parse_args()

# Import the Segment
uri = f"gs://waymo_open_dataset_v_1_4_1/individual_files/{args.datatype}/segment-{args.segment}_with_camera_labels.tfrecord"
dataset = tf.data.TFRecordDataset(uri, compression_type='')

# initialize annnotations dictionary
annotations = autonomoeye.initialize_annotations_dict()

# process images and annotations in frame and return metadata
print(f"Loading {uri}")
metadata = autonomoeye.process_segment(dataset, annotations, args.bucket, args.datatype, "tmp")

