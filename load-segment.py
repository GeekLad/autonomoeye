# Sample usage:
#     python3 load-segment.py -s 967082162553397800_5102_900_5122_900 -b waymo-processed-images -d validation

# Imports
import autonomoeye
import google.auth
import tensorflow as tf
import argparse
import autonomoeye

# Setup credentials
credentials, _ = google.auth.default()

# Parse the args
parser = argparse.ArgumentParser()
parser.add_argument('-b', dest='bucket', required=True)
parser.add_argument('-s', dest='segment', required=True)
parser.add_argument('-d', dest='datatype', required=True)
args = parser.parse_args()

# Fetch & process
autonomoeye.fetch_segment(args.bucket, args.datatype, args.segment)
