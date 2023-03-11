# Imports
import tensorflow as tf
import autonomoeye
import google.auth
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'  # to suppress tensorflow warnings

# Setup credentials
credentials, _ = google.auth.default()

# Import missing segments
while True:
    # Fetch missing segments
    missing_segments = autonomoeye.get_missing_segments("train")

    # If there aren't any, we're done
    if len(missing_segments) == 0:
        break

    # Fetch the first missing segment
    print(f"Fetching {missing_segments[0]}")
    autonomoeye.fetch_segment("waymo-processed-images",
                              "train", missing_segments[0])

print("Done!  No more missing segments.")
