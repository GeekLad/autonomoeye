'''
Contains various helper functions for Waymo dataset processing
'''

# Required imports
from google.cloud import storage

def get_uris(bucket_name: str, folder: str, credentials=None) -> list[str]:
  '''Returns a list of URIs for a Google storage bucket and path'''
  storage_client = storage.Client(credentials=credentials)
  bucket = storage_client.get_bucket(bucket_name)
  uris = []
  blobs = bucket.list_blobs(prefix=folder)
  for blob in blobs:
    uris.append(f'gs://{bucket_name}/{blob.name}')
  return uris[1:]
