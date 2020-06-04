from google.cloud import storage
from io import BytesIO



class GCSAccess(object):

    def __init__(self, bucket_name):
        self._bucket_name = bucket_name
        self._storage_client = storage.Client()
        self._bucket = self._storage_client.get_bucket(bucket_name)
        assert self._bucket is not None

    def file_exists(self, file_path):
        blob = self._bucket.get_blob(file_path)
        return blob.exists()

    def read_binary_file(self, file_path):
        blob = self._bucket.get_blob(file_path)
        if blob is None:
            # Maybe throw an exception...
            return None
        string_buffer = BytesIO()
        blob.download_to_file(string_buffer)
        string_buffer.seek(0)
        return string_buffer

    def read_text_file(self, file_path):
        blob = self._bucket.get_blob(file_path)
        if blob is None:
            # Maybe throw an exception...
            return None
        string_buffer = BytesIO()
        blob.download_to_file(string_buffer)
        string_buffer.seek(0)
        return string_buffer


    def list_directories(self, prefix):
        # from https://github.com/GoogleCloudPlatform/google-cloud-python/issues/920
        iterator = self._bucket.list_blobs(prefix=prefix, delimiter='/')
        prefixes = set()
        for page in iterator.pages:
            #print(page, page.prefixes)
            prefixes.update(page.prefixes)
        return prefixes


