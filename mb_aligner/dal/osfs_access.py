from io import BytesIO
import os



class OSFSAccess(object):

    def __init__(self):
        pass

    def file_exists(self, file_path):
        return os.path.isfile(file_path) 

    def read_binary_file(self, file_path):
        with open(file_path, 'rb') as in_f:
            buf = in_f.read()
        return buf

    def read_text_file(self, file_path):
        with open(file_path, 'rt') as in_f:
            lines = in_f.readlines()
        return lines


    def list_directories(self, prefix):
        dirs = set()
        for entry in os.listdir(prefix):
            if os.path.isdir(entry):
                dirs.add(entry)
        return dirs

