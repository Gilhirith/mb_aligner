import cv2
import numpy as np
from .fs_access import FSAccess

def read_image_file(fname_url):
    with FSAccess(fname_url, True) as image_f:
        img_buf = image_f.read()
        np_arr = np.frombuffer(img_buf, np.uint8)
        img = cv2.imdecode(np_arr, 0)
    return img

def write_image_file(fname_url, img):
    np_arr = np.getbuffer(img)
    img_buf = cv2.imencode(os.path.splitext(fname_url)[1], np_arr)
    with FSAccess(fname_url, True, read=False) as image_f:
        image_f.write(img_buf)
