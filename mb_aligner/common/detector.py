import cv2
from enum import Enum
import numpy as np

class FeaturesDetector(object):

    class Type(Enum):
        SIFT = 1
        ORB = 2
        SURF = 3
        BRISK = 4
        AKAZE = 5

    def __init__(self, detector_type_name, **kwargs):
        detector_type = FeaturesDetector.Type[detector_type_name]
        if detector_type == FeaturesDetector.Type.SIFT:
            self._detector_init_fn = lambda: cv2.xfeatures2d.SIFT_create(**kwargs)
        elif detector_type == FeaturesDetector.Type.ORB:
            cv2.ocl.setUseOpenCL(False) # Avoiding a bug in OpenCV 3.1
            self._detector_init_fn = lambda: cv2.ORB_create(**kwargs)
        elif detector_type == FeaturesDetector.Type.SURF:
            self._detector_init_fn = lambda: cv2.xfeatures2d.SURF_create(**kwargs)
        elif detector_type == FeaturesDetector.Type.BRISK:
            self._detector_init_fn = lambda: cv2.BRISK_create(**kwargs)
        elif detector_type == FeaturesDetector.Type.AKAZE:
            self._detector_init_fn = lambda: cv2.AKAZE_create(**kwargs)
        else:
            raise("Unknown feature detector algorithm given")

        self._init_detector()

    @staticmethod
    def get_matcher_init_fn(detector_type_name):
        detector_type = FeaturesDetector.Type[detector_type_name]
        if detector_type == FeaturesDetector.Type.SIFT:
            matcher_init_fn = lambda: cv2.BFMatcher(cv2.NORM_L2)
        elif detector_type == FeaturesDetector.Type.ORB:
            matcher_init_fn = lambda: cv2.BFMatcher(cv2.NORM_HAMMING)
        elif detector_type == FeaturesDetector.Type.SURF:
            matcher_init_fn = lambda: cv2.BFMatcher(cv2.NORM_L2)
        elif detector_type == FeaturesDetector.Type.BRISK:
            matcher_init_fn = lambda: cv2.BFMatcher(cv2.NORM_HAMMING)
        elif detector_type == FeaturesDetector.Type.AKAZE:
            matcher_init_fn = lambda: cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            raise("Unknown feature detector algorithm given")
        return matcher_init_fn
 


    def _init_detector(self):
        self._detector = self._detector_init_fn()

    def detect(self, img):
        if self._detector is None:
            self.init_detector()
        return self._detector.detectAndCompute(img, None)

