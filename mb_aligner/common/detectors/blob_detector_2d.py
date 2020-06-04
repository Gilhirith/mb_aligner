import cv2
import numpy as np
import pyximport
pyximport.install()
from mb_aligner.common.detectors.blob_detector_impl.oriented_simple_blob_detector import PyOrientedSimpleBlobDetector, PyOrientedSimpleBlobDetector_Params

class BlobDetector2D(object):
    '''
    A detector that is based on OpenCV's simple blob detector and SIFT descriptor
    '''
    def __init__(self, **kwargs):
        # create the blob detector and the sift detector (for the descriptor)
        blob_params = kwargs.get("blob_params", {})
        sift_params = kwargs.get("sift_params", {})
        self._use_clahe = kwargs.get("use_clahe", True)

        self._blob_detector = BlobDetector2D._create_blob_detector(**blob_params)
        self._sift_detector = cv2.xfeatures2d.SIFT_create(**sift_params)
        self._clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))

    @staticmethod
    def _create_blob_detector(**kwargs):
        args = {}
        if kwargs is not None:
            args = kwargs
        # Create the blob detector (large blobs, and not very circular and convex)
        params = PyOrientedSimpleBlobDetector_Params()
        #params.filterByColor = True
        #params.blobColor = 255
        if "blobColor" in args:
            params.filterByColor = True
            params.blobColor = args.get("blobColor")
        #params.minThreshold = 100
        #params.maxThreshold = 250
        params.filterByArea = True
        params.minArea = args.get("minArea", 100)
        params.maxArea = args.get("maxArea", 1000)
        #params.filterByCircularity = False
        params.filterByCircularity = True
        #params.minCircularity = 0.2
        params.minCircularity = args.get("minCircularity", 0.2)
        params.filterByConvexity = False
        if "minConvexity" in args:
            params.filterByConvexity = True
            params.minConvexity = args.get("minConvexity")
        #params.filterByConvexity = True
        #params.minConvexity = 0.2
        params.filterByInertia = False
        if "minInertiaRatio" in args:
            params.filterByInertia = True
            params.minInertiaRatio = args.get("minInertiaRatio")
        detector = PyOrientedSimpleBlobDetector(params)
        return detector

    def detectAndCompute(self, img, kps=None):
        '''
        Performs the detection and description on the given image.
        Note that the kps parameter is unused (only here to be coherent with OpenCV's feature detector methods)
        '''
        img_filtered = img
        # apply a clahe filter
        if self._use_clahe:
            img_filtered = self._clahe.apply(img)
            #img_med_clahe = clahe.apply(img_med)
            #img_clahe = self._clahe.apply(img)
            #img_equ = cv2.equalizeHist(img)

        # Detect round blobs
        #blob_kps = blob_detector.detect(img_med_clahe)
        blob_kps = self._blob_detector.detect(img_filtered)
        #blob_kps = blob_detector.detect(img_equ)

        if blob_kps is None or len(blob_kps) == 0:
            blob_descs = []
            blob_pts = []
        else:
            # Use the sift detector to extract the features in the blobs
            blob_pts, blob_descs = self._sift_detector.compute(img, blob_kps)
            blob_descs = np.array(blob_descs, dtype=np.uint8)

        return blob_pts, blob_descs
        
        
    @staticmethod
    def create_detector(**kwargs):
        return BlobDetector2D(**kwargs)

    @staticmethod
    def create_matcher():
        return cv2.BFMatcher(cv2.NORM_L2)

