#wraps OrientedSimpleBlobDetector

# largely based on:
#   https://bitbucket.org/petters/hakanardo/src/0f994ec688cfba42a4e019fa80c189a13f28b109/copencv/libopencv.pxd?at=default&fileviewer=file-view-default
#   https://bitbucket.org/petters/hakanardo/src/0f994ec688cfba42a4e019fa80c189a13f28b109/copencv/copencv.pyx?at=default&fileviewer=file-view-default

import numpy as np
cimport numpy as np
cimport cython
import cv2
cimport numpy as np # for np.ndarray
from libcpp.string cimport string
from cython.operator import dereference
#from libc.string cimport memcpy
from libcpp.vector cimport vector
from libcpp cimport bool

#ctypedef void* int_parameter
#ctypedef int_parameter two "2"
#ctypedef Point_[float, two] Point2f


cdef class WrappedMat:
    cdef Mat* _mat
    def __cinit__(self):
        self._mat = NULL
    def __dealloc__(self): 
        if self._mat:            
            del self._mat
            self._mat = NULL

np.import_array()

cdef extern from "opencv2/core/core.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat()
        Mat(Mat& m)
        Mat(int _rows, int _cols, int _type)
        Mat(int rows, int cols, int type, void* data)        
        int dims, rows, cols
        unsigned char *data
        bool isContinuous()
        int type()
        int channels()

    cdef cppclass _InputArray:
        _InputArray()
        _InputArray(Mat& m)
        
    ctypedef _InputArray& InputArray

    cdef cppclass Point_[_Tp]:
        _Tp x, y
    #ctypedef Point_[float] Point2f

cdef extern from "opencv2/core/types_c.h":
    enum:
        CV_8UC1
        CV_8UC2
        CV_8UC3
        CV_8UC4
        CV_32FC1
        CV_64FC1
        CV_32SC1
        CV_32SC2
        CV_32SC3
        CV_32SC4

cdef extern from "opencv2/features2d/features2d.hpp" namespace "cv":
    cdef cppclass KeyPoint:
        Point_[float] pt
        float size
        float angle


# cdef extern from "opencv2/core/types.hpp" namespace "cv":
#     struct KeyPoint:
#         pass
# 
# cdef extern from "opencv2/core/mat.hpp" namespace "cv":
#     cdef cppclass Mat:
#         Mat() except +
#         void create(int, int, int)
#         void* data
#     #struct InputArray:
#     #    pass

# cdef void ary2cvMat(np.ndarray ary, Mat& out):
#     assert(ary.ndim==2 and ary.shape[2]==2, "ASSERT::2channel grayscale only!!")
#      
#     cdef np.ndarray[np.uint8_t, ndim=2, mode = 'c'] np_buff = np.ascontiguousarray(ary, dtype = np.uint8)
#     cdef unsigned int* im_buff = <unsigned int*> np_buff.data
#     cdef int r = ary.shape[0]
#     cdef int c = ary.shape[1]
#     out.create(r, c, CV_8UC3)
#     memcpy(out.data, im_buff, r*c*3)
cdef Mat numpy2mat(img):
    cdef WrappedMat wmat
    if hasattr(img, '_wmat'):
        wmat = img._wmat
        return dereference(wmat._mat)
    else:
        assert img.flags.contiguous
        if len(img.shape) == 3 and img.shape[2] == 3 and img.dtype == 'B':
            return Mat(img.shape[0], img.shape[1], CV_8UC3, np.PyArray_DATA(img))
        elif len(img.shape) == 2 and img.dtype == 'd':
            return Mat(img.shape[0], img.shape[1], CV_64FC1, np.PyArray_DATA(img))
        elif len(img.shape) == 2 and img.dtype == 'B':
            return Mat(img.shape[0], img.shape[1], CV_8UC1, np.PyArray_DATA(img))
        else:
            assert False

cdef extern from "OrientedSimpleBlobDetector.hpp":# namespace "cv":


    cdef cppclass OrientedSimpleBlobDetector:
        cppclass Params:
            float thresholdStep
            float minThreshold
            float maxThreshold
            int minRepeatability
            float minDistBetweenBlobs

            bool filterByColor
            cython.uchar blobColor

            bool filterByArea
            float minArea, maxArea

            bool filterByCircularity
            float minCircularity, maxCircularity

            bool filterByInertia
            float minInertiaRatio, maxInertiaRatio

            bool filterByConvexity
            float minConvexity, maxConvexity
 

        OrientedSimpleBlobDetector(OrientedSimpleBlobDetector.Params& params)
        OrientedSimpleBlobDetector()
        void detect(InputArray image, vector[KeyPoint]& keypoints, InputArray mask)
        void detect(InputArray image, vector[KeyPoint]& keypoints)
#        @staticmethod
#        #OrientedSimpleBlobDetector create(OrientedSimpleBlobDetector_Params)
#        OrientedSimpleBlobDetector create()

    ctypedef OrientedSimpleBlobDetector.Params OrientedSimpleBlobDetector_Params

cdef class PyOrientedSimpleBlobDetector_Params:
    cdef OrientedSimpleBlobDetector_Params *c_params
    def __cinit__(self):
        self.c_params = new OrientedSimpleBlobDetector_Params()

    def __dealloc__(self):
        del self.c_params

    property thresholdStep:
        def __get__(self):
            return self.c_params.thresholdStep
        def __set__(self, v):
            self.c_params.thresholdStep = v

    property minThreshold:
        def __get__(self):
            return self.c_params.minThreshold
        def __set__(self, v):
            self.c_params.minThreshold = v

    property maxThreshold:
        def __get__(self):
            return self.c_params.maxThreshold
        def __set__(self, v):
            self.c_params.maxThreshold = v

    property minRepeatability:
        def __get__(self):
            return self.c_params.minRepeatability
        def __set__(self, v):
            self.c_params.minRepeatability = v

    property minDistBetweenBlobs:
        def __get__(self):
            return self.c_params.minDistBetweenBlobs
        def __set__(self, v):
            self.c_params.minDistBetweenBlobs = v

    property filterByColor:
        def __get__(self):
            return self.c_params.filterByColor
        def __set__(self, v):
            self.c_params.filterByColor = v

    property blobColor:
        def __get__(self):
            return self.c_params.blobColor
        def __set__(self, v):
            self.c_params.blobColor = v

    property filterByArea:
        def __get__(self):
            return self.c_params.filterByArea
        def __set__(self, v):
            self.c_params.filterByArea = v

    property minArea:
        def __get__(self):
            return self.c_params.minArea
        def __set__(self, v):
            self.c_params.minArea = v

    property maxArea:
        def __get__(self):
            return self.c_params.maxArea
        def __set__(self, v):
            self.c_params.maxArea = v

    property filterByCircularity:
        def __get__(self):
            return self.c_params.filterByCircularity
        def __set__(self, v):
            self.c_params.filterByCircularity = v

    property minCircularity:
        def __get__(self):
            return self.c_params.minCircularity
        def __set__(self, v):
            self.c_params.minCircularity = v

    property maxCircularity:
        def __get__(self):
            return self.c_params.maxCircularity
        def __set__(self, v):
            self.c_params.maxCircularity = v

    property filterByInertia:
        def __get__(self):
            return self.c_params.filterByInertia
        def __set__(self, v):
            self.c_params.filterByInertia = v

    property minInertiaRatio:
        def __get__(self):
            return self.c_params.minInertiaRatio
        def __set__(self, v):
            self.c_params.minInertiaRatio = v

    property maxInertiaRatio:
        def __get__(self):
            return self.c_params.maxInertiaRatio
        def __set__(self, v):
            self.c_params.maxInertiaRatio = v

    property filterByConvexity:
        def __get__(self):
            return self.c_params.filterByConvexity
        def __set__(self, v):
            self.c_params.filterByConvexity = v

    property minConvexity:
        def __get__(self):
            return self.c_params.minConvexity
        def __set__(self, v):
            self.c_params.minConvexity = v

    property maxConvexity:
        def __get__(self):
            return self.c_params.maxConvexity
        def __set__(self, v):
            self.c_params.maxConvexity = v



    
cdef class PyOrientedSimpleBlobDetector:
    cdef OrientedSimpleBlobDetector c_detector

    def __cinit__(self, params=None):
        cdef OrientedSimpleBlobDetector_Params c_params
        if params is None:
            self.c_detector = OrientedSimpleBlobDetector()
        else:
            #self.c_params = params.c_params
            #cdef OrientedSimpleBlobDetector_Params p = params.c_params
            c_params.thresholdStep = params.thresholdStep
            c_params.minThreshold = params.minThreshold
            c_params.maxThreshold = params.maxThreshold
            c_params.minRepeatability = params.minRepeatability
            c_params.minDistBetweenBlobs = params.minDistBetweenBlobs
            c_params.filterByColor = params.filterByColor
            c_params.blobColor = params.blobColor
            c_params.filterByArea = params.filterByArea
            c_params.minArea = params.minArea
            c_params.maxArea = params.maxArea
            c_params.filterByCircularity = params.filterByCircularity
            c_params.minCircularity = params.minCircularity
            c_params.maxCircularity = params.maxCircularity
            c_params.filterByInertia = params.filterByInertia
            c_params.minInertiaRatio = params.minInertiaRatio
            c_params.maxInertiaRatio = params.maxInertiaRatio
            c_params.filterByConvexity = params.filterByConvexity
            c_params.minConvexity = params.minConvexity
            c_params.maxConvexity = params.maxConvexity
            self.c_detector = OrientedSimpleBlobDetector(c_params)

    def detect(self, img):
        cdef vector[KeyPoint] kps
        cdef Mat m
        cdef int kp_num
        self.c_detector.detect(InputArray(numpy2mat(img)), kps)
        kps_list = []
        kps_num = kps.size()
        for i in range(kps_num):
            kp = kps[i]
            kps_list.append(cv2.KeyPoint(kp.pt.x, kp.pt.y, kp.size, kp.angle))
        return kps_list

