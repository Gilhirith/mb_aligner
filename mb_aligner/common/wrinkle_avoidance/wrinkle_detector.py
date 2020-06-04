import cv2
import numpy as np

class WrinkleDetector(object):

    def __init__(self, kernel_size=3, threshold=200, min_line_length=200):
        self._kernel_size = kernel_size
        self._threshold = threshold
        self._min_line_length = min_line_length

        self._initialize()

    def _initialize(self):
        self._kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (self._kernel_size, self._kernel_size));
 
        #self._dilation_kernel = np.ones((self._dilation, self._dilation), np.uint8)


    def _pre_process_img(self, img):
        res, img = cv2.threshold(img, self._threshold, 255, cv2.THRESH_TOZERO)
        img = cv2.medianBlur(img, 3)
        img = cv2.medianBlur(img, 3)
#         return img

        #img = cv2.Canny(img,self._threshold,255)
        return img

#         res, img = cv2.threshold(img, self._threshold, 255, cv2.THRESH_TOZERO)
#         skel = np.zeros_like(img)
#         done = False
#         counter = 0
#         while not done:
#             print("iteration {}".format(counter))
#             eroded = cv2.erode(img, self._kernel)
#             temp = cv2.dilate(eroded, self._kernel)
#             temp = img - temp
#             skel = skel | temp
#             temp = cv2.bitwise_or(skel, temp)
#             img = eroded
#             done = np.sum(img) == 0
#             counter += 1
#         
#         return skel

    def detect(self, img):
        # find connected components
        
        img_processed = self._pre_process_img(img)
        #show_img(img_processed)
        im2, contours, hierarchy = cv2.findContours(img_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print("found {} countours".format(len(contours)))

        # for test
        res_contours = []
        for c_idx in range(len(contours)):
            c_length = cv2.arcLength(contours[c_idx], True)
            if c_length < self._min_line_length:
                continue
#             c_area = cv2.contourArea(contours[c_idx])
#             M = cv2.moments(contours[c_idx])
#             cx = int(M['m10']/M['m00'])
#             cy = int(M['m01']/M['m00'])
#             print("({}, {}): len={}, area={}".format(cx, cy, c_length, c_area))
#             if c_area > 5000:
#                 continue
            #print("{}: {}".format(c_idx, contours[c_idx]))
            epsilon = 0.1 * c_length
            approx = cv2.approxPolyDP(contours[c_idx], epsilon, True)
            print("{}: {}".format(c_idx, approx))
            res_contours.append(approx)
            # res_contours.append(contours[c_idx])
        return res_contours

    def visualize_contours(self, img, contours, out_fname):
        out_img = cv2.cvtColor(255-img, cv2.COLOR_GRAY2BGR);
        np.random.seed(7)
        for c_idx in range(len(contours)):
            #color = (int(np.random.random_integers(0, 256)), int(np.random.random_integers(0, 256)), int(np.random.random_integers(0, 256)))
            color = (0, 0, 255)
            #cv2.drawContours(out_img, contours, c_idx, color, 3)
            cv2.drawContours(out_img, contours, c_idx, color, 3)
#             rect = cv2.minAreaRect(contours[c_idx])
#             box = cv2.boxPoints(rect)
#             box = np.int0(box)
#             cv2.drawContours(out_img, [box], 0, color, 3)
        cv2.imwrite(out_fname, out_img)

def show_img(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        img_fname = sys.argv[1]
    else:
        #img_fname = '/n/lichtmangpfs01/Instrument_drop/U19_Zebrafish/EM/w019/w019_h02_20190325_14-10-55/001_S15R1/000021/001_000021_028_2019-03-25T1414044834028.bmp'
        img_fname = '/n/lichtmangpfs01/Instrument_drop/U19_Zebrafish/EM/w019/w019_h02_20190326_00-43-52/002_S16R1/000021/002_000021_040_2019-03-26T0046443382649.bmp'

    img = cv2.imread(img_fname, 0)
    detector = WrinkleDetector(kernel_size=3, threshold=200, min_line_length=100)
    contours = detector.detect(img)
    detector.visualize_contours(img, contours, "temp_wrinkle_detector4_pre.png")




