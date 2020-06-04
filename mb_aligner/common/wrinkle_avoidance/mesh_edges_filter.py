import cv2
from scipy.spatial import Delaunay
import numpy as np
from mb_aligner.common import utils
from mb_aligner.common.wrinkle_avoidance.wrinkle_detector import WrinkleDetector
from shapely.geometry import Point, LineString, MultiLineString

class MeshEdgesFilter(object):

    def __init__(self, mesh_tri, contours=None):
        self._mesh_tri = mesh_tri
        self._points = self._mesh_tri.points
        self._simplices = self._mesh_tri.simplices.astype(np.uint32)


        if contours is not None:
            self.set_contours(contours)

    def set_contours(self, contours):
        # add all contours to a line collection
        contours = [[p[0] for p in cnt] for cnt in contours]
        self._wrinkle_lines = MultiLineString(contours)



    def filter_mesh_by_wrinkles(self):
        # find unique edges - iterate over all simplices triangle edges
        def _filter_specifc_simplices_edge(cur_simplex_edge_indices, simplices_mask, points, wrinkle_lines):
            cur_simplex_edge_mask = np.zeros((len(cur_simplex_edge_indices), ), dtype=np.bool)
            for simplex_idx, simplex_edge_idxs in enumerate(cur_simplex_edge_indices):
                edge_pts = points[simplex_edge_idxs]
                edge_line = LineString(edge_pts)
                if wrinkle_lines.intersection(edge_line):
                    print("Found an intersection of simplex_idx: {}, edge_pts: {}".format(simplex_idx, edge_pts))
                    cur_simplex_edge_indices[simplex_idx] = True
                    simplices_mask[simplex_idx] = True
            return cur_simplex_edge_mask
                
        relevant_edges_indices = []
        simplices_mask = np.zeros((len(self._simplices), ), dtype=np.bool)
        edges_indices = np.vstack((
                self._simplices[:, :2][~_filter_specifc_simplices_edge(self._simplices[:, :2], simplices_mask, self._points, self._wrinkle_lines)],
                self._simplices[:, 1:][~_filter_specifc_simplices_edge(self._simplices[:, 1:], simplices_mask, self._points, self._wrinkle_lines)],
                self._simplices[:, [0, 2]][~_filter_specifc_simplices_edge(self._simplices[:, [0, 2]], simplices_mask, self._points, self._wrinkle_lines)]
            ))
        edges_indices = np.vstack({tuple(sorted(tuple(row))) for row in edges_indices}).astype(np.uint32)
        
           
            
#         edge_indices = np.vstack((simplices[:, :2],
#                                   simplices[:, 1:],
#                                   simplices[:, [0, 2]]))
#         edge_indices = np.vstack({tuple(sorted(tuple(row))) for row in edge_indices}).astype(np.uint32)
# 
#         edges_mask = np.zeros((len(edge_indices), ), dtype=np.bool)
# 
# #         for cnt in contours:
# #             for p_idx in range(len(cnt) - 1):
# #                 wrinkle_line = [cnt[p_idx][0], cnt[p_idx][1]]
#                 
#                 
# 
#         # iterate over edges and check for intersection with contours
#         for edge_idx, edge_idxs in enumerate(edge_indices):
#             edge_pts = points[edge_idxs]
#             edge_line = LineString(edge_pts)
#             if wrinkle_lines.intersection(edge_line):
#                 print("Found an intersection of edge_idx: {}, edge_pts: {}".format(edge_idx, edge_pts))
#                 edges_mask[edge_idx] = True

        return edges_indices, self._simplices[~simplices_mask], ~simplices_mask

    def filter_edges_by_wrinkles(self, edges_pts):
        """
        Returns a mask where indices corresponding to edges that cross wrinkles are True
        """

        edges_mask = np.zeros((len(edges_pts), ), dtype=np.bool)

        # iterate over all edges, and for each edge find if it crosses the wrinkle_lines
        for edge_idx, edge_pts in enumerate(edge_pts):
            edge_line = LineString(edge_pts)
            if self._wrinkle_lines.intersection(edge_line):
                print("Found an intersection of edge_idx: {}, edge_pts: {}".format(edge_idx, edge_pts))
                edges_mask[edge_idx] = True

        return edges_mask
 

#     def _initialize(self):
#         self._kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (self._kernel_size, self._kernel_size));
#  
#         #self._dilation_kernel = np.ones((self._dilation, self._dilation), np.uint8)
# 
# 
#     def _pre_process_img(self, img):
#         res, img = cv2.threshold(img, self._threshold, 255, cv2.THRESH_TOZERO)
#         img = cv2.medianBlur(img, 3)
#         img = cv2.medianBlur(img, 3)
# #         return img
# 
#         #img = cv2.Canny(img,self._threshold,255)
#         return img
# 
# #         res, img = cv2.threshold(img, self._threshold, 255, cv2.THRESH_TOZERO)
# #         skel = np.zeros_like(img)
# #         done = False
# #         counter = 0
# #         while not done:
# #             print("iteration {}".format(counter))
# #             eroded = cv2.erode(img, self._kernel)
# #             temp = cv2.dilate(eroded, self._kernel)
# #             temp = img - temp
# #             skel = skel | temp
# #             temp = cv2.bitwise_or(skel, temp)
# #             img = eroded
# #             done = np.sum(img) == 0
# #             counter += 1
# #         
# #         return skel
# 
#     def detect(self, img):
#         # find connected components
#         
#         img_processed = self._pre_process_img(img)
#         show_img(img_processed)
#         im2, contours, hierarchy = cv2.findContours(img_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         print("found {} countours".format(len(contours)))
# 
#         # for test
#         res_contours = []
#         for c_idx in range(len(contours)):
#             c_length = cv2.arcLength(contours[c_idx], True)
#             if c_length < self._min_line_length:
#                 continue
# #             c_area = cv2.contourArea(contours[c_idx])
# #             M = cv2.moments(contours[c_idx])
# #             cx = int(M['m10']/M['m00'])
# #             cy = int(M['m01']/M['m00'])
# #             print("({}, {}): len={}, area={}".format(cx, cy, c_length, c_area))
# #             if c_area > 5000:
# #                 continue
#             #print("{}: {}".format(c_idx, contours[c_idx]))
#             epsilon = 0.1 * c_length
#             approx = cv2.approxPolyDP(contours[c_idx], epsilon, True)
#             print("{}: {}".format(c_idx, approx))
#             res_contours.append(approx)
#         return res_contours
# 
#     def visualize_contours(self, img, contours, out_fname):
#         out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR);
#         np.random.seed(7)
#         for c_idx in range(len(contours)):
#             color = (int(np.random.random_integers(0, 256)), int(np.random.random_integers(0, 256)), int(np.random.random_integers(0, 256)))
#             #cv2.drawContours(out_img, contours, c_idx, color, 3)
#             cv2.drawContours(out_img, contours, c_idx, color, 3)
# #             rect = cv2.minAreaRect(contours[c_idx])
# #             box = cv2.boxPoints(rect)
# #             box = np.int0(box)
# #             cv2.drawContours(out_img, [box], 0, color, 3)
#         cv2.imwrite(out_fname, out_img)
# 

# def visualize_filtered_mesh(mesh_tri, edges_mask, simplices_mask, img):
#     import matplotlib.pyplot as plt
# 
#     simplices = mesh_tri.simplices
#     # find unique edges
#     edge_indices = np.vstack((simplices[:, :2],
#                               simplices[:, 1:],
#                               simplices[:, [0, 2]]))
#     edge_indices = np.vstack({tuple(sorted(tuple(row))) for row in edge_indices}).astype(np.uint32)
# 
#     plt.imshow(255-img, cmap='gray')
# 
#     points = mesh_tri.points
#     for edge_mask, edge_idxs in zip(edges_mask, edge_indices):
#         if not edge_mask:
#             edge_pts = points[edge_idxs]
#             plt.plot(edge_pts[:, 0], edge_pts[:, 1], 'ro-', linewidth=2, markersize=2)
# 
# 
# #    plt.triplot(points[:,0], points[:,1], mesh_tri.simplices.copy())
# #    plt.plot(points[:,0], points[:,1], 'o')
#     plt.show()

def visualize_filtered_mesh(mesh_tri, filterd_edges_indices, filtered_simplices, img):
    import matplotlib.pyplot as plt

    plt.imshow(255-img, cmap='gray')

    points = mesh_tri.points
    # find unique edges
    edge_indices = np.vstack((filtered_simplices[:, :2],
                              filtered_simplices[:, 1:],
                              filtered_simplices[:, [0, 2]]))
    edge_indices = np.vstack({tuple(sorted(tuple(row))) for row in edge_indices}).astype(np.uint32)

    filtered_edges_pts = points[edge_indices]
    plt.triplot(points[:,0], points[:,1], filtered_simplices, color='r')
    plt.plot(points[:,0], points[:,1], 'ro', markersize=2)
    #plt.plot(filtered_edges_pts[:, 0], filtered_edges_pts[:, 1], 'ro-', linewidth=2, markersize=2)

#    plt.triplot(points[:,0], points[:,1], mesh_tri.simplices.copy())
#    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()



if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        img_fname = sys.argv[1]
    else:
        img_fname = '/n/lichtmangpfs01/Instrument_drop/U19_Zebrafish/EM/w019/w019_h02_20190325_14-10-55/001_S15R1/000021/001_000021_028_2019-03-25T1414044834028.bmp'
        #img_fname = '/n/lichtmangpfs01/Instrument_drop/U19_Zebrafish/EM/w019/w019_h02_20190326_00-43-52/002_S16R1/000021/002_000021_040_2019-03-26T0046443382649.bmp'

    img = cv2.imread(img_fname, 0)
    detector = WrinkleDetector(kernel_size=3, threshold=200, min_line_length=100)
    contours = detector.detect(img)
    #detector.visualize_contours(img, contours, "temp_wrinkle_detector4.png")

    hex_grid = utils.generate_hexagonal_grid([0, img.shape[1], 0, img.shape[0]], 500)
    mesh_tri = Delaunay(hex_grid)
    edges_filter = MeshEdgesFilter(mesh_tri)

    filtered_edges_indices, filtered_simplices = edges_filter.filter_by_wrinkles(contours)

    visualize_filtered_mesh(mesh_tri, filtered_edges_indices, filtered_simplices, img)



