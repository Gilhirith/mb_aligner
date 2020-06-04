import cv2
from scipy.spatial import Delaunay
import numpy as np
from mb_aligner.common import utils
from mb_aligner.common.wrinkle_avoidance.wrinkle_detector import WrinkleDetector
from shapely.geometry import Point, LineString, MultiLineString

class EdgesFilter(object):

    def __init__(self, contours=None):

        if contours is not None:
            self.set_contours(contours)

    def set_contours(self, contours):
        # add all contours to a line collection
        contours = [[p[0] for p in cnt] for cnt in contours]
        self._wrinkle_lines = MultiLineString(contours)



    def filter_mesh_by_wrinkles(self, mesh_tri):
        mesh_tri = mesh_tri
        points = self._mesh_tri.points
        simplices = self._mesh_tri.simplices.astype(np.uint32)


        # find unique edges - iterate over all simplices triangle edges
        def _filter_specifc_simplices_edge(cur_simplex_edge_indices, simplices_mask, points, wrinkle_lines):
            cur_simplex_edge_mask = np.zeros((len(cur_simplex_edge_indices), ), dtype=np.bool)
            for simplex_idx, simplex_edge_idxs in enumerate(cur_simplex_edge_indices):
                edge_pts = points[simplex_edge_idxs]
                edge_line = LineString(edge_pts)
                if wrinkle_lines.intersection(edge_line):
                    #print("Found an intersection of simplex_idx: {}, edge_pts: {}".format(simplex_idx, edge_pts))
                    cur_simplex_edge_indices[simplex_idx] = True
                    simplices_mask[simplex_idx] = True
            return cur_simplex_edge_mask
                
        relevant_edges_indices = []
        simplices_mask = np.zeros((len(simplices), ), dtype=np.bool)
        edges_indices = np.vstack((
                simplices[:, :2][~_filter_specifc_simplices_edge(simplices[:, :2], simplices_mask, points, self._wrinkle_lines)],
                simplices[:, 1:][~_filter_specifc_simplices_edge(simplices[:, 1:], simplices_mask, points, self._wrinkle_lines)],
                simplices[:, [0, 2]][~_filter_specifc_simplices_edge(simplices[:, [0, 2]], simplices_mask, points, self._wrinkle_lines)]
            ))
        edges_indices = np.vstack({tuple(sorted(tuple(row))) for row in edges_indices}).astype(np.uint32)
        
        return edges_indices, simplices[~simplices_mask], ~simplices_mask

    def filter_edges_by_wrinkles(self, edges_pts):
        """
        Returns a mask where indices corresponding to edges that cross wrinkles are True
        """

        edges_mask = np.zeros((len(edges_pts), ), dtype=np.bool)

        # iterate over all edges, and for each edge find if it crosses the wrinkle_lines
        for edge_idx, edge_pts in enumerate(edges_pts):
            edge_line = LineString(edge_pts)
            if self._wrinkle_lines.intersection(edge_line):
                #print("Found an intersection of edge_idx: {}, edge_pts: {}".format(edge_idx, edge_pts))
                edges_mask[edge_idx] = True

        return edges_mask
 
    def get_min_lengths(self, pts):
        min_lengths = [Point(pt).distance(self._wrinkle_lines) for pt in pts]
        return min_lengths


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



