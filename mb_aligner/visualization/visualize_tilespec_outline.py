import numpy as np
import sys
import shapely.geometry
import shapely.ops
import ujson as json
from descartes.patch import PolygonPatch
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
from collections import defaultdict
import os

class TilespecVisualizer(object):

    def __init__(self, scale=0.1):
        self._scale = 0.1

    def _get_pts_polygons(self, all_tiles_pts):
        polygons = [
            shapely.geometry.Polygon([
                t_pts[0],
                t_pts[1],
                t_pts[2],
                t_pts[3],
                t_pts[0]])
            for t_pts in all_tiles_pts]

        return polygons

    def _get_pts_unified_polygon(self, all_tiles_pts):
        polygons = self._get_pts_polygons(all_tiles_pts)
        return shapely.ops.cascaded_union(polygons)

    def _plot_polygons(self, polygons):

        fig = plt.figure()
        plt.gca().invert_yaxis()
        ax = fig.add_subplot(111)
        #multi = shapely.geometry.MultiPolygon([[p, []] for p in polygons])
        multi = shapely.geometry.MultiPolygon(polygons)

        for p in multi:
            # plot coordinates system
            x, y = p.exterior.xy
            ax.plot(x, y, 'o', color='#999999', zorder=1)

            #patch = PolygonPatch(p, facecolor='#6699cc', edgecolor='#6699cc', alpha=0.5, zorder=2)
            patch = PolygonPatch(p, edgecolor='#6699cc', alpha=0.5, zorder=2)
            ax.add_patch(patch)
        #polygons = [for ts in tilespecs]

        return fig, ax

    def _find_center(self, all_tiles_pts):
        # receives a list of a (tile) points list
        # find the minimial and maximal x and y values, and return the center of the mfov
        min_x = min([np.min(tile_pts[:, 0]) for tile_pts in all_tiles_pts])
        min_y = min([np.min(tile_pts[:, 1]) for tile_pts in all_tiles_pts])
        max_x = max([np.max(tile_pts[:, 0]) for tile_pts in all_tiles_pts])
        max_y = max([np.max(tile_pts[:, 1]) for tile_pts in all_tiles_pts])

        return np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0]).astype(np.int)

    def visualize_tilespecs(self, tilespecs, title=None):

        # get the per-mfov transformation
        mfovs = set([ts["mfov"] for ts in tilespecs])
        mfovs = sorted(list(mfovs))

        # Get the tilespecs boundaries for each mfov
        mfovs_tiles_orig = defaultdict(list)

        for tile_ts in tilespecs:
            tile_mfov = tile_ts["mfov"]

            orig_pts = np.array([
                    [tile_ts["bbox"][0], tile_ts["bbox"][2]],
                    [tile_ts["bbox"][1], tile_ts["bbox"][2]],
                    [tile_ts["bbox"][1], tile_ts["bbox"][3]],
                    [tile_ts["bbox"][0], tile_ts["bbox"][3]]
                ], dtype=np.float64)
            mfovs_tiles_orig[tile_mfov].append(orig_pts)

        for cur_mfov in mfovs:
            for idx in range(len(mfovs_tiles_orig[cur_mfov])):
                mfovs_tiles_orig[cur_mfov][idx] *= self._scale

        # Create the polygons for each of the mfovs
        polygons_orig = [self._get_pts_unified_polygon(mfovs_tiles_orig[cur_mfov]) for cur_mfov in mfovs]

        # Create the figures for both the original and the projected outlines
        fig_orig, ax_orig = self._plot_polygons(polygons_orig)

        # Add mfov indices to the center of each relevant mfov
        # first, find normalized centers
        mfov_centers_orig = {cur_mfov: self._find_center(mfovs_tiles_orig[cur_mfov]) for cur_mfov in mfovs}

        for cur_mfov in mfovs:
            ax_orig.text(mfov_centers_orig[cur_mfov][0], mfov_centers_orig[cur_mfov][1], '{}'.format(cur_mfov), color='red')
        
        if title is not None:
            ax_orig.set_title(title)

        return fig_orig
        
        


    def visualize_ts_file(self, ts_fname):
        """
        Given a tilespec file name, opens, and outlines the mfovs boundaries as polygons,
        """
        with open(ts_fname, 'rt') as ts_f:
            tilespecs = json.load(ts_f)

        title = os.path.basename(ts_fname)

        return self.visualize_tilespecs(tilespecs, title)


if __name__ == '__main__':
    in_files = sys.argv[1:]

    visualizer = TilespecVisualizer()
    for in_file in in_files:
        fig_orig = visualizer.visualize_ts_file(in_file)
        plt.show()

