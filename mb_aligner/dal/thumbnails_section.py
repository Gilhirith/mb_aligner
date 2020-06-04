from collections import defaultdict
import os
import json
import csv
import numpy as np
import cv2
from mb_aligner.dal.section import Section
from mb_aligner.dal.thumbnails_tile import ThumbnailsTile
import multiprocessing as mp

class ThumbnailsSection(object):
    """
    Represents a single section (at least one mfov) of thumbnails in the system.
    Each tile is actually the "stitched" mfov (using the thumbnail coordinates)
    """

    def __init__(self, mfovs_dict, **kwargs):
        self._mfovs_dict = mfovs_dict

        # Initialize default values
        self._layer = None

        # initialize values using kwargs
        if len(kwargs) > 0:
            if "layer" in kwargs:
                self._layer = kwargs["layer"]
        #elif self._mfovs_dict is not None and len(self._mfovs_dict) > 0:
        #    self._layer = self._mfovs_dict.values()[0].layer
            

    @classmethod
    def _parse_coordinates_file(cls, input_file):
        # Read the relevant mfovs tiles locations
        images_dict = {}
        images = []
        x = []
        y = []
        with open(input_file, 'r') as csvfile:
            data_reader = csv.reader(csvfile, delimiter='\t')
            for row in data_reader:
                img_fname = row[0].replace('\\', '/')
                # Make sure that the mfov appears in the relevant mfovs
                if not (img_fname.split('/')[0]).isdigit():
                    # skip the row
                    continue
                img_sec_mfov_beam = '_'.join(img_fname.split('/')[-1].split('_')[1:4])
                # Make sure that no duplicates appear
                if img_sec_mfov_beam not in images_dict.keys():
                    images.append(img_fname)
                    images_dict[img_sec_mfov_beam] = len(images) - 1
                    cur_x = float(row[1])
                    cur_y = float(row[2])
                    x.append(cur_x)
                    y.append(cur_y)
                else:
                    # Either the image is duplicated, or a newer version was taken,
                    # so make sure that the newer version is used
                    prev_img_idx = images_dict[img_sec_mfov_beam]
                    prev_img = images[prev_img_idx]
                    prev_img_date = prev_img.split('/')[-1].split('_')[-1]
                    curr_img_date = img_fname.split('/')[-1].split('_')[-1]
                    if curr_img_date > prev_img_date:
                        images[prev_img_idx] = img_fname
                        images_dict[img_sec_mfov_beam] = img_fname
                        cur_x = float(row[1])
                        cur_y = float(row[2])
                        x[prev_img_idx] = cur_x
                        y[prev_img_idx] = cur_y

        return images, np.array(x), np.array(y)


    @classmethod
    def create_from_full_thumbnail_coordinates(cls, full_thumbnail_coordinates_fname, layer, thumb_tile_size=None, processes_num=1):
        """
        Creates a section from a given full_thumbnail_coordinates filename
        """
        images, x_locs, y_locs = ThumbnailsSection._parse_coordinates_file(full_thumbnail_coordinates_fname)
        assert(len(images) > 0)
        section_folder = os.path.dirname(full_thumbnail_coordinates_fname)

        # Update thumb_tile_size if needed
        if thumb_tile_size is None:
            # read the first image
            img_fname = os.path.join(section_folder, images[0])
            img = cv2.imread(img_fname, 0)
            thumb_tile_size = img.shape

        # normalize the locations of all the tiles (reset to (0, 0))
        x_locs -= np.min(x_locs)
        y_locs -= np.min(y_locs)


        # Create all the tiles
        per_mfov_tiles_fnames = defaultdict(list)
        per_mfov_tiles_locs = defaultdict(list)
        for tile_fname, tile_x, tile_y, in zip(images, x_locs, y_locs):
            tile_fname = os.path.join(section_folder, tile_fname)
            # fetch mfov_idx, and tile_idx
            split_data = os.path.basename(tile_fname).split('_')
            mfov_idx = int(split_data[2])
            #tile_idx = int(split_data[3])
            #print('adding mfov_idx %d, tile_idx %d' % (mfov_idx, tile_idx))
            per_mfov_tiles_fnames[mfov_idx].append(tile_fname)
            per_mfov_tiles_locs[mfov_idx].append(np.array([tile_x, tile_y]))
            

        all_mfovs = {}
        if processes_num == 1:
            for mfov_idx in per_mfov_tiles_fnames.keys():
                print("Creating mfov thumbnails stitched tile: {}".format(mfov_idx))
                mfov_thumbs_tile = ThumbnailsTile.create_from_input(per_mfov_tiles_fnames[mfov_idx], thumb_tile_size, per_mfov_tiles_locs[mfov_idx], layer, mfov_idx)
                all_mfovs[mfov_idx] = mfov_thumbs_tile
        else:
            pool = mp.Pool(processes=processes_num)

            pool_results = []
            for mfov_idx in per_mfov_tiles_fnames.keys():
                print("Creating mfov thumbnails stitched tile: {}".format(mfov_idx))
                res = pool.apply_async(ThumbnailsTile.create_from_input, (per_mfov_tiles_fnames[mfov_idx], thumb_tile_size, per_mfov_tiles_locs[mfov_idx], layer, mfov_idx))
                pool_results.append(res)

            for res in pool_results:
                mfov_thumbs_tile = res.get()
                all_mfovs[mfov_thumbs_tile.mfov_index] = mfov_thumbs_tile
 
            pool.close()
            pool.join()

        return ThumbnailsSection(all_mfovs, kwargs={'layer':layer})



    @property
    def layer(self):
        """
        Returns the section layer number
        """
        return self._layer

#     @property
#     def tilespec(self):
#         """
#         Returns a tilespec representation of the mfov
#         """
#         ret = []
#         # Order the mfovs by the mfov index
#         sorted_mfov_idxs = sorted(self._mfovs_dict.keys())
#         for mfov_idx in sorted_mfov_idxs:
#             ret.extend(self._mfovs_dict[mfov_idx].tilespec)
#         return ret
# 
#     def save_as_json(self, out_fname):
#         """
#         Saves the section as a tilespec
#         """
#         with open(out_fname, 'w') as out_f:
#             json.dump(self.tilespec, out_f, sort_keys=True, indent=4)

    def mfovs(self):
        '''
        A generator that iterates over all the mfovs (tiles in our case) in the section
        '''
        mfov_keys = sorted(self._mfovs_dict.keys())
        for mfov_idx in mfov_keys:
            yield self._mfovs_dict[mfov_idx]

    def tiles(self):
        '''
        A generator that iterates over all the tiles (mfovs in our case) in the section
        '''
        return self.mfovs()

        

if __name__ == '__main__':
    section = ThumbnailsSection.create_from_full_thumbnail_coordinates('/n/lichtmanfs2/Alex/EM/ROI2_w03/W03_H01_ROI2_20180113_00-26-33/021_S56R1/full_thumbnail_coordinates.txt', 56)

    for mfov in section.mfovs():
        print("Mfov idx: %d" % mfov.mfov_index)

