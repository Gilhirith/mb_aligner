from mb_aligner.stitching.stitcher import Stitcher
from mb_aligner.dal.thumbnails_section import ThumbnailsSection
from rh_logger.api import logger
import logging
import rh_logger
import cv2
import numpy as np


def render_section(out_fname, section, ds_rate):
    # go through the section's tiles (mfovs), erode, downsample, and normalize each mfov

    # find the output shape
    tiles_bboxes = np.array([tile.bbox for tile in section.tiles()])
    min_x = np.min(tiles_bboxes[:, 0])
    max_x = np.max(tiles_bboxes[:, 1])
    min_y = np.min(tiles_bboxes[:, 2])
    max_y = np.max(tiles_bboxes[:, 3])

    out_shape = tuple(np.ceil(np.array([max_y - min_y, max_x - min_x]) * ds_rate).astype(np.int))
    print("out_shape: {}".format(out_shape))
    out_image = np.zeros(out_shape, dtype=np.uint8)

    for tile in section.tiles():
        print("Placing tile/mfov: {}".format(tile.mfov_index))
        # load mfov image and mask
        mfov_image = 255 - tile.image
        mfov_mask = tile.mask

        # erode the image and mask
        mfov_image = cv2.erode(mfov_image, np.ones((3,3), np.uint8))
        mfov_mask = cv2.erode(mfov_mask, np.ones((3,3), np.uint8))
        #mfov_image = cv2.medianBlur(mfov_image, 3)
        #mfov_mask = cv2.medianBlur(mfov_mask, 3)

        # downsample the images
        mfov_image = cv2.resize(mfov_image, None, fx=ds_rate, fy=ds_rate, interpolation=cv2.INTER_CUBIC)
        mfov_mask = cv2.resize(mfov_mask, None, fx=ds_rate, fy=ds_rate, interpolation=cv2.INTER_NEAREST)

        # normalize and clamp values
        mfov_image[mfov_mask > 0] = ((mfov_image[mfov_mask > 0] - np.mean(mfov_image[mfov_mask > 0]))/np.std(mfov_image[mfov_mask > 0])*40 + 127).astype(np.int)
        mfov_image[mfov_image < 0] = 0
        mfov_image[mfov_image > 255] = 255
        mfov_image = mfov_image.astype(np.uint8)

        #mfov_image = 255 - mfov_image
        #cv2.imwrite("mfov_{}.jpg".format(tile.mfov_index), mfov_image)
        #cv2.imwrite("mfov_mask_{}.jpg".format(tile.mfov_index), mfov_mask)
        # place the image using the mask
        assert(len(tile.transforms) == 1)
        starting_point = np.floor(np.array([tile.bbox[0], tile.bbox[2]]) * ds_rate).astype(np.int)
        print("tile bbox: {}".format(tile.bbox))
        print("tile starting point: {}".format(starting_point))
        out_image[
                    starting_point[1]:starting_point[1] + mfov_image.shape[0],
                    starting_point[0]:starting_point[0] + mfov_image.shape[1]
                 ][mfov_mask > 0] = mfov_image[mfov_mask > 0]

    print("Saving outpt image to: {}".format(out_fname))
    cv2.imwrite(out_fname, out_image)

if __name__ == '__main__':
    #section_dir = '/n/lichtmanfs2/100um_Sept2017/EM/w01h03/100umsept2017_20170912_17-52-13/181_S181R1/adi_full_thumbnail_coordinates.txt'
    #section_dir = '/n/lichtmanfs2/100um_Sept2017/EM/w01h03/100umsept2017_20170912_17-52-13/181_S181R1/full_thumbnail_coordinates.txt'
    #section_num = 181
    #out_jpg_fname = './output_stitched_thumbs_100um_Sept2017_w01h03_100umsept2017_20170912_17-52-13_181_S181R1.jpg'
    #section_dir = '/n/lichtmanfs2/Alex/EM/ROI2_w04/W04_H04_ROI2_20180109_16-51-34/003_S3R1/full_thumbnail_coordinates.txt'
    #section_num = 3
    #out_jpg_fname = './output_stitched_thumbs_Alex_ROI2_w04_W04_H04_ROI2_20180109_16-51-34_003_S3R1.jpg'
    #section_dir = '/n/lichtmanfs2/Alex/EM/ROI2_w04/W04_H04_ROI2_20180109_16-51-34/004_S4R1/full_thumbnail_coordinates.txt'
    #section_num = 4
    #out_jpg_fname = './output_stitched_thumbs_Alex_ROI2_w04_W04_H04_ROI2_20180109_16-51-34_004_S4R1.jpg'
    section_dir = '/n/lichtmanfs2/Alex/EM/ROI2_w08/W08_H01_ROI2_20171227_00-07-19/010_S10R1/full_thumbnail_coordinates.txt'
    section_num = 10
    out_jpg_fname = './output_stitched_thumbs_Alex_ROI2_w08_W08_H01_ROI2_20171227_00-07-19_010_S10R1.jpg'
    #section_dir = '/n/lichtmanfs2/Alex/EM/ROI2_w08/W08_H01_ROI2_20171227_00-07-19/011_S11R1/full_thumbnail_coordinates.txt'
    #section_num = 11
    #out_jpg_fname = './output_stitched_thumbs_Alex_ROI2_w08_W08_H01_ROI2_20171227_00-07-19_011_S11R1.jpg'
    conf_fname = '../../conf/conf_thumbs_example.yaml'
    processes_num = 8
#
#     section = Section.create_from_full_image_coordinates(section_dir, section_num)
#     conf = Stitcher.load_conf_from_file(conf_fname)
#     stitcher = Stitcher(conf, processes_num)
#     stitcher.stitch_section(section) # will stitch and update the section
#
#     # Save the transforms to file
#     import json
#     print('Writing output to: {}'.format(out_fname))
#     section.save_as_json(out_fname)
# #     img_fnames, imgs = StackAligner.read_imgs(imgs_dir)
# #     for img_fname, img, transform in zip(img_fnames, imgs, transforms):
# #         # assumption: the output image shape will be the same as the input image
# #         out_fname = os.path.join(out_path, os.path.basename(img_fname))
# #         img_transformed = cv2.warpAffine(img, transform[:2,:], (img.shape[1], img.shape[0]), flags=cv2.INTER_AREA)
# #         cv2.imwrite(out_fname, img_transformed)

# Testing
#    test_detector('/n/home10/adisuis/Harvard/git/rh_aligner/tests/ECS_test9_cropped/images/010_S10R1', conf_fname, 8, 500)

    logger.start_process('main', 'stitcher.py', [section_dir, conf_fname])
    section = ThumbnailsSection.create_from_full_thumbnail_coordinates(section_dir, section_num, processes_num=processes_num)
    conf = Stitcher.load_conf_from_file(conf_fname)
    stitcher = Stitcher(conf)
    stitcher.stitch_section(section) # will stitch and update the section tiles' transformations

    # render the stitched section
    ds_rate = 1.0/8
    render_section(out_jpg_fname, section, ds_rate)

    # TODO - output the section
    logger.end_process('main ending', rh_logger.ExitCode(0))

