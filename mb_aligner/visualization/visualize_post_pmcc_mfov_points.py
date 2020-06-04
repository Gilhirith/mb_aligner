from __future__ import print_function
import sys
import cv2
from rh_renderer.tilespec_affine_renderer import TilespecAffineRenderer
import argparse
from rh_renderer import models
from mb_aligner.dal.section import Section
import numpy as np
import time
import os
import tinyr
import pickle
import ujson


def find_relevant_tilespecs(ts, points):
    # create an rtree of the second section, so we can quickly look for the tiles that have the matched points
    sec_rtree = tinyr.RTree(interleaved=False, max_cap=5, min_cap=2)
    for tile_ts_idx, tile_ts in enumerate(ts):
        bbox = tile_ts["bbox"]
        # using the (x_min, x_max, y_min, y_max) notation
        sec_rtree.insert(tile_ts_idx, (bbox[0], bbox[1], bbox[2], bbox[3]))

    # Find the relevant tiles from section 2
    section_tiles_indices = set()
    for pt in points:
        rect_res = sec_rtree.search( (pt[0] - 1, pt[0] + 1, pt[1] - 1, pt[1] + 1) )
        for t_sec_idx in rect_res:
            section_tiles_indices.add(t_sec_idx)
            #print("tile img1: (m: {}, t: {}) - tile img2: (global tile idx: {})".format(tile["mfov"], tile["tile_index"], t_sec2_idx))

    assert(len(section_tiles_indices) > 0)

    # find sec min and max X,Ys
    ts_relevant_bboxes = np.array([ts[ts_tile_idx]["bbox"] for ts_tile_idx in section_tiles_indices])
    ts_relevant_min_x = np.min(ts_relevant_bboxes[:, 0])
    ts_relevant_max_x = np.max(ts_relevant_bboxes[:, 1])
    ts_relevant_min_y = np.min(ts_relevant_bboxes[:, 2])
    ts_relevant_max_y = np.max(ts_relevant_bboxes[:, 3])


    # find the box of tiles of the second section that needs to be renvered
    rect_res = sec_rtree.search( (ts_relevant_min_x, ts_relevant_max_x, ts_relevant_min_y, ts_relevant_max_y) )
    section_relevant_box_tilespecs = []
    for t_sec_idx in rect_res:
        section_relevant_box_tilespecs.append(ts[t_sec_idx])

    return section_relevant_box_tilespecs


def draw_circles(img, start_point, pts, scale):
    # downsample the pts, and subtract the start_point
    pts *= scale
    pts -= np.array(start_point)

    for pt_idx, pt in enumerate(pts):
        circle_color = ((pt_idx + 1) * 231 % 255, (pt_idx + 1) * 17 % 255, (pt_idx + 1) * 129 % 255)
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, circle_color, -1)

def load_tilespecs(ts_fname):
    with open(ts_fname, 'rt') as in_f:
        tilespec = ujson.load(in_f)
    wafer_num = int(os.path.basename(ts_fname).split('_')[0].split('W')[1])
    #wafer_num = 1
    #sec_num = int(os.path.basename(ts_fname).split('_')[-1].split('S')[1].split('R')[0])
    sec_num = int(os.path.basename(ts_fname).split('Sec')[1].split('_')[0])
    section = Section.create_from_tilespec(tilespec, wafer_section=(wafer_num, sec_num))
    return tilespec, section

def visualize_post_pmcc_mfov_points(pmcc_matches_fname, ts1_fname, ts2_fname, mfov, scale, output_fname_prefix):

    # Load tha pmcc matches file
    with open(pmcc_matches_fname, 'rb') as in_f:
        intermed_results = pickle.load(in_f)

    # load the sections and make sure we have the same data
    ts1, sec1 = load_tilespecs(ts1_fname)
    ts2, sec2 = load_tilespecs(ts2_fname)

    if sec1.canonical_section_name_no_layer == intermed_results['metadata']['sec1'] and \
       sec2.canonical_section_name_no_layer == intermed_results['metadata']['sec2']:
        print("loading matches between: {} and {}".format(sec1.canonical_section_name_no_layer, sec2.canonical_section_name_no_layer))
        matches = intermed_results['contents'][0]
    elif sec1.canonical_section_name_no_layer == intermed_results['metadata']['sec2'] and \
       sec2.canonical_section_name_no_layer == intermed_results['metadata']['sec1']:
        print("loading matches between: {} and {}".format(sec2.canonical_section_name_no_layer, sec1.canonical_section_name_no_layer))
        matches = intermed_results['contents'][1]
    else:
        raise Exception("The sections canoncical names didn't match the pmcc matches file contents (sec1: {}, sec2: {}, pmcc_file1: {}, pmcc_file2: {}".format(sec1.canonical_section_name_no_layer, sec2.canonical_section_name_no_layer, intermed_results['metadata']['sec1'], intermed_results['metadata']['sec2']))

    print("Matches shape: {}".format(matches.shape))
    assert(len(matches) > 0 and len(matches[0]) > 0)

    # find relevant points of section1
    #ts1_relevant = find_relevant_tilespecs(ts1, matches[0])
    if mfov is None:
        # need to take the entire section into account
        print("No mfov information in matches file for ts1, taking the relevant tiles from entire section")
        ts1_relevant = find_relevant_tilespecs(ts1, matches[0])
    else:
        ts1_relevant = sec1.get_mfov(mfov).tilespec
        mfov1_min_xy = np.min([[ts["bbox"][0], ts["bbox"][2]] for ts in ts1_relevant], axis=0)
        mfov1_max_xy = np.max([[ts["bbox"][1], ts["bbox"][3]] for ts in ts1_relevant], axis=0)
        # restrict the matches to only the relevant matches
        restricted_matches_mask = (mfov1_min_xy[0] <= matches[0][:, 0]) & (matches[0][:, 0] <= mfov1_max_xy[0])
        restricted_matches_mask &= (mfov1_min_xy[1] <= matches[0][:, 1]) & (matches[0][:, 1] <= mfov1_max_xy[1])
        matches = np.array([matches[0][restricted_matches_mask], matches[1][restricted_matches_mask]])
    ts2_relevant = find_relevant_tilespecs(ts2, matches[1])
    print("Relevant section2 tiles {}: {}".format(len(ts2_relevant), [(ts["mfov"], ts["tile_index"]) for ts in ts2_relevant]))


    # Create the (lazy) renderers for the two sections
    img1_renderer = TilespecAffineRenderer(ts1_relevant)
    img2_renderer = TilespecAffineRenderer(ts2_relevant)

    scale_transformation = np.array([
                                [ scale, 0., 0. ],
                                [ 0., scale, 0. ]
                            ])
    img1_renderer.add_transformation(scale_transformation)
    img2_renderer.add_transformation(scale_transformation)

    # Render the two images
    start_time = time.time()
    img1, start_point1 = img1_renderer.render()
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    print("image 1 rendered in {} seconds".format(time.time() - start_time))
    start_time = time.time()
    img2, start_point2 = img2_renderer.render()
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    print("image 2 rendered in {} seconds".format(time.time() - start_time))

    # draw circles on the matches locations
    print("Drawing circles")
    draw_circles(img1, start_point1, matches[0], scale)
    draw_circles(img2, start_point2, matches[1], scale)


    # save the output
    img1_output_fname = "{}_1.jpg".format(output_fname_prefix)
    cv2.imwrite(img1_output_fname, img1)
    img2_output_fname = "{}_2.jpg".format(output_fname_prefix)
    cv2.imwrite(img2_output_fname, img2)
    



        

def main():
    # print(sys.argv)
    # Command line parser
    parser = argparse.ArgumentParser(description='Given a between slices post pmcc-match json file (of a given mfov), renders the non-transformed original mfov, and the matching area in the second section, and draws circles on the matches.')
    parser.add_argument('pmcc_matches_file', metavar='pmcc_matches_file', type=str,
                        help='an intermediate result file that contains the pmcc matches')
    parser.add_argument('ts1_fname', metavar='ts1_fname', type=str,
                        help='The first section tilespec file name')
    parser.add_argument('ts2_fname', metavar='ts2_fname', type=str,
                        help='The second section tilespec file name')
    parser.add_argument('-m', '--mfov', type=int,
                        help='mfov to show (default: None, show entire section)',
                        default=None)
    parser.add_argument('-s', '--scale', type=float,
                        help='output scale (default: 0.1)',
                        default=0.1)
    parser.add_argument('-o', '--output_file_prefix', type=str,
                        help='a prefix for the output jpg file (default: ./output)',
                        default='./output')

    args = parser.parse_args()
    visualize_post_pmcc_mfov_points(args.pmcc_matches_file, args.ts1_fname, args.ts2_fname, args.mfov, args.scale, args.output_file_prefix)

if __name__ == '__main__':
    main()
