import argparse
import sys
from mb_aligner.stitching.stitcher import Stitcher
from mb_aligner.dal.section import Section
from rh_logger.api import logger
import logging
import rh_logger
import os
import re

def sec_dir_to_wafer_section(sec_dir):
    wafer_folder = os.path.split(sec_dir)[-3]
    section_folder = os.path.split(sec_dir)[-1]

    m = re.match('.*_[W|w]([0-9])+.*', wafer_folder)
    if m is None:
        raise Exception("Couldn't find wafer number from section directory {} (wafer dir is: {})".format(sec_dir, wafer_folder))
    wafer_num = int(m.group(1))

    m = re.match('.*_S([0-9])+.*', section_folder)
    if m is None:
        raise Exception("Couldn't find section number from section directory {} (section dir is: {})".format(sec_dir, section_folder))
    sec_num = int(m.group(1))

    return wafer_num, sec_num


def get_layer_num(sec_num, initial_layer_num):
    layer_num = sec_num + initial_layer_num - 1
    return layer_num

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Runs the stitching algorithm on the given file that includes a list of sections (each section on a separate line)")
    parser.add_argument("--sections_list_file", metavar="sections_list_file",
                        help="a file containing the list of raw multibeam image directories, each of a different section")
    parser.add_argument("-o", "--output_dir", metavar="output_dir",
                        help="The output directory where the stitched tilespecs will be stored")
    parser.add_argument("-c", "--conf_fname", metavar="conf_name",
                        help="The configuration file (yaml). (default: None)",
                        default=None)
    parser.add_argument("-i", "--initial_layer_num", metavar="initial_layer_num", type=int,
                        help="The layer# of the first section in the list. (default: 1)",
                        default=1)
    
    return parser.parse_args(args)


def run_stitcher(args):

    # Make a list of all the relevant sections
    with open(args.sections_list_file, 'rt') as in_f:
        secs_dirs = in_f.readlines()
    secs_dirs = [dirname.strip() for dirnamename in secs_dirs]
    
    # Make sure the folders exist
    all_dirs_exist = True
    for sec_dir in secs_dirs:
        if not os.path.exists(sec_dir):
            print("Cannot find folder: {}".format(sec_dir))
            all_dirs_exist = False

    if not all_dirs_exist:
        print("One or more directories could not be found, exiting!")
        return


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    conf = None
    if args.conf_fname is not None:
        conf = Stitcher.load_conf_from_file(args.conf_fname)
    stitcher = Stitcher(conf)

    for sec_dir in sec_dirs:
        # extract wafer and section# from directory name
        wafer_num, sec_num = sec_dir_to_wafer_section(sec_dir)
        out_ts_fname = os.path.join(args.output_dir, 'W{}_Sec{}_montaged.json'.format(str(wafer_num).zfill(2), str(sec_num).zfill(3)))
        if os.path.exists(out_ts_fname):
            continue

        layer_num = get_layer_num(sec_num, args.initial_layer_num)

        print("Stitching {}".format(sec_dir))
        section = Section.create_from_full_image_coordinates(os.path.join(sec_dir, 'full_image_coordinates.txt'), layer_num)
        stitcher.stitch_section(section)

        # Save the tilespec
        section.save_as_json(out_ts_fname)
#         out_tilespec = section.tilespec
#         import json
#         with open(out_ts_fname, 'wt') as out_f:
#             json.dump(out_tilespec, out_f, sort_keys=True, indent=4)

    del stitcher


if __name__ == '__main__':
    args = parse_args()

    logger.start_process('main', 'stitch_2d_raw_folders_list.py', [args])
    run_stitcher(args)
    logger.end_process('main ending', rh_logger.ExitCode(0))


