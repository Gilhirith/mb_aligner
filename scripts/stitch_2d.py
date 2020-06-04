import argparse
import sys
from mb_aligner.stitching.stitcher import Stitcher
from mb_aligner.dal.section import Section
from rh_logger.api import logger
import logging
import rh_logger

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Runs the stitching algorithm on the given input section")
    parser.add_argument("--images_coords_file", metavar="images_coords_file",
                        help="the section's coordinates file for all images (typically called full_image_coordinates.txt)")
    parser.add_argument("--output_json", metavar="output_json",
                        help="The output file name to write the data to")
    parser.add_argument("-c", "--conf_fname", metavar="conf_name",
                        help="The configuration file (yaml). (default: None)",
                        default=None)
    parser.add_argument("-s", "--sec_num", metavar="sec_num", type=int,
                        help="The section number. (default: 1)",
                        default=1)
    
    return parser.parse_args(args)

def run_stitcher(args):
    # Make up a section number

    section = Section.create_from_full_image_coordinates(args.images_coords_file, args.sec_num)
    conf = Stitcher.load_conf_from_file(args.conf_fname)
    stitcher = Stitcher(conf)
    stitcher.stitch_section(section) # will stitch and update the section

    # Save the transforms to file
    print('Writing output to: {}'.format(args.output_json))
    section.save_as_json(args.output_json)


if __name__ == '__main__':
    args = parse_args()

    logger.start_process('main', 'stitch_2d.py', [args])
    run_stitcher(args)
    logger.end_process('main ending', rh_logger.ExitCode(0))


