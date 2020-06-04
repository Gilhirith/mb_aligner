import argparse
import sys
from mb_aligner.alignment.aligner import StackAligner
from mb_aligner.dal.section import Section
from rh_logger.api import logger
import logging
import rh_logger
import os
import ujson

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Runs the alignment algorithm on the given file that includes a list of sections (each section on a separate line)")
    parser.add_argument("--sections_list_file", metavar="sections_list_file",
                        help="a file containing the list of ordered sections (each section on a separate line)")
    parser.add_argument("-o", "--output_dir", metavar="output_dir",
                        help="The output directory where the aligned tilespecs will be stored")
    parser.add_argument("-c", "--conf_fname", metavar="conf_name",
                        help="The configuration file (yaml). (default: None)",
                        default=None)
    
    return parser.parse_args(args)

def run_aligner(args):

    # Make a list of all the relevant sections
    with open(args.sections_list_file, 'rt') as in_f:
        secs_ts_fnames = in_f.readlines()
    secs_ts_fnames = [fname.strip() for fname in secs_ts_fnames]
    
    # Make sure the tilespecs exist
    all_files_exist = True
    for sec_ts_fname in secs_ts_fnames:
        if not os.path.exists(sec_ts_fname):
            print("Cannot find tilespec file: {}".format(sec_ts_fname))
            all_files_exist = False

    if not all_files_exist:
        print("One or more tilespecs could not be found, exiting!")
        return

    out_folder = './output_aligned_ECS_test9_cropped'
    conf_fname = '../../conf/conf_example.yaml'


    conf = StackAligner.load_conf_from_file(args.conf_fname)
    logger.report_event("Loading sections", log_level=logging.INFO)
    sections = []
    # TODO - Should be done in a parallel fashion
    for ts_fname in secs_ts_fnames:
        with open(ts_fname, 'rt') as in_f:
            tilespec = ujson.load(in_f)
        
        wafer_num = int(os.path.basename(ts_fname).split('_')[0].split('W')[1])
        sec_num = int(os.path.basename(ts_fname).split('.')[0].split('_')[1].split('Sec')[1])
        sections.append(Section.create_from_tilespec(tilespec, wafer_section=(wafer_num, sec_num)))

    logger.report_event("Initializing aligner", log_level=logging.INFO)
    aligner = StackAligner(conf)
    logger.report_event("Aligning sections", log_level=logging.INFO)
    aligner.align_sections(sections) # will align and update the section tiles' transformations


    del aligner

    logger.end_process('main ending', rh_logger.ExitCode(0))


if __name__ == '__main__':
    args = parse_args()

    logger.start_process('main', 'align_3d_tilespecs_list.py', [args])
    run_aligner(args)
    logger.end_process('main ending', rh_logger.ExitCode(0))


