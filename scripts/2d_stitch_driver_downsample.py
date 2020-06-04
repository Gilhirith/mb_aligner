import argparse
import sys
import os
import glob
import ujson
from mb_aligner.stitching.stitcher_downsampled import Stitcher
from mb_aligner.dal.section import Section
from rh_logger.api import logger
import logging
import rh_logger
import common
import fs

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Runs the stitching algorithm on the given tilespecs input directory")
    parser.add_argument("--ts_dir", metavar="ts_dir", required="True",
                        help="the tilespecs to be stitched directory")
    parser.add_argument("-o", "--output_dir", metavar="output_dir",
                        help="The output directory that will hold the stitched tilespecs (default: ./output_dir)",
                        default="./output_dir")
    parser.add_argument("-c", "--conf_fname", metavar="conf_name",
                        help="The configuration file (yaml). (default: None)",
                        default=None)
    return parser.parse_args(args)

def get_ts_files(cur_fs, ts_folder):
    all_ts_fnames = []
    all_ts_fnames_glob = cur_fs.glob("*.json")
    for ts_fname_glob in all_ts_fnames_glob:
        ts_fname = ts_fname_glob.path
        all_ts_fnames.append(cur_fs.geturl(ts_fname))
    return sorted(all_ts_fnames)



def run_stitcher(args):

    #common.fs_create_dir(args.output_dir)

    conf = None
    if args.conf_fname is not None:
        conf = Stitcher.load_conf_from_file(args.conf_fname)
    stitcher = Stitcher(conf)

    # read the inpput tilespecs
    in_fs = fs.open_fs(args.ts_dir)
    in_ts_fnames = get_ts_files(in_fs, args.ts_dir) #sorted(glob.glob(os.path.join(args.ts_dir, "*.json")))

    out_fs = fs.open_fs(args.output_dir)
    logger.report_event("Stitching {} sections".format(len(in_ts_fnames)), log_level=logging.INFO)
    for in_ts_fname in in_ts_fnames:
        logger.report_event("Stitching {}".format(in_ts_fname), log_level=logging.DEBUG)
        out_ts_fname = args.output_dir + "/" + fs.path.basename(in_ts_fname)
        if out_fs.exists(out_ts_fname):
            continue

        print("Stitching {}".format(in_ts_fname))
        with in_fs.open(fs.path.basename(in_ts_fname), 'rt') as in_f:
            in_ts = ujson.load(in_f)

        wafer_num = int(fs.path.basename(in_ts_fname).split('_')[0].split('W')[1])
        sec_num = int(fs.path.basename(in_ts_fname).split('.')[0].split('_')[1].split('Sec')[1])
        section = Section.create_from_tilespec(in_ts, wafer_section=(wafer_num, sec_num))
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

    logger.start_process('main', '2d_stitcher_driver.py', [args])
    run_stitcher(args)
    logger.end_process('main ending', rh_logger.ExitCode(0))


