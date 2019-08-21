import numpy as np
from fragmenter import RegionExtractor as re
import argparse

from connectopy import utilities as uti
from connectopy import plotting

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subject', help='Subjject ID', required=True, type=str)
parser.add_argument('-conn', '--conndir', help='Connectopy directory.', required=True, type=str)
parser.add_argument('-hemi', '--hemisphere', help='Hemisphere.', required=True, type=str)
parser.add_argument('-l', '--label', help='Label file.', required=True, type=str)

args = parser.parse_args()

subject = args.subject
dir_conn = args.conndir
hemisphere = args.hemisphere

R = re.Extractor(args.label)
regions = R.map_regions()

uti.s2t_correlations_aggregate(subject_id=subject, region_map=regions, 
                               hemisphere=hemisphere, connectopy_dir=dir_conn)
