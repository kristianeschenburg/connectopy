import numpy as np
from fragmenter import RegionExtractor as re
import argparse

from connectopy import utilities as uti

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subject', help='Subjject ID', required=True, type=str)
parser.add_argument('-conn', '--dir_conn', help='Connectopy directory.', required=True, type=str)
parser.add_argument('-hemi', '--hemisphere', help='Hemisphere.', required=True, type=str)
parser.add_argument('-l', '--label', help='Label file.', required=True, type=str)

args = parser.parse_args()

R = re.Extractor(args.label)
region_map = R.map_regions()

uti.s2t_correlations_aggregate(subject_id=args.subject, 
                               region_map=region_map, 
                               hemisphere=args.hemisphere,
                               connectopy_dir=args.dir_conn)