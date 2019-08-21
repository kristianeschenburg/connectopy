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

[amp, exp] = plotting.csv2matrix(args.subject, args.hemisphere, dir_out)
