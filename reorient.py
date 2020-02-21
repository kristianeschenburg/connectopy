import argparse
import os

import numpy as np
from scipy.spatial.distance import cdist

from niio import lodaed, write
from fragmenter import RegionExtractor as re

parser = argparse.ArgumentParser(decscription='Switch and reorient connectopic maps, based on ' \
        'their correlation with MDS vectors computed from geodesic distance matrices.')

parser.add_argument('-s', '--subject', help='Subject ID.',
    required=True, type=str)
parser.add_argument('-m', '--mds', help='Path to MDS vectors.',
    required=True, type=str)
parser.add_argument('-l', '--label', help='Label file.',
    required=True, type=str)
parser.add_argument('-c', '--connect', help='Connectopy file.',
    required=True, type=str)
parser.add_argument('-r', '--region', help='Region ID.',
    required=True, type=str)

args = parser.parse_args()

subject = args.subject
mds_file = args.mds
conn_file = args.connect
label_file = args.label
region = args.region

R = re.Extractor(label_file)
index_map = R.map_regions()
indices = index_map[region]
indices.sort()

mds = loaded.load(mds_file)[indices, :]
conn = loaded.load(conn_file)[indices, :]

corr = 1 - cdist(mds, conn, metric='correlation')
