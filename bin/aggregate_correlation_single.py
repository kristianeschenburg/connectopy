import numpy as np
from niio import loaded, write
from fragmenter import RegionExtractor as re
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subject', help='Subjject ID', required=True, type=str)
parser.add_argument('-conn', '--dir_conn', help='Connectopy directory.', required=True, type=str)
parser.add_argument('-hemi', '--hemisphere', help='Hemisphere.', required=True, type=str)
parser.add_argument('-l', '--label', help='Label file.', required=True, type=str)
parser.add_argument('-t', '--target', help='Target region.', required=True, type=str)

args = parser.parse_args()

hemi_map = {'L': 'LEFT',
                'R': 'RIGHT'}

cortex_map = {'L': 'CortexLeft',
              'R': 'CortexRight'}

R = re.Extractor(args.label)
regions = R.map_regions()

subject = args.subject
conn_dir = args.dir_conn
hemisphere = args.hemisphere

for j, target_region in enumerate(list([args.target])):
        if target_region != 'corpuscallosum':
            
            print('Source Map ID: {:}'.format(j))
            target_inds = regions[target_region]

            z = np.zeros((32492, 50))
            for source_region in regions.keys():
                if source_region not in ['corpuscallosum', target_region]:
                    source_inds = regions[source_region]

                    knn_file = '{:}NeighborFunctional/{:}/{:}.{:}.{:}.2.{:}.mean_knn.func.gii'.format(conn_dir,
                                                                                                      subject,
                                                                                                      subject,
                                                                                                      hemisphere,
                                                                                                      source_region,
                                                                                                      target_region)
                    print(knn_file)
                    knn = loaded.load(knn_file)
                    z += knn

            z[target_inds, :] = np.nan
            out_dir = '{:}NeighborFunctional/{:}'.format(conn_dir, subject)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
                
            write.save(z, '{:}/{:}.{:}.knn_mean.2.{:}.func.gii'.format(out_dir,
                                                                       subject,
                                                                       hemisphere,
                                                                       target_region), cortex_map[hemisphere])