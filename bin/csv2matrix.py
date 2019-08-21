import argparse
from connectopy import plotting
import scipy.io as sio

from connectopy import utilities as uti

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subject', help='Subjject ID', required=True, type=str)
parser.add_argument('-hemi', '--hemisphere', help='Hemisphere.', required=True, type=str)
parser.add_argument('-md', '--modeldir', help='Model directory.', required=True, type=str)

args = parser.parse_args()

subject_id = args.subject
hemisphere = args.hemisphere
modeldir = args.modeldir

[amp, exp] = plotting.csv2matrix(subject_id, hemisphere, modeldir)
amp = {'amplitude': amp}
exp = {'exponent': exp}

dir_out = '%s%s/' % (modeldir, subject_id)

out_amp = '%s%s.%s.Amplitude.mat' % (dir_out,
                                    subject_id, 
                                    hemisphere)
out_exp = '%s%s.%s.Exponent.mat' % (dir_out,
                                    subject_id, 
                                    hemisphere)

sio.savemat(file_name=out_amp, mdict=amp)
sio.savemat(file_name=out_exp, mdict=exp)
