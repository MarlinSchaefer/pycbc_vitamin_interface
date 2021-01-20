from pycbc.frame import write_frame
from pycbc.types import TimeSeries
import numpy as np
import h5py
import argparse
import warnings
import logging

"""
This file contains code to split training data used for VitaminB into
frame files that are readable by PyCBC.

The file needs to contain three datasets: 'y_data_noisy', 'x_data',
'rand_pars'.
"""

def main():
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=logging.INFO, datefmt='%d-%m-%Y %H:%M:%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True, type=str,
                        help="The file from which samples are read.")
    parser.add_argument('--num-samples', type=int,
                        help="How many samples should be produced. Default: Number of samples in file")
    parser.add_argument('--detectors', type=str, nargs='+', default=['H1', 'L1', 'V1'],
                        help='The order of detectors. Default: H1 L1 V1')
    parser.add_argument('--sample-rate', type=int, default=256,
                        help='The sample rate of the data. Default: 256')
    parser.add_argument('--output', type=str, default='{detector}-frame-{sample}.hdf',
                        help='The location under which the frame files are stored. Possible macros include `detector` and `sample`. Default: `{detector}-frame-{sample}.hdf`.')
    parser.add_argument('--channel-name', type=str, default='{detector}:CHANNEL',
                        help='The channel names given to the data. Possible macros include `detector` Default: `{detector}:CHANNEL`.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress updates.')
    parser.add_argument('--parameter-file', type=str,
                        help='Write the parameters of the training samples to a file. To not write files do not set this option. Possible macros are: `sample`.')
    
    args = parser.parse_args()
    
    translation = {'mass_1': 'mass1',
                   'mass_2': 'mass2',
                   'luminosity_distance': 'distance',
                   'theta_jn': 'inclination',
                   'phase': 'coa_phase',
                   'psi': 'polarization',
                   'dec': 'dec',
                   'ra': 'ra',
                   'geocent_time': 'tc'
                   }
    
    with h5py.File(args.input_file, 'r') as fp:
        if args.num_samples is None:
            args.num_samples = len(fp['y_data_noisy'])
        if args.num_samples > len(fp['y_data_noisy']):
            warnings.warn('More samples requested than available. Reducing to available samples and continuing. (requested: {}, available: {})'.format(args.num_samples, len(fp['y_data_noisy'])), RuntimeWarning)
            args.num_samples = len(fp['y_data_noisy'])
        
        if len(args.detectors) > fp['y_data_noisy'].shape[1]:
            raise RuntimeError('Requesing more detectors than available.')
        
        for samp in range(args.num_samples):
            for i, det in enumerate(args.detectors):
                data = fp['y_data_noisy'][samp,i,:]
                tmp = TimeSeries(data, delta_t=1./args.sample_rate)
                write_frame(args.output.format(detector=det, sample=samp),
                            args.channel_name.format(detector=det),
                            tmp)
                if args.verbose:
                    logging.info('Wrote sample {} to {}.'.format(samp, args.output.format(detector=det, sample=samp)))
            if args.parameter_file is not None:
                line = ''
                rand_pars = fp['rand_pars'][()].astype(str)
                for idx, par in enumerate(rand_pars):
                    line += translation[str(par)]
                    line += ':'
                    line += str(fp['x_data'][samp,idx])
                    if idx < len(rand_pars) - 1:
                        line += ' '
                with open(args.parameter_file.format(sample=samp), 'w') as txt:
                    txt.write(line)
                if args.verbose:
                    logging.info('Wrote parameters of sample {} to {}.'.format(samp, args.parameter_file.format(detector=det, sample=samp)))
    return

if __name__ == "__main__":
    main()
