import argparse
from pycbc.inference.io import PosteriorFile
from pycbc import strain, psd, DYN_RANGE_FAC
from pycbc.inject import InjFilterRejector
from pycbc.types import copy_opts_for_single_ifo
from pycbc.types import MultiDetOptionAppendAction, MultiDetOptionAction
from pycbc.inference.io.base_hdf import format_attr
from vitamin_b.models.CVAE_model import run as get_posterior
import numpy as np
import json
import os
import h5py
from BnsLib.types.argparseActions import TranslationAction, str2bool, TypedDictAction
from BnsLib.data.transform import whiten

"""This code takes a VitaminB network and options to specify
input-strain. It produces a posterior file understandable by PyCBC.

For the input-strain, one may specify an injection file (as provided
by pycbc_create_injections) or frame files. To convert numpy-arrays into
frame files use the provided code training_to_test_samples.py.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network-dir', type=str,
                        help='The directory in which the network was stored. (Defaults to last checkpoint, which is specified in the params-file)')
    parser.add_argument('--params-file', type=str, required=True,
                        help='The path to the parameter-file (as defined by VitaminB) that was used to train the network.')
    parser.add_argument('--bounds-file', type=str, required=True,
                        help='A file that specifies the bounds. Can be omitted if a injection file with bounds on the parameters is used. May be either the file defined by VitaminB or a file that contains param_min and param_max for all inferred parameters. [For now required in json-format]')
    parser.add_argument('--posterior-file', type=str, default='./posterior.hdf',
                        help='The path to the output file.')
    parser.add_argument('--y-normscale', type=float,
                        help='A factor by which the data is divided. Should be the same as for training. Do not set this option if you want to use the value from the params-file.')
    parser.add_argument('--translation', nargs='+', action=TranslationAction,
                        help='Translate the parameter names. (Input as par1:par1new par2:par2new ...) [The old names should be the ones specified by VitaminB, the new names should be the ones used by PyCBC, leave blank to use the standard translation.]')
    parser.add_argument('--whiten', type=str2bool, default=True,
                        help='Whether or not to whiten the data.')
    parser.add_argument('--whiten-by-psd', type=str2bool, default=False,
                        help='If set to True the PSD-model given in the PSD-options-group will be used. Otherwise the strain will be whitened by an internal estimate.')
    parser.add_argument('--max-filter-duration', type=float, default=4.,
                        nargs="+", action=MultiDetOptionAction,
                        help='The maximum duration of the PSD in the time domain. Used to crop corrupted data.')
    parser.add_argument('--psd-segment-length', type=float, default=4.,
                        nargs="+", action=MultiDetOptionAction,
                        help='The duration of the parts used to estimate the PSD, if it is being estimated.')
    parser.add_argument('--true-params', action=TypedDictAction,
                        help='Pass the injected parameters. These will be stored along the posterior samples. Give as `--true-params param1:type1:val1 param2:type2:val2` to ensure the correct types are used. For automatic type inference use `--true-params param1:val1 param2:val2`.')
    #TODO: Make this a cutoff dependent on the ifo
    parser.add_argument('--low-frequency-cutoff', type=float, default=20.,
                        help='The low-frequency cutoff used when whitening the data.')
    
    ############################
    #Add standard PyCBC options#
    ############################
    strain.insert_strain_option_group_multi_ifo(parser)
    
    ####################
    #Actual computation#
    ####################
    #Load stuff
    args = parser.parse_args()
    
    with open(args.params_file, 'r') as fp:
        params = json.load(fp)
    
    with open(args.bounds_file, 'r') as fp:
        bounds = json.load(fp)
    
    #Create waveform
    strain_dict = strain.from_cli_multi_ifos(args,
                                             params['det'],
                                             #inj_filter_rejector,
                                             dyn_range_fac=DYN_RANGE_FAC)
    
    #Whiten the data
    if args.whiten:
        if args.whiten_by_psd:
            for ifo in params['det']:
                opts = copy_opts_for_single_ifo(args, ifo)
                strain_dict[ifo] = whiten(strain_dict[ifo], psd=opts.fake_strain,
                                        low_freq_cutoff=args.low_frequency_cutoff,
                                        max_filter_duration=args.max_filter_duration[ifo])
        else:
            for ifo in params['det']:
                corrupt_samples = int(strain_dict[ifo].sample_rate * max_filter_duration)
                strain_dict[ifo] = strain_dict[ifo].whiten(segment_duration=args.psd_segment_length[ifo],
                                                        max_filter_duration=args.max_filter_duration[ifo],
                                                        low_frequency_cutoff=args.low_frequency_cutoff)
    
    #Scale to mean=0, std=1
    scaled_strain = {}
    for ifo in params['det']:
        tmp = np.array(strain_dict[ifo])
        tmp = tmp - np.mean(tmp)
        tmp = tmp / np.std(tmp)
        scaled_strain[ifo] = tmp
    
    #Crop data to correct length
    for ifo in params['det']:
        data = scaled_strain[ifo]
        diff = len(data) - params['ndata']
        if diff < 0:
            prepend = np.random.normal(size=-diff)
            scaled_strain[ifo] = np.concatenate([prepend, scaled_strain[ifo]])
        elif diff > 0:
            scaled_strain[ifo] = scaled_strain[ifo][-params['ndata']:]
        cropped_time = diff * strain_dict[ifo].sample_rate
    
    #Shape data
    y_data = np.zeros((1, params['ndata'], len(params['det'])))
    for i, ifo in enumerate(params['det']):
        y_data[0,:,i] = scaled_strain[ifo]
    
    #Get the arguments for running the VitaminB-sampler    
    if args.network_dir is None:
        network_dir = './inverse_model_dir_' + params['run_label']
    else:
        network_dir = args.network_dir
    if not os.path.isdir(network_dir):
        raise ValueError('Could not find a network at {}; directory does not exist.'.format(network_dir))
    network_path = os.path.join(network_dir, 'inverse_model.ckpt')
    num_infer = len(params['inf_pars'])
    if args.y_normscale is None:
        y_sc = params['y_normscale']
    else:
        y_sc = args.y_normscale
    
    #Run the sampler
    posterior, dt, _ = get_posterior(params, y_data, num_infer, y_sc, network_path)
    
    #Translate VitaminB parameter names -> PyCBC parameter names
    base_translation = {'geocent_time': 'tc',
                        'mass_1': 'mass1',
                        'mass_2': 'mass2',
                        'luminosity_distance': 'distance',
                        'theta_jn': 'inclination',
                        'phase': 'coa_phase',
                        'psi': 'pol'}
    translation = args.translation if args.translation is not None else base_translation
    translated = []
    for name in params['inf_pars']:
        if str(name) in translation.keys():
            translated.append(translation[name])
        else:
            translated.append(name)
    
    #Get bounds on the inferred parameters and re-scale Vitamin output
    posterior_dict = {}
    for i, (old_name, new_name) in enumerate(zip(params['inf_pars'], translated)):
        min_val = None
        if new_name + '_min' in bounds:
            min_val = bounds[new_name + '_min']
        elif old_name + '_min' in bounds:
            if min_val is not None:
                if new_name in translation.keys():
                    min_val = bounds[new_name + '_min']
                elif old_name in translation.values():
                    pass
                elif min_val == bounds[old_name + '_min']:
                    pass
                else:
                    msg = 'Two different minimum values were given in the bounds file.'
                    raise RuntimeError(msg)
            else:
                min_val = bounds[old_name + '_min']
        if min_val is None:
            msg = 'Could not find a minimum value for {}.'.format(old_name)
            raise RuntimeError(msg)
        
        max_val = None
        if new_name + '_max' in bounds:
            max_val = bounds[new_name + '_max']
        elif old_name + '_max' in bounds:
            if max_val is not None:
                if new_name in translation.keys():
                    max_val = bounds[new_name + '_max']
                elif old_name in translation.values():
                    pass
                elif max_val == bounds[old_name + '_max']:
                    pass
                else:
                    msg = 'Two different maximum values were given in the bounds file.'
                    raise RuntimeError(msg)
            else:
                max_val = bounds[old_name + '_max']
        if max_val is None:
            msg = 'Could not find a maximum value for {}.'.format(old_name)
            raise RuntimeError(msg)
        tmp = posterior.T[i]
        tmp = tmp * (max_val - min_val)
        tmp = tmp + min_val
        if new_name == 'tc':
            tmp = args.gps_start_time[params['det'][0]] + args.max_filter_duration[params['det'][0]] + cropped_time + tmp
        posterior_dict[new_name] = tmp
    
    #Write samples to file
    post_file = PosteriorFile(args.posterior_file, mode='w')
    post_file.write_samples(posterior_dict)
    if args.injection_file is not None:
        with h5py.File(args.injection_file[params['det'][0]], 'r') as fp:
            post_file.attrs['variable_params'] = np.array(fp.keys(), dtype='S')
            for key, val in dict(fp.attrs).items():
                if key == 'static_args':
                    post_file.attrs['static_params'] = val
                else:
                    post_file.attrs[key] = val
    if args.true_params is not None:
        inj_gr = post_file.create_group(post_file.injections_group)
        for key, val in args.true_params.items():
            if isinstance(val, list):
                inj_gr.create_dataset(key, data=np.array(val))
            else:
                inj_gr.create_dataset(key, data=np.array([val]))
    elif args.injection_file is not None:
        post_file.write_injections(args.injection_file[params['det'][0]])
    post_file.close()
    return

if __name__ == "__main__":
    main()
