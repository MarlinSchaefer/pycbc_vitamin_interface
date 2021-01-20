from BnsLib.utils import progress_tracker
from BnsLib.utils.formatting import field_array_to_dict
from BnsLib.data.generate_train import WFParamGenerator, WaveformGetter
from BnsLib.data import whiten
from BnsLib.types.utils import NamedPSDCache
from BnsLib.types.argparseActions import TranslationAction, str2bool
from BnsLib.utils.vitaminb import vitaminb_params_to_pycbc_params, params_files_from_config
from pycbc.types import MultiDetOptionAction, TimeSeries
from pycbc.filter import sigma
import h5py
import os
import sys
import argparse
import numpy as np
import json
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default=0, type=int,
                        help='The index of the first template to generate.')
    parser.add_argument('--number-samples', default=1000, type=int,
                        help='The number of samples to generate.')
    parser.add_argument('--workers',
                        help='The number of processes to spwan. (default: number processes = number available cores)')
    parser.add_argument('--name', required=True,
                        help='The base-name for the output file.')
    parser.add_argument('--params-config-file', required=True,
                        help='The configuration file from which the templates are generated.')
    parser.add_argument('--network-config-file', required=True,
                        help='The config-file specifying the network to use for training.')
    parser.add_argument('--file-name', default='{name}-{start}-{stop}-{samples}.hdf', type=str,
                        help='A formatting string from which the file names are constructed. Available keys are "name", "start", "stop", "samples".')
    parser.add_argument('--detectors', nargs='+', default=['H1', 'L1', 'V1'],
                        help='The detectors for which to generate the waveforms. (Detector names as understood by PyCBC)')
    parser.add_argument('--verbose', default=False, type=str2bool,
                        help='Whether or not to print a progress bar. (default: false)')
    parser.add_argument('--psd-names', action=MultiDetOptionAction,
                        help='The names (as understood by PyCBC) to use for whitening and noise generation. (Add as det_name1:psd_name1 det_name2:psd_name2)')
    parser.add_argument('--time-shift-name', default='tc',
                        help='The name of the variable that gives the length of the time-shift to apply to every template')
    parser.add_argument('--final-duration', default=1.0, type=float,
                        help='The duration of the final template in seconds.')
    parser.add_argument('--translation', nargs='+', action=TranslationAction, default={},
                        help='Translate the parameter names. (Input as par1:par1new par2:par2new ...)')
    parser.add_argument('--training-data', type=str2bool, default=True,
                        help='If true it uses the directory for the training data to store the results. Otherwise the testing data directory will be used.')
    parser.add_argument('--output-vitaminb-params-files', type=str2bool, default=False,
                        help='Whether or not to generate the params, bounds and fixed-vals files that are required by VItaminB.')
    parser.add_argument('--seed', help='The seed to use to draw waveform-parameters.')
    
    args = parser.parse_args()
    
    stop = args.start + args.number_samples
    
    file_name = args.file_name.format(name=args.name, start=args.start,
                                      stop=stop, samples=args.number_samples)
    
    ######################
    #Convert config files#
    ######################
    #Create .ini-file for PyCBC
    config_file = 'tmp_parameter_config.ini'
    vitaminb_params_to_pycbc_params(args.params_config_file, config_file)
    
    #Convert .ini-files to vitaminb understandable .json files.
    params_file = 'tmp_params.json'
    bounds_file = 'tmp_bounds.json'
    fixed_vals_file = 'tmp_fixed_vals.json'
    tmp_params, tmp_bounds, tmp_fixed = params_files_from_config(args.params_config_file,
                                                                 args.network_config_file,
                                                                 translation=args.translation)
    if args.output_vitaminb_params_files:
        with open(params_file, 'w') as fp:
            json.dump(tmp_params, fp, indent=4)
        with open(bounds_file, 'w') as fp:
            json.dump(tmp_bounds, fp, indent=4)
        with open(fixed_vals_file, 'w') as fp:
            json.dump(tmp_fixed, fp, indent=4)
    
    ####################
    #Generate waveforms#
    ####################
    if args.seed is None:
        seed = int(time.time())
    else:
        seed = int(args.seed)
    param_gen = WFParamGenerator(config_file, seed=seed)
    
    params = param_gen.draw_multiple(args.number_samples)
    params = field_array_to_dict(params)
    try:
        params['tc'] = params.pop(args.time_shift_name)
    except:
        pass
    
    getter = WaveformGetter(variable_params=params,
                            static_params=param_gen.static,
                            detectors=args.detectors)
    
    wavs = getter.generate_mp(workers=args.workers,
                              verbose=args.verbose)
    try:
        params[args.time_shift_name] = params.pop('tc')
    except:
        pass
    
    #############
    #Whiten data#
    #############
    
    #TODO: Make this use multiprocessing
    
    psd_names = {key: val for (key, val) in args.psd_names.items()}
    for key in wavs.keys():
        if key not in psd_names:
            if key == 'H1' or key == 'L1':
                psd_names[key] = 'aLIGOZeroDetHighPower'
            elif key == 'V1':
                #psd_names[key] = 'AdVDesignSensitivityP1200087'
                psd_names[key] = 'Virgo'
            elif key == 'K1':
                psd_names[key] = 'KAGRADesignSensitivityT1600593'
    
    flow = getter.static_params['f_lower']
    if args.verbose:
        bar = progress_tracker(len(wavs)*len(getter), name='Whitening samples')
    white_wavs = {}
    psd_cache = NamedPSDCache(list(set(psd_names.values())))
    for key in wavs.keys():
        white_wavs[key] = []
        psd_name = psd_names.get(key)
        for wav in wavs[key]:
            tmp = TimeSeries(np.zeros(int(8 * wav.sample_rate + len(wav))),
                             delta_t=wav.delta_t,
                             epoch=wav.start_time-4)
            tmp.data[int(4*wav.sample_rate):int(-4*wav.sample_rate)] = wav.data[:]
            psd = psd_cache.get_from_timeseries(tmp, max(flow-2, 0),
                                                psd_name=psd_names[key])
            white_wavs[key].append(whiten(tmp,
                                          low_freq_cutoff=flow,
                                          max_filter_duration=4.,
                                          psd=psd))
            if args.verbose:
                bar.iterate()
    
    #############################
    #Crop data to correct length#
    #############################
    time_shifted = {}
    final_len = int(args.final_duration / getter.static_params['delta_t'])
    if args.verbose:
        bar = progress_tracker(len(white_wavs) * len(getter),
                               name='Applying time shifts')
    for key in white_wavs.keys():
        time_shifted[key] = []
        for i, wav in enumerate(white_wavs[key]):
            t0_idx = int((args.final_duration / 2 - float(wav.start_time)) * wav.sample_rate)
            #Pad too short waveforms
            if t0_idx < 0:
                wav.prepend_zeros(-t0_idx)
                t0_idx = 0
            time_shifted[key].append(wav[t0_idx:t0_idx+final_len])
            if args.verbose:
                bar.iterate()
    
    ################
    #Store the data#
    ################
    #Set the output file location
    if args.training_data:
        file_loc = os.path.join(tmp_params['train_set_dir'],
                                file_name)
    else:
        file_loc = os.path.join(tmp_params['test_set_dir'],
                                file_name)
    
    #Ensure that a translation is given for every parameter
    translation = args.translation
    keys = list(getter.variable_params)
    for key in keys:
        if key not in translation:
            translation[key] = key
    
    #Format the data to store it properly
    store_array = np.zeros((len(time_shifted), len(getter), final_len))
    for i, key in enumerate(time_shifted.keys()):
        for j, dat in enumerate(time_shifted[key]):
            store_array[i][j][:] = np.array(dat)[:]
    store_array = store_array.transpose(1, 0, 2)
    
    #print(f"Store array shape: {store_array.shape}")
    
    x_data = np.vstack([np.array(params[key]) for key in keys])
    x_data = x_data.transpose()
    #x_data = x_data.reshape((len(getter), 1, len(keys)))
    
    #Calculate the optimal SNR of all signals
    if args.verbose:
        bar = progress_tracker(len(time_shifted) * len(getter),
                               name='Calculating SNRs')
    snrs = []
    for key in time_shifted.keys():
        tmp = []
        for wav in time_shifted[key]:
            tmp.append(sigma(wav, low_frequency_cutoff=flow))
        snrs.append(tmp)
    snrs = np.array(snrs)
    snrs = snrs.transpose()
    
    #Set some auxiliary data
    y_normscale = tmp_params['y_normscale']
    tmpkeys = [translation[key] for key in keys]
    
    #Store the data
    if args.training_data:
        with h5py.File(file_loc, 'w') as fp:
            fp.create_dataset('x_data', data=x_data)
            fp.create_dataset('y_data_noisefree', data=store_array*np.sqrt(getter.static_params['delta_t']))
#            fp.create_dataset('y_data_noisefree', data=store_array)
            fp.create_dataset('y_data_noisy', data=store_array+np.random.normal(size=store_array.shape))
#            fp.create_dataset('y_data_noisy', data=store_array+np.random.normal(scale=np.sqrt(1. / getter.static_params['delta_t']), size=store_array.shape))
            fp.create_dataset('rand_pars', data=np.array(tmpkeys, dtype='S'))
            fp.create_dataset('snrs', data=snrs)
            fp.create_dataset('y_normscale', data=y_normscale)
            for key, val in tmp_bounds.items():
                fp.create_dataset(key, data=val)
    else:
        for i in range(args.number_samples):
            
            file_loc = os.path.join(tmp_params['test_set_dir'],
                                    args.file_name.format(name=args.name,
                                                          start=args.start+i,
                                                          stop=args.start+i+1,
                                                          samples=args.number_samples)
                                    )
            with h5py.File(file_loc, 'w') as fp:
                fp.create_dataset('x_data', data=np.expand_dims(x_data[i], axis=0))
                fp.create_dataset('y_data_noisefree', data=store_array[i]*np.sqrt(getter.static_params['delta_t']))
#                fp.create_dataset('y_data_noisefree', data=store_array[i])
                fp.create_dataset('y_data_noisy', data=store_array[i]+np.random.normal(size=store_array[i].shape))
#                fp.create_dataset('y_data_noisy', data=store_array[i]+np.random.normal(scale=np.sqrt(1. / getter.static_params['delta_t']), size=store_array[i].shape))
                fp.create_dataset('rand_pars', data=np.array(tmpkeys, dtype='S'))
                fp.create_dataset('snrs', data=snrs[i])
                fp.create_dataset('y_normscale', data=y_normscale)
                for key, val in tmp_bounds.items():
                    fp.create_dataset(key, data=val)
    
    if args.output_vitaminb_params_files:
        return params_file, bounds_file, fixed_vals_file
    else:
        return None

if __name__ == "__main__":
    main()
