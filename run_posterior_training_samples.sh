#! /usr/bin/env bash

base_dir="/work/marlin.schaefer/projects/collab_glasgow/data_gen"
#data_dir="${base_dir}/training_data_frames"
data_dir="${base_dir}/frames_public_test"

for i in {0..3..1}
do

python pycbc_vitaminb_posterior.py \
--frame-files H1:${data_dir}/H1-frame-${i}.hdf L1:${data_dir}/L1-frame-${i}.hdf V1:${data_dir}/V1-frame-${i}.hdf \
--channel-name H1:H1:CHANNEL L1:L1:CHANNEL V1:V1:CHANNEL \
--gps-start-time 0 \
--gps-end-time 1 \
--sample-rate 256 \
--pad-data H1:0 L1:0 V1:0 \
--params-file /work/marlin.schaefer/projects/collab_glasgow/vitamin_b/vitamin_b/params_files/params.json \
--bounds-file /work/marlin.schaefer/projects/collab_glasgow/vitamin_b/vitamin_b/params_files/bounds.json \
--posterior-file ${data_dir}/posteriors/posterior_${i}.hdf \
--whiten False \
--true-params $(< ${data_dir}/params-${i}.txt) \
--network-dir "${base_dir}/public_model"
#--network-dir "${base_dir}/inverse_model_dir_demo_3det_9par_256Hz"
#--params-file tmp_params.json \
#--bounds-file tmp_bounds.json \

done
