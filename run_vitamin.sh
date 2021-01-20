#!/usr/env/bin/bash

python /work/marlin.schaefer/projects/collab_glasgow/vitamin_b/vitamin_b/run_vitamin.py \
--train True \
--params_file /work/marlin.schaefer/projects/collab_glasgow/data_gen/tmp_params.json \
--bounds_file /work/marlin.schaefer/projects/collab_glasgow/data_gen/tmp_bounds.json \
--fixed_vals_file /work/marlin.schaefer/projects/collab_glasgow/data_gen/tmp_fixed_vals.json
