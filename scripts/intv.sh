#!/bin/bash

# Model-specific configurations ###############################################
model_name="gpt-4o"
model_short_name="gpt-4o"
max_model_len=128000

judge_model_name="gpt-4o"
judge_model_short_name="gpt-4o"

# Blackbox-specific configurations ###########################################
blackbox_name="programs"
# blackbox_name="languages"
# blackbox_name="ces"

# Experiment-specific configurations #########################################
# experiment_stages="obs"
# experiment_stages="obs+intv"
experiment_stages="obs+intv+reason"

num_datapoints=5
nobs=10
num_interventions=(5)

batch_size=1
# run_modes="datagen+eval"
# run_modes="datagen"
# run_modes="eval"
run_modes="datagen+eval+judge"
# run_modes="judge"
use_azure="none"

seeds=0

difficulty="none"

for nintv in ${num_interventions[@]}; do
    python -m main \
        --blackbox_name $blackbox_name \
        --model_name $model_name \
        --model_short_name $model_short_name \
        --max_model_len $max_model_len \
        --judge_model_name $judge_model_name \
        --experiment_stages $experiment_stages \
        --run_modes $run_modes \
        --nblackbox_instance $num_datapoints \
        --nobs $nobs \
        --nintv $nintv \
        --seed $seed \
        --difficulty $difficulty \
        --use_azure $use_azure
done


