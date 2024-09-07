#!/bin/bash
CMDSTAN_HOME="$HOME/.cmdstan/cmdstan-2.35.0"
$CMDSTAN_HOME/cmdstan-cuda/model/model_cu_bnb \
    random seed=1234 \
    data file=$CMDSTAN_HOME/cmdstan-cuda/output/data.json \
    output file=$CMDSTAN_HOME/cmdstan-cuda/output/model_cu_bnb-$(date +"%Y%m%d%H%M%S").csv \
    refresh=20 \
    method=sample \
    num_samples=100 \
    num_warmup=100 \
    algorithm=hmc \
    adapt engaged=1
