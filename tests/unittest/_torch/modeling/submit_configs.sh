#!/bin/bash

# a config consists of TP, CP, EP
configs="8,1,2 16,1,4 32,1,8 1,8,2 1,16,4 1,32,8 2,4,2 2,8,4 2,16,8"

for config in $configs; do
    IFS=","
    set -- $config
    tp=$1
    cp=$2
    ep=$3
    echo "TP $tp, CP $cp, EP $ep, MoE"
    bash test_helix_deepseek_sbatch.sh $tp $cp $ep
    echo "TP $tp, CP $cp, EP $ep, Dense"
    bash test_helix_deepseek_sbatch.sh $tp $cp $ep 1
done
