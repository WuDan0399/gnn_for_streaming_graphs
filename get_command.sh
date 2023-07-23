#!/bin/bash

# Set your script name
# selected_script="myGCN.py"
selected_script="GCN_timing_original.py"
# Set your lists here
datasets=("PubMed" "reddit" "yelp")  # "cora" "amazon")
aggregators=("min")
streams=("add" "delete" "mix")

# Range of iterations for data_it
data_its=($(seq 0 9))

# File to store commands
cmd_file="commands.txt"

# Loop through all combinations
for selected_dataset in "${datasets[@]}"
do
  for aggregator in "${aggregators[@]}"
  do
    for stream in "${streams[@]}"
    do
      if [[ "${stream}" == "mix" ]]
      then
        for data_it in "${data_its[@]}"
        do
          # When stream is 'mix', use --perbatch 2
          echo "python ${selected_script} --dataset ${selected_dataset} --aggr ${aggregator} --range affected --perbatch 2 --stream ${stream} --it ${data_it} --save_int" >> $cmd_file
        done
      else
        for data_it in "${data_its[@]}"
        do
          # When stream is not 'mix', use --perbatch 1 and iterate with --it "${data_it}"
          echo "python ${selected_script} --dataset ${selected_dataset} --aggr ${aggregator} --range affected --perbatch 1 --stream ${stream} --it ${data_it} --save_int" >> $cmd_file
        done
      fi
      echo "python ${selected_script} --dataset ${selected_dataset} --aggr ${aggregator} --range affected --perbatch 10 --stream ${stream} --save_int" >> $cmd_file
      echo "python ${selected_script} --dataset ${selected_dataset} --aggr ${aggregator} --range affected --perbatch 100 --stream ${stream} --save_int" >> $cmd_file
    done
  done
done
