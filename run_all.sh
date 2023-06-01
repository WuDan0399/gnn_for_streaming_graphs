#!/bin/bash

# Generate by Chatgpt

# Set your script name
# read -p "Enter the script name: " selected_script

# Set your lists here
selected_script="GCN_inf_dynamic.py"

# Remove the last three characters (like .py) from selected_script
selected_script_name=${selected_script%???}

datasets=("cora" "reddit" "yelp")  # "amazon")
aggregators=("min" "mean")
distributions=("random" "burst")

# Loop through all combinations
for selected_dataset in "${datasets[@]}"
do
  for aggregator in "${aggregators[@]}"
  do
    python GCN_neighbor_loader.py --dataset "${selected_dataset}" --aggr "${aggregator}"
    wait
    for dist in "${distributions[@]}"
    do
      # Run the command and redirect output to a log file
      python "${selected_script}" --dataset "${selected_dataset}" --aggr "${aggregator}" --distribution "${dist}" > "${selected_script_name}_${selected_dataset}_${aggregator}_${dist}.log" #2>&1
      # Wait for the command to finish before starting the next one
      wait
    done
  done
done