#!/usr/bin/env bash
source activate ucla-la5

allclusters=(3 4 5 6 7 8 9 10)
tasktype=rest

# Graph Analysis
#------------------------------------------------------------------------------
for i in "${allclusters[@]}"; do
    srun -n 1 python main_analysis.py -n 1 -r -a -c \
        --analysis-type "$tasktype" --data-analysis-type graph_analysis \
        --window-type sliding --network-type full_network \
        --ica_aroma-type nonaggr --glm_denoise --nclusters "$i" --rand-ind 20 \
        --group-analysis-type ttest &> /dev/null & pids+=($!)
done

for pid in "${pids[@]}"; do
   wait "$pid"
   echo $pid
done

for i in "${allclusters[@]}"; do
    srun -n 1 python main_analysis.py -n 20 -a -r -g \
        --analysis-type "$tasktype" --data-analysis-type graph_analysis \
        --window-type sliding --network-type full_network \
        --ica_aroma-type nonaggr --glm_denoise --nclusters "$i" --rand-ind 20 \
        --group-analysis-type ttest &> /dev/null & pids+=($!)
done

# Synchrony Analysis
#------------------------------------------------------------------------------
for i in "${allclusters[@]}"; do
    srun -n 1 python main_analysis.py -n 1 -r -a -c \
        --analysis-type "$tasktype" --data-analysis-type synchrony \
        --window-type sliding --network-type full_network \
        --ica_aroma-type nonaggr --glm_denoise --nclusters "$i" --rand-ind 20 \
        --group-analysis-type ttest &> /dev/null & pids+=($!)
done

for pid in "${pids[@]}"; do
   wait "$pid"
   echo $pid
done

for i in "${allclusters[@]}"; do
    srun -n 1 python main_analysis.py -n 20 -a -r -g \
        --analysis-type "$tasktype" --data-analysis-type synchrony \
        --window-type sliding --network-type full_network \
        --ica_aroma-type nonaggr --glm_denoise --nclusters "$i" --rand-ind 20 \
        --group-analysis-type ttest &> /dev/null & pids+=($!)
done

# BOLD
#------------------------------------------------------------------------------
for i in "${allclusters[@]}"; do
    srun -n 1 python main_analysis.py -n 1 -r -a -c \
        --analysis-type "$tasktype" --data-analysis-type BOLD \
        --window-type sliding --network-type full_network \
        --ica_aroma-type nonaggr --glm_denoise --nclusters "$i" --rand-ind 20 \
        --group-analysis-type ttest &> /dev/null & pids+=($!)
done

for pid in "${pids[@]}"; do
   wait "$pid"
   echo $pid
done

for i in "${allclusters[@]}"; do
    srun -n 1 python main_analysis.py -n 20 -a -r -g \
        --analysis-type "$tasktype" --data-analysis-type BOLD \
        --window-type sliding --network-type full_network \
        --ica_aroma-type nonaggr --glm_denoise --nclusters "$i" --rand-ind 20 \
        --group-analysis-type ttest &> /dev/null & pids+=($!)
done
