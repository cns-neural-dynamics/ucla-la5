#!/usr/bin/env bash
source activate ucla-la5

allclusters=(3 5)
tasktype=rest

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
