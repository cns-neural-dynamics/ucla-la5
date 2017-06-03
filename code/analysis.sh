#!/usr/bin/env bash
source activate ucla-la5

allclusters=(3 4 5 6 7 8 9 10)
tasktype=rest

for i in "${allclusters[@]}"
do
    tmux new-session -d -s cluster"$i"_gs 'python main_analysis.py -n 1 -r -a -c --analysis-type '"$tasktype"' --data-analysis-type graph_analysis --window-type sliding --network-type full_network --ica_aroma-type nonaggr --glm_denoise --nclusters '"$i"' --rand-ind 20 --group-analysis-type ttest'
done

# wait until the last of the job for golden subjects has been submitted
#$? corresponds to the output from the previous commit
tmux has-session -t cluster"${allclusters[-1]}"_gs
while [ $? -ne 0 ];
do
    sleep 10s
done

echo "submitting analysis"
for i in "${allclusters[@]}"
do
    tmux new-session -d -s cluster"$i" 'python main_analysis.py -n 20 -a -g --analysis-type '"$tasktype"' --data-analysis-type graph_analysis --window-type sliding --network-type full_network --ica_aroma-type nonaggr --glm_denoise --nclusters '"$i"' --rand-ind 20 --group-analysis-type ttest'
done
