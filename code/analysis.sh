source activate ucla-la5

###############################################################################
#                           ncluster = 4
###############################################################################

# Golden Subjects
###############################################################################
# Perform ROI extraction and calculate dynamic measures for the golden subjects
tmux new-session -d -s cluster4_gs 'python main_analysis.py -n 1 -r -a -c --analysis-type task --data-analysis-type graph_analysis --window-type sliding --network-type full_network --ica_aroma-type nonaggr --glm_denoise  --nclusters 4 --rand-ind 20 --group-analysis-type ttest'

# Group Analysis (roi extraction, analysis and group analysis)
###############################################################################
tmux new-session -d -s cluster4 'python main_analysis.py -n 20 -a -r -g --analysis-type task --data-analysis-type graph_analysis --window-type sliding --network-type full_network --ica_aroma-type nonaggr --glm_denoise  --nclusters 4 --rand-ind 20 --group-analysis-type ttest'

sleep 5s
###############################################################################
#                           ncluster = 5
###############################################################################

# Golden Subjects
###############################################################################
# Perform ROI extraction and calculate dynamic measures for the golden subjects
tmux new-session -d -s cluster5_gs 'python main_analysis.py -n 1 -r -a -c --analysis-type task --data-analysis-type graph_analysis --window-type sliding --network-type full_network --ica_aroma-type nonaggr --glm_denoise  --nclusters 5 --rand-ind 20 --group-analysis-type ttest'

# Group Analysis (roi extraction, analysis and group analysis)
###############################################################################
tmux new-session -d -s cluster5 'python main_analysis.py -n 20 -a -r -g --analysis-type task --data-analysis-type graph_analysis --window-type sliding --network-type full_network --ica_aroma-type nonaggr --glm_denoise  --nclusters 5 --rand-ind 20 --group-analysis-type ttest'

sleep 5s
###############################################################################
#                           ncluster = 7
###############################################################################

# Golden Subjects
###############################################################################
# Perform ROI extraction and calculate dynamic measures for the golden subjects
tmux new-session -d -s cluster5_gs 'python main_analysis.py -n 1 -r -a -c --analysis-type task --data-analysis-type graph_analysis --window-type sliding --network-type full_network --ica_aroma-type nonaggr --glm_denoise  --nclusters 10 --rand-ind 20 --group-analysis-type ttest'

# Group Analysis (roi extraction, analysis and group analysis)
###############################################################################
tmux new-session -d -s cluster7 'python main_analysis.py -n 20 -a -r -g --analysis-type task --data-analysis-type graph_analysis --window-type sliding --network-type full_network --ica_aroma-type nonaggr --glm_denoise  --nclusters 7 --rand-ind 20 --group-analysis-type ttest'

sleep 5s
###############################################################################
#                           ncluster = 10
###############################################################################

# Golden Subjects
###############################################################################
# Perform ROI extraction and calculate dynamic measures for the golden subjects
# python main_analysis.py -n 1 -r -a -c --analysis-type task --data-analysis-type graph_analysis --window-type sliding --network-type full_network --ica_aroma-type nonaggr --glm_denoise  --nclusters 10 --rand-ind 20 --group-analysis-type ttest

# Group Analysis
###############################################################################
tmux new-session -d -s cluster10 'python main_analysis.py -n 20 -a -g --analysis-type task --data-analysis-type graph_analysis --window-type sliding --network-type full_network --ica_aroma-type nonaggr --glm_denoise  --nclusters 10 --rand-ind 20 --group-analysis-type ttest'
