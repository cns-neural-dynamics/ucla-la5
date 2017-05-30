source activate ucla-la5

###############################################################################
# Golden Subjects
###############################################################################
# Perform ROI extraction and calculate dynamic measures for the golden subjects
python main_analysis.py -n 1 -r -a -c --analysis-type task --data-analysis-type graph_analysis --window-type sliding --network-type full_network  --ica_aroma-type nonaggr --glm_denoise  --nclusters 10 --rand-ind 20 --group-analysis-type ttest
