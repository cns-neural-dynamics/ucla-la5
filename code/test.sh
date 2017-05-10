
###############################################################################
# Golden Subjects
###############################################################################
# Perform ROI extraction and calculate dynamic measures for the golden subjects
python main_analysis.py  -c -r -a --analysis-type task --data-analysis-type graph_analysis --window-type sliding --network-type full_network --nclusters 10 --rand-ind 20

###############################################################################
# Extract ROI
###############################################################################
python main_analysis.py -n 1 -r --analysis-type task --network-type full_network
python main_analysis.py -n 1 -r --analysis-type task --network-type between_network
python main_analysis.py -n 1 -r --analysis-type task --network-type within_network

###############################################################################
# Data analysis
###############################################################################
# BOLD
python main_analysis.py -n 1 -a --analysis-type task --window-type sliding     --data-analysis-type BOLD           --network-type full_network    --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --analysis-type task --window-type non-sliding --data-analysis-type BOLD           --network-type full_network    --nclusters 15 --rand-ind 20

# Synchrony
python main_analysis.py -n 1 -a --analysis-type task --window-type sliding     --data-analysis-type synchrony      --network-type full_network    --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --analysis-type task --window-type non-sliding --data-analysis-type synchrony      --network-type full_network    --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --analysis-type task --window-type sliding     --data-analysis-type synchrony      --network-type between_network --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --analysis-type task --window-type non-sliding --data-analysis-type synchrony      --network-type between_network --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --analysis-type task --window-type sliding     --data-analysis-type synchrony      --network-type within_network  --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --analysis-type task --window-type non-sliding --data-analysis-type synchrony      --network-type within_network  --nclusters 15 --rand-ind 20

# Graph analysis
python main_analysis.py -n 1 -a --analysis-type task --window-type sliding     --data-analysis-type graph_analysis --network-type full_network    --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --analysis-type task --window-type non-sliding --data-analysis-type graph_analysis --network-type full_network    --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --analysis-type task --window-type sliding     --data-analysis-type graph_analysis --network-type between_network --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --analysis-type task --window-type non-sliding --data-analysis-type graph_analysis --network-type between_network --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --analysis-type task --window-type sliding     --data-analysis-type graph_analysis --network-type within_network  --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --analysis-type task --window-type non-sliding --data-analysis-type graph_analysis --network-type within_network  --nclusters 15 --rand-ind 20
