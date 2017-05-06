# Extract ROI
python main_analysis.py -n 1 -r --network-type full_network
python main_analysis.py -n 1 -r --network-type between_network
python main_analysis.py -n 1 -r --network-type within_network

# Data analysis - BOLD
python main_analysis.py -n 1 -a --window-type sliding     --data-analysis-type BOLD           --network-type full_network    --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --window-type non-sliding --data-analysis-type BOLD           --network-type full_network    --nclusters 15 --rand-ind 20

# Data analysis - synchrony
python main_analysis.py -n 1 -a --window-type sliding     --data-analysis-type synchrony      --network-type full_network    --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --window-type non-sliding --data-analysis-type synchrony      --network-type full_network    --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --window-type sliding     --data-analysis-type synchrony      --network-type between_network --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --window-type non-sliding --data-analysis-type synchrony      --network-type between_network --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --window-type sliding     --data-analysis-type synchrony      --network-type within_network  --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --window-type non-sliding --data-analysis-type synchrony      --network-type within_network  --nclusters 15 --rand-ind 20

# Data analysis - graph analysis
python main_analysis.py -n 1 -a --window-type sliding     --data-analysis-type graph_analysis --network-type full_network    --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --window-type non-sliding --data-analysis-type graph_analysis --network-type full_network    --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --window-type sliding     --data-analysis-type graph_analysis --network-type between_network --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --window-type non-sliding --data-analysis-type graph_analysis --network-type between_network --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --window-type sliding     --data-analysis-type graph_analysis --network-type within_network  --nclusters 15 --rand-ind 20
python main_analysis.py -n 1 -a --window-type non-sliding --data-analysis-type graph_analysis --network-type within_network  --nclusters 15 --rand-ind 20
