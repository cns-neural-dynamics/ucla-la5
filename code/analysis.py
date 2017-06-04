import subprocess
import libtmux
import time


all_clusters = [3, 4, 5, 6, 7, 8, 9, 1, 4, 5, 6, 7, 8, 9, 10]
tasktype='rest'
server = libtmux.Server()

for ncluster in all_clusters:
    cmd1 = "tmux new-session -d -s cluster%d_gs 'python main_analysis.py -n 1 -r -a -c --analysis-type %s --data-analysis-type graph_analysis --window-type sliding --network-type full_network --ica_aroma-type nonaggr --glm_denoise --nclusters %d --rand-ind 20 --group-analysis-type ttest'" \
          %(ncluster, tasktype, ncluster)
    subprocess.check_call(cmd1, shell=True)
    print cmd1

# check if golden subjects is still runing, in case yes, wait
last_gs = ''.join(['cluster', str(all_clusters[-1]), '_gs'])
while server.find_where({'session_name':last_gs}):
    time.sleep(15)

for ncluster in all_clusters:
    cmd2 = "tmux new-session -d -s cluster%d 'python main_analysis.py -n 20 -r -a -g --analysis-type %s --data-analysis-type graph_analysis --window-type sliding --network-type full_network --ica_aroma-type nonaggr --glm_denoise --nclusters %d --rand-ind 20 --group-analysis-type ttest'" \
           %(ncluster, tasktype, ncluster)
    subprocess.check_output(cmd2, shell=True)
    print cmd2


