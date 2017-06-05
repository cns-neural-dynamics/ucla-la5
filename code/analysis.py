import subprocess
import libtmux
import time


clusters = ['3', '4', '5', '6', '7', '8', '9', '10']
tasktypes = ['rest', 'task']
server = libtmux.Server()

# Note: because of the way find_where works, if there are no tmux windows the code will fail. Thefore, we create a dummy
# tmux that will be present throught the code and killed at the end
dummy_tmux_cmd = "tmux new-session -d -s dummy"
subprocess.call(dummy_tmux_cmd, shell=True)

for tasktype in tasktypes:
    for cluster in clusters:
        cmd1 = "tmux new-session -d -s cluster%s_gs1 'python main_analysis.py -n 1 -r -a -c --analysis-type %s --data-analysis-type graph_analysis --window-type sliding --network-type full_network --ica_aroma-type nonaggr --glm_denoise --nclusters %s --rand-ind 20 --group-analysis-type ttest'" \
              %(cluster, tasktype, cluster)
        print cmd1
        subprocess.call(cmd1, shell=True)
        curr_gs = ''.join(['cluster', cluster, '_gs1'])
        while server.find_where({'session_name':curr_gs}):
            time.sleep(15)

for tasktype in tasktypes:
    for cluster in clusters:
        cmd2 = "tmux new-session -d -s cluster%s 'python main_analysis.py -n 20 -r -a -g --analysis-type %s --data-analysis-type graph_analysis --window-type sliding --network-type full_network --ica_aroma-type nonaggr --glm_denoise --nclusters %s --rand-ind 20 --group-analysis-type ttest'" \
               %(cluster, tasktype, cluster)
        subprocess.call(cmd2, shell=True)
        print cmd2

# kill dummy tmux
kill_dummy = "tmux kill-session -t dummy"
subprocess.check_output(kill_dummy, shell=True)
