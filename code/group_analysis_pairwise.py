from __future__ import division

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import csv
import itertools
from scipy import stats
from math import log, sqrt


def group_analysis_subject_basepath(basepath,
                                    network_type,
                                    window_type,
                                    subject):
    return os.path.join(basepath, network_type, window_type, subject)

def group_analysis_group_basepath(basepath,
                                  network_type,
                                  window_type):
    return os.path.join(basepath, network_type, window_type)

def group_analysis_pairwise(subjects,
                            input_basepath,
                            output_basepath,
                            network_type,
                            window_type,
                            data_analysis_type,
                            group_analysis_type,
                            nclusters,
                            rand_ind,
                            significancy=.05):

    print('--------------------------------------------------------------------')
    print(' Group analysis')
    print('--------------------------------------------------------------------')
    print('')
    print('* PARAMETERS')
    print('network type:        %s' %(network_type))
    print('windowk type:        %s' %(window_type))
    print('data analysis type:  %s' %(data_analysis_type))
    print('group analysis type: %s' %(group_analysis_type))
    print('nclusters:           %d' %(nclusters))
    print('rand_ind:            %d' %(rand_ind))
    print('')

    # Parameters of interest for the different data analysis types.
    if data_analysis_type == 'graph_analysis':
        measures = ['cluster_coefficient', 'degree_centrality', 'weight']
    elif data_analysis_type == 'synchrony':
        measures = ['synchrony']
    elif data_analysis_type == 'BOLD':
        measures = ['BOLD']
    # All 3 analysis are comparing the entropy values betwen groups
    parameters = ['entropy']

    # Generate the output folders.
    group_path = group_analysis_group_basepath(output_basepath,
                                               network_type,
                                               window_type)
    if data_analysis_type == 'graph_analysis':
        group_path = os.path.join(group_path,
                                  'nclusters_%s' % nclusters,
                                  'rand_ind_%d' % rand_ind)
    elif data_analysis_type == 'synchrony':
        group_path = os.path.join(group_path,
                                  'nclusters_%s' % nclusters)
    elif data_analysis_type == 'BOLD':
        group_path = os.path.join(group_path,
                                  'nclusters_%s' % nclusters)
    if not os.path.isdir(group_path):
        os.makedirs(group_path)

    ########################################################################
    # Input aggregation
    ########################################################################
    # Aggregate input  group for all subjects in just 2 groups.
    healthy_parameters = {}
    schizo_parameters = {}
    for subject in subjects:
        # Extract the input data.
        subject_basepath = group_analysis_subject_basepath(input_basepath,
                                                           network_type,
                                                           window_type,
                                                           subject)
        if data_analysis_type == 'graph_analysis':
            data_filepath = os.path.join(subject_basepath,
                                         'nclusters_%s' % nclusters,
                                         'rand_ind_%d' % rand_ind,
                                         'graph_analysis_shannon_entropy_measures.pickle')
        elif data_analysis_type == 'synchrony':
            data_filepath = os.path.join(subject_basepath,
                                         'nclusters_%s' % nclusters,
                                         'synchrony_shannon_entropy_measures.pickle')
        elif data_analysis_type == 'BOLD':
            data_filepath = os.path.join(subject_basepath,
                                         'nclusters_%s' % nclusters,
                                         'bold_shannon.pickle')
        data = pickle.load(open(data_filepath, 'rb'))

        # Aggregate all data by measure and by healthy/schizophrenic subjects.
        for network in data:
            if int(subject.strip('sub-')) < 40000:
                if network not in healthy_parameters:
                    healthy_parameters[network] = {}
                for measure in measures:
                    if measure not in healthy_parameters[network]:
                        healthy_parameters[network][measure] = {}
                    for parameter in parameters:
                        if parameter not in healthy_parameters[network][measure]:
                            healthy_parameters[network][measure][parameter] = []
                        healthy_parameters[network][measure][parameter].append(data[network][measure][parameter])
            elif int(subject.strip('sub-')) > 50000:
                if network not in schizo_parameters:
                    schizo_parameters[network] = {}
                for measure in measures:
                    if measure not in schizo_parameters[network]:
                        schizo_parameters[network][measure] = {}
                    for parameter in parameters:
                        if parameter not in schizo_parameters[network][measure]:
                            schizo_parameters[network][measure][parameter] = []
                        schizo_parameters[network][measure][parameter].append(data[network][measure][parameter])
            else:
                 raise ValueError('Unexpected subject ID: %s.' % (subject))

    ########################################################################
    # Results generation
    ########################################################################
    print('* RESULTS')
    results = {}
    for network in data:
        print('Network: %d' % (network))
        results[network] = {}
        for measure in measures:
            print('  Measure: %s' % (measure))
            results[network][measure] = {}
            for parameter in parameters:
                print('    Parameter: %s' % (parameter))
                results[network][measure][parameter] = []
                results[network][measure][parameter + '_std'] = []
                for group in [healthy_parameters, schizo_parameters]:
                    print('      Subjects: %s' % ('healthy' if group == healthy_parameters else 'schizo'))
                    results[network][measure][parameter].append(np.mean(group[network][measure][parameter]))
                    results[network][measure][parameter + '_std'].append(np.std(group[network][measure][parameter]))
                    print('        Mean: %f' % (results[network][measure][parameter][-1]))
                    print('        STD:  %f' % (results[network][measure][parameter + '_std'][-1]))

                if group_analysis_type == 'ttest':
                    t12, p12 = stats.ttest_ind(healthy_parameters[network][measure][parameter],
                                               schizo_parameters[network][measure][parameter])
                    print('      t-value: %f' % (t12))
                    print('      p-value: %f (difference between HC and SC: %s)' %
                          (p12, 'significant' if p12 < significancy else 'not significant'))

            # Save all entropy values into a CSV file, just because.
            entropy_filename = data_analysis_type + '_' + measure + '_entropy_network_%d.csv' % (network)
            entropy_filepath = os.path.join(group_path, entropy_filename)
            entropy = {
                'Healthy': healthy_parameters[network][measure]['entropy'],
                'Schizo': schizo_parameters[network][measure]['entropy']
            }
            with open(entropy_filepath, 'wb') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(entropy.keys())
                writer.writerows(itertools.izip_longest(*entropy.values()))
    print('')

    print('--------------------------------------------------------------------')
    print('')
    print('')
    return

    #------------------------------------------------------------------------------
    # Plot Graph Theory Parameteres
    #------------------------------------------------------------------------------

    ind = np.arange(ngroups)
    width = 0.7
    if data_analysis_type == 'synchrony' or data_analysis_type == 'BOLD':
        print('plotting and saving %s' %parameters_s)
        fig, ax = plt.subplots()
        ax.bar(ind, parameters, width, yerr=parameters_std,
            ecolor='black', # black error bar
            alpha=0.5,      # transparency
            align='center')
    elif data_analysis_type == 'graph_theory':
            print('plotting and saving %s' %parameters_s[element])
            fig, ax = plt.subplots()
            ax.bar(ind, parameters[element], width, yerr=parameters_std[element],
                ecolor='black', # black error bar
                alpha=0.5,      # transparency
                align='center')
    for element in range(len(parameters_s)):
        ax.set_xticklabels(('HC', 'SC'))
        # set height of the y-axis
        if data_analysis_type == 'BOLD':
            max_y = .4
        else:
            max_y = 5
        plt.ylim([0, max_y])

        # adding horizontal grid lines
        ax.yaxis.grid(True)

        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # hiding axis ticks
        plt.tick_params(axis="both", which="both", bottom="off", top="off",
                                            labelbottom="on", left="off", right="off",
                                            labelleft="on")

        # set labels and title
        ax.set_xticks(ind)
        ax.set_ylabel('Shannon Entropy (bits)')
        ax.set_title(parameters_s[element])
        # increase font size
        plt.rcParams.update({'font.size': 28})
        plt.tight_layout()
        plt.savefig(os.path.join(output_basepath, network_type, 'rand_ind_%02d' % rand_ind,
            'group_comparison', '%02d_clusters' % nclusters,
            ''.join([parameters_s[element], '_%s.png' % group_analysis_type])))
        plt.close('all')
