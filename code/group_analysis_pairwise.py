from __future__ import division

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import csv
import itertools
import logging
from scipy import stats

from data_analysis import data_analysis_subject_basepath

def group_analysis_group_basepath(basepath,
                                  network_type,
                                  window_type,
                                  data_analysis_type,
                                  nclusters,
                                  rand_ind,
                                  group_analysis_type):
    group_basepath = data_analysis_subject_basepath(basepath,
                                                    network_type,
                                                    window_type,
                                                    data_analysis_type,
                                                    nclusters,
                                                    rand_ind,
                                                    'dummy')
    group_basepath = os.path.split(group_basepath)[0]
    return os.path.join(group_basepath, group_analysis_type)

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

    logging.info('--------------------------------------------------------------------')
    logging.info(' Group analysis')
    logging.info('--------------------------------------------------------------------')
    logging.info('')
    logging.info('* PARAMETERS')
    logging.info('network type:        %s' %(network_type))
    logging.info('windowk type:        %s' %(window_type))
    logging.info('data analysis type:  %s' %(data_analysis_type))
    logging.info('group analysis type: %s' %(group_analysis_type))
    logging.info('nclusters:           %d' %(nclusters))
    logging.info('rand_ind:            %d' %(rand_ind))
    logging.info('')

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
    group_output_basepath = group_analysis_group_basepath(output_basepath,
                                                          network_type,
                                                          window_type,
                                                          data_analysis_type,
                                                          nclusters,
                                                          rand_ind,
                                                          group_analysis_type)
    if not os.path.isdir(group_output_basepath):
        os.makedirs(group_output_basepath)

    ########################################################################
    # Input aggregation
    ########################################################################
    # Aggregate input  group for all subjects in just 2 groups.
    healthy_parameters = {}
    schizo_parameters = {}
    for subject in subjects:
        # Extract the input data.
        subject_basepath = data_analysis_subject_basepath(input_basepath,
                                                          network_type,
                                                          window_type,
                                                          data_analysis_type,
                                                          nclusters,
                                                          rand_ind,
                                                          subject)
        if not os.path.isdir(subject_basepath):
            raise IOError('Input folder not found: %s. Have you run the data analysis yet?' %
                          subject_basepath)
        if data_analysis_type == 'graph_analysis':
            data_filepath = os.path.join(subject_basepath,
                                         'graph_analysis_shannon_entropy_measures.pickle')
            # load data for flexibilty
            graph_measures_path = os.path.join(subject_basepath,
                                               'graph_analysis_measures.pickle')
            data_flexibility = pickle.load(open(graph_measures_path, 'rb'))

        elif data_analysis_type == 'synchrony':
            data_filepath = os.path.join(subject_basepath,
                                         'synchrony_shannon_entropy_measures.pickle')
        elif data_analysis_type == 'BOLD':
            data_filepath = os.path.join(subject_basepath,
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
                if 'flexibility' not in healthy_parameters[network]:
                    healthy_parameters[network]['flexibility'] = {}
                    healthy_parameters[network]['flexibility']['mean'] = []
                healthy_parameters[network]['flexibility']['mean'].append(data_flexibility[network]['flexibility'])
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
                if 'flexibility' not in schizo_parameters[network]:
                    schizo_parameters[network]['flexibility'] = {}
                    schizo_parameters[network]['flexibility']['mean'] = []
                schizo_parameters[network]['flexibility']['mean'].append(data_flexibility[network]['flexibility'])
            else:
                 raise ValueError('Unexpected subject ID: %s.' % (subject))

    ########################################################################
    # Results generation
    ########################################################################
    logging.info('* RESULTS')
    results = {}
    for network in healthy_parameters:
        logging.info('Network: %d' % (network))
        results[network] = {}
        for measure in healthy_parameters[network].keys():
            logging.info('  Measure: %s' % (measure))
            results[network][measure] = {}
            for parameter in healthy_parameters[network][measure].keys():
                logging.info('    Parameter: %s' % (parameter))
                results[network][measure][parameter] = []
                results[network][measure][parameter + '_std'] = []
                for group in [healthy_parameters, schizo_parameters]:
                    logging.info('      Subjects: %s' % ('healthy' if group == healthy_parameters else 'schizo'))
                    results[network][measure][parameter].append(np.mean(group[network][measure][parameter]))
                    results[network][measure][parameter + '_std'].append(np.std(group[network][measure][parameter]))
                    logging.info('        Mean: %f' % (results[network][measure][parameter][-1]))
                    logging.info('        STD:  %f' % (results[network][measure][parameter + '_std'][-1]))

                if group_analysis_type == 'ttest':
                    t12, p12 = stats.ttest_ind(healthy_parameters[network][measure][parameter],
                                               schizo_parameters[network][measure][parameter])
                    logging.info('      t-value: %f' % (t12))
                    logging.info('      p-value: %f (difference between HC and SC: %s)' %
                          (p12, 'significant' if p12 < significancy else 'not significant'))

            # Save all entropy values into a CSV file, just because.
            group_results_filename = measure + '_' + parameter + '_network_%d.csv' % (network)
            group_results_filepath = os.path.join(group_output_basepath, group_results_filename)
            group_results = {
                'Healthy': healthy_parameters[network][measure][parameter],
                'Schizo': schizo_parameters[network][measure][parameter]
            }
            with open(group_results_filepath, 'wb') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(group_results.keys())
                writer.writerows(itertools.izip_longest(*group_results.values()))
    logging.info('')

    logging.info('--------------------------------------------------------------------')
    logging.info('')
    logging.info('')
    return

    #------------------------------------------------------------------------------
    # Plot Graph Theory Parameteres
    #------------------------------------------------------------------------------

    ind = np.arange(ngroups)
    width = 0.7
    if data_analysis_type == 'synchrony' or data_analysis_type == 'BOLD':
        logging.info('plotting and saving %s' %parameters_s)
        fig, ax = plt.subplots()
        ax.bar(ind, parameters, width, yerr=parameters_std,
            ecolor='black', # black error bar
            alpha=0.5,      # transparency
            align='center')
    elif data_analysis_type == 'graph_theory':
            logging.info('plotting and saving %s' %parameters_s[element])
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
        plt.savefig(os.path.join(group_output_basepath,
                                 ''.join([parameters_s[element], '.png'])))
        plt.close('all')
