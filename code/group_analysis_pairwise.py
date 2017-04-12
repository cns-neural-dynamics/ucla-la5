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

def hutcheson_test(H1, H2, s21, s22, nlabels):
    """ This function is defined in accordance to the hutchenson test to
    calculate the p-values. This function assumes that a two-sided analysis
    is being considered"""
    # FIXME: calculate s2 based on the formula on Marxico
    t = abs((H1 - H2) / float(sqrt(s21 + s22)))
    df = ((s21 + s22) **2) / (((s21 ** 2) / float(nlabels)) + ((s22 ** 2) /
                                                               float(nlabels)))
    # because I am assuming that this function wil only deal with two tailed
    # distributions the resulting p-value is multiplied by two.
    # check if df is bigger then 1 (it should always be)
    p = stats.t.sf(t, df) * 2
    return t, df, p

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
                            significancy=.05): # FIXME: Define this in code, not as input.

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
                    healthy_parameters[network][measure] = {}
                    for parameter in parameters:
                        if parameter not in healthy_parameters[network]:
                            healthy_parameters[network][measure][parameter] = []
                        healthy_parameters[network][measure][parameter].append(data[network][measure][parameter])
            elif int(subject.strip('sub-')) > 50000:
                if network not in schizo_parameters:
                    schizo_parameters[network] = {}
                for measure in measures:
                    schizo_parameters[network][measure] = {}
                    for parameter in parameters:
                        if parameter not in schizo_parameters[network]:
                            schizo_parameters[network][measure][parameter] = []
                        schizo_parameters[network][measure][parameter].append(data[network][measure][parameter])
            else:
                 raise ValueError('Unexpected subject ID: %s.' % (subject))

    # Generate the results dictionary.
    results = {}
    for network in data:
        results[network] = {}
        if data_analysis_type == 'synchrony':
            measure = 'synchrony'
            results[network][measure] = {}
            for group in [healthy_parameters, schizo_parameters]:
                for parameter in parameters:
                    if parameter not in results[network][measure]:
                        results[network][measure][parameter] = []
                    results[network][measure][parameter].append(np.mean(group[network][measure][parameter]))
                    if parameter + '_std' not in results[network][measure]:
                        results[network][measure][parameter + '_std'] = []
                    results[network][measure][parameter + '_std'].append(np.std(group[network][measure][parameter]))
                    # re-implement for hutchenson
                    # if group_analysis_type == 'hutchenson':
                    #     results['s2'].append(np.mean(g_i['s2']))

            # Save all entropy values into a CSV file, just because.
            entropy_filepath = os.path.join(group_path,
                                            'synchrony_entropy_network_%d.csv' %
                                            (network))
            entropy = {
                'Healthy': healthy_parameters[network][measure]['entropy'],
                'Schizo': schizo_parameters[network][measure]['entropy']
            }
            with open(entropy_filepath, 'wb') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(entropy.keys())
                writer.writerows(itertools.izip_longest(*entropy.values()))
        elif data_analysis_type == 'graph_analysis':
            results[network] = {}
            for measure in measures:
                results[network][measure] = {}
                for group in [healthy_parameters, schizo_parameters]:
                    for parameter in parameters:
                        if parameter not in results[network][measure]:
                            results[network][measure][parameter] = []
                        results[network][measure][parameter].append(np.mean(group[network][measure][parameter]))
                        if parameter + '_std' not in results[network][measure]:
                            results[network][measure][parameter + '_std'] = []
                        results[network][measure][parameter + '_std'].append(np.std(group[network][measure][parameter]))

            value = []
            # generate dictionary
            results = {key: list(value) for key in keys}

            for index in range(ngroups):
                g_i = groups[index]
                # select all elements in s2_par
                ii = 0
                for key in par:
                    results[key + '_h'].append(np.mean (g_i[key + '_h']))
                    results[key + '_h_std'].append(np.std(g_i[key + '_h']))
                    if group_analysis_type == 'hutchenson':
                        results[key + '_s2'].append(np.mean(g_i[key + 's2']))

            # save value for each key into a dictionary which will be used
            # afterwards for saving the values
            g_all = {}
            for key in par:
                g_temp = {'g1_%s'%(key + '_h'): g1[key + '_h'],
                          'g2_%s'%(key + '_h'): g2[key + '_h'],
                          }
                g_all.update(g_temp)
            # itertools allow dictionary entries to have different sizes
            with open('gm_entropy.csv', 'wb') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(g_all.keys())
                writer.writerows(itertools.izip_longest(*g_all.values()))

        elif data_analysis_type == 'BOLD':
            measure = 'BOLD'
            results[network][measure] = {}
            for group in [healthy_parameters, schizo_parameters]:
                for parameter in parameters:
                    if parameter not in results[network][measure]:
                        results[network][measure][parameter] = []
                    results[network][measure][parameter].append(np.mean(group[network][measure][parameter]))
                    if parameter + '_std' not in results[network][measure]:
                        results[network][measure][parameter + '_std'] = []
                    results[network][measure][parameter + '_std'].append(np.std(group[network][measure][parameter]))
                    # re-implement for hutchenson
                    # if group_analysis_type == 'hutchenson':
                    #     results['s2'].append(np.mean(g_i['s2']))


    res = {}
    for ii in range(len(parameters)):
        # qplot = stats.probplot(g1[key], plot=plt)
        # plt.savefig('plot.png')
        # pdb.set_trace()
        if group_analysis_type == 'hutchenson':
            # use Hutcheson t-test
            # as thecnumber of cluster is the same for both groups (it was
            # defined using k-means) and the number of labels is defined by the
            # temporal resolution of date -- which agin is the same for both
            # groups -- are passed only once to the function
            t12, df12, p12 = hutcheson_test(results[parameters[ii]][0], results[parameters[ii]][1],
                                            results[s2_par[ii]][0],       results[s2_par[ii]][1],
                                            data['n_labels_'  + n_labels ],
                                            data['n_classes_' + n_labels])

        elif group_analysis_type == 'ttest':
            t12, p12 = stats.ttest_ind(g1[parameters[ii]], g2[parameters[ii]])

        if group_analysis_type == 'ttest' or group_analysis_type == 'hutchenson':
            # Print results
            print('num clusters: %02d' % nclusters)
            print('p and t-value for difference between HC and PD')
            print(p12,t12)

            if p12 < significancy:
                print('Significant difference btw HC and PD when looking at %s' %key)
            else:
                print('No significant difference among groups for %s' %key)


    #------------------------------------------------------------------------------
    # Plot Graph Theory Parameteres
    #------------------------------------------------------------------------------
    print('-----------------------------------------------------------------------')
    ind = np.arange(ngroups)
    width = 0.7
    if data_analysis_type == 'graph_analysis':
        parameters = [results['degree_centrality_h'],
                      results['small_wordness_h'],
                      results['path_distance_h'],
                      results['weight_h']]
        parameters_s = ['Degree Centrality', 'Small Worldness', 'Path Distance', 'Weight']
        parameters_std = [results['degree_centrality_h_std'],
                          results['small_wordness_h_std'],
                          results['path_distance_h_std'],
                          results['weight_h_std']]
        print (results['degree_centrality_h'],
               results['small_wordness_h'],
               results['path_distance_h'],
               results['weight_h'])

    elif data_analysis_type == 'synchrony':
        parameters = results['synchrony_h']
        parameters_s = ['Synchrony']
        parameters_std = results['synchrony_h_std']
        print ('mean synchrony: %s') % str(results['synchrony_h']).strip('[]')
        print ('synchrony std : %s') % str(results['synchrony_h_std']).strip('[]')

    elif data_analysis_type == 'BOLD':
        parameters = results['bold_h']
        parameters_s = ['BOLD correlation']
        parameters_std = results['bold_h_std']
        print ('mean bold: %s') % str(results['bold_h']).strip('[]')
        print ('bold std : %s') % str(results['bold_h_std']).strip('[]')

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
