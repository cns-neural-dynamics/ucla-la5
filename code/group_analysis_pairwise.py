#!/share/apps/anaconda/bin/python
from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import pdb
import csv
import itertools
from scipy import stats
from math import log, sqrt


def group_analysis_pairwise(subjects_id, n_cluster, n_groups, base_path,
        rand_ind, analysis_type, statistics_type, network_comp, significancy=.05):

    def hutcheson_test(H1, H2, s21, s22, n_labels,counts):
        """ This function is defined in accordance to the hutchenson test to
        calculate the p-values. This function assumes that a two-sided analysis
        is being considered"""
        # as it does not matter if the t-value is negative or postive, we here
        # use the abs of the t-statistic. A negative t-statistic tells you that
        # the observed mean is smaller then the hypothesised value. (This is
        # only valid for a two sided statistic analysis)
        t = abs((H1 - H2) / float(sqrt(s21 + s22)))
        df = ((s21 + s22) **2) / ( ((s21 ** 2) / float(n_labels)) + ((s22 ** 2) /
                float(n_labels)) )
        # because I am assuming that this function wil only deal with two tailed
        # distributions the resulting p-value is multiplied by two.
        p = stats.t.sf(t, df) * 2
        return t, df, p
# check if df is bigger then 1 (it should always be)

    # create folder wehere results will be stored
    if not os.path.isdir(os.path.join(base_path, network_comp, 'rand_ind_%02d' %rand_ind, 'group_comparison', '%02d_clusters' %n_cluster)):
        os.makedirs(os.path.join(base_path, network_comp, 'rand_ind_%02d' %rand_ind, 'group_comparison', '%02d_clusters' %n_cluster))

    if analysis_type == 'graph_analysis':
        g1 = {'degree_centrality_h': [],
             'path_distance_h': [],
             'small_wordness_h': [],
             'weight_h': [],
             'degree_centralitys2': [],
             'path_distances2': [],
             'small_wordnesss2': [],
             'weights2': []}

        g2 = {'degree_centrality_h': [],
             'path_distance_h': [],
             'small_wordness_h': [],
             'weight_h': [],
             'degree_centralitys2': [],
             'path_distances2': [],
             'small_wordnesss2': [],
             'weights2': []}

        g3 = {'degree_centrality_h': [],
             'path_distance_h': [],
             'small_wordness_h': [],
             'weight_h': [],
             'degree_centralitys2': [],
             'path_distances2': [],
             'small_wordnesss2': [],
             'weights2': []}

    elif analysis_type == 'synchrony':
        g1 = {'synchrony_h': [], 's2': []}
        g2 = {'synchrony_h': [], 's2': []}
        g3 = {'synchrony_h': [], 's2': []}

    elif analysis_type == 'BOLD':
        g1 = {'bold_h': [], 's2': []}
        g2 = {'bold_h': [], 's2': []}
        g3 = {'bold_h': [], 's2': []}

    else:
        raise ValueError('The analysis type was passed incorrectly')



    # load data for each subject and divide the calculated graph theory metrics
    # according to the category the participants belong to
    for subject_id in subjects_id:
        if analysis_type == 'graph_analysis':
            data_path = os.path.join(base_path, network_comp, 'rand_ind_%02d' %rand_ind,
                '%s' %(subject_id), '%02d_clusters' %n_cluster, 'graph_measures_labels_shannon_%s.pickle' %subject_id)
        elif analysis_type == 'synchrony':
            data_path = os.path.join(base_path, network_comp, 'rand_ind_%02d' %rand_ind,
                '%s' %(subject_id), '%02d_clusters' %n_cluster,
                '%s_total_shannon_entropy.pickle' %subject_id)
        elif analysis_type == 'BOLD':
            data_path = os.path.join(base_path, network_comp, 'rand_ind_%02d' %rand_ind,
                '%s' %(subject_id), '%02d_clusters' %n_cluster,
                'bold_shannon_%s.pickle' %subject_id)

        data = pickle.load(open(data_path, 'rb'))
       # separate subjects according to their grups.
       # g1: healthy controls
       # g2: prodromals
       # g3: schizophrenic
        if int(subject_id.strip('sub')) < 200:
            # iterate over all keys on the dictionary
            for key in g1:
                g1[key].append(data[key])
        elif int(subject_id.strip('sub')) > 300:
            for key in g3:
                g3[key].append(data[key])
        else:
            for key in g2:
                g2[key].append(data[key])
    groups = [g1, g2, g3]

    if analysis_type == 'graph_analysis':
        # define which parameters are of interest
        par = ['degree_centrality', 'small_wordness', 'path_distance', 'weight']
        par_interest = ['degree_centrality_h', 'small_wordness_h',
                'path_distance_h', 'weight_h']
        s2_par = ['degree_centrality_s2', 'path_distance_s2',
                'small_wordness_s2', 'weight_s2']
        gm_std = ['degree_centrality_h_std', 'path_distance_h_std',
                'small_wordness_h_std', 'weight_h_std']
        n_labels = 'gm'
        keys = []
        # append all keys to the list of keys
        for ii in range(len(par_interest)):
            keys.extend([par_interest[ii], s2_par[ii], gm_std[ii]])

        value = []
        # generate dictionary
        results = {key: list(value) for key in keys}

        for index in range(n_groups):
            g_i = groups[index]
            # select all elements in s2_par
            ii = 0
            for key in par:
                results[key + '_h'].append(np.mean (g_i[key + '_h']))
                results[key + '_h_std'].append(np.std(g_i[key + '_h']))
                if statistics_type == 'hutchenson':
                    results[key + '_s2'].append(np.mean(g_i[key + 's2']))

        # save value for each key into a dictionary which will be used
        # afterwards for saving the values
        g_all = {}
        for key in par:
            g_temp = {'g1_%s'%(key + '_h'): g1[key + '_h'],
                      'g2_%s'%(key + '_h'): g2[key + '_h'],
                      'g3_%s'%(key + '_h'): g3[key + '_h']}
            g_all.update(g_temp)
        # itertools allow dictionary entries to have different sizes
        with open('gm_entropy.csv', 'wb') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(g_all.keys())
            writer.writerows(itertools.izip_longest(*g_all.values()))

    elif analysis_type == 'synchrony':
        par_interest =  ['synchrony_h']
        s2_par = ['s2']
        n_labels = 'syn'
        keys = [par_interest[0], s2_par[0], 'synchrony_h_std']
        value = []
        # generate dictionary
        results = {key: list(value) for key in keys}

        for index in range(n_groups):
            g_i = groups[index]
            # select all elements in s2_par
            for key in par_interest:
                results[key].append(np.mean(g_i[key]))
                results[key + '_std'].append(np.std(g_i[key]))
            if statistics_type == 'hutchenson':
                results['s2'].append(np.mean(g_i['s2']))

        # save all entropy values into a csv file
        g_all = {'g1': g1[par_interest[0]], 'g2': g2[par_interest[0]],
                 'g3': g3[par_interest[0]]}

        # itertools allow dictionary entries to have different sizes
        with open('synchrony_entropy.csv', 'wb') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(g_all.keys())
            writer.writerows(itertools.izip_longest(*g_all.values()))

    elif analysis_type == 'BOLD':
        par_interest = ['bold_h']
        s2_par = ['s2']
        n_labels ='bold'
        keys = [par_interest[0], s2_par[0], 'bold_h_std']
        value = []
        # generate dictionary
        results = {key: list(value) for key in keys}

        for index in range(n_groups):
            g_i = groups[index]
            # select all elements in s2_par
            for key in par_interest:
                results[key].append(np.mean(g_i[key]))
                results[key + '_std'].append(np.std(g_i[key]))
            if statistics_type == 'hutchenson':
                results['s2'].append(np.mean(g_i['s2']))

        # save all entropy values into a csv file
        g_all = {'g1': g1[par_interest[0]], 'g2': g2[par_interest[0]],
                 'g3': g3[par_interest[0]]}
        # itertools allow dictionary entries to have different sizes
        with open('bold_entropy.csv', 'wb') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(g_all.keys())
            writer.writerows(itertools.izip_longest(*g_all.values()))

             # find significant difference between categories
    res = {}
    for ii in range(len(par_interest)):
        # qplot = stats.probplot(g1[key], plot=plt)
        # plt.savefig('plot.png')
        # pdb.set_trace()
        if statistics_type == 'hutchenson':
            # use Hutcheson t-test
            # as thecnumber of cluster is the same for both groups (it was
            # defined using k-means) and the number of labels is defined by the
            # temporal resolution of date -- which agin is the same for both
            # groups -- are passed only once to the function
            t12, df12, p12 = hutcheson_test(results[par_interest[ii]][0], results[par_interest[ii]][1],
                                            results[s2_par[ii]][0],       results[s2_par[ii]][1],
                                            data['n_labels_'  + n_labels ],
                                            data['n_classes_' + n_labels])
            t23, df23, p23 = hutcheson_test(results[par_interest[ii]][1], results[par_interest[ii]][2],
                                            results[s2_par[ii]][1],       results[s2_par[ii]][2],
                                            data['n_labels_'  + n_labels ],
                                            data['n_classes_' + n_labels])
            t13, df13, p13 = hutcheson_test(results[par_interest[ii]][0], results[par_interest[ii]][2],
                                            results[s2_par[ii]][0],       results[s2_par[ii]][2],
                                            data['n_labels_'  + n_labels ],
                                            data['n_classes_' + n_labels])

        elif statistics_type == 'ttest':
            t12, p12 = stats.ttest_ind(g1[par_interest[ii]], g2[par_interest[ii]])
            t23, p23 = stats.ttest_ind(g2[par_interest[ii]], g3[par_interest[ii]])
            t13, p13 = stats.ttest_ind(g1[par_interest[ii]], g3[par_interest[ii]])

        elif statistics_type == '1ANOVA':
            # calculate one-way ANOVA for all parameters of interest
            f, pv = stats.f_oneway(g1[par_interest[ii]], g2[par_interest[ii]],
                                   g3[par_interest[ii]])
            res[par_interest[ii]] = [f, pv]
            print('p and f-value for %s') %key
            print (pv, f)
            if pv < significancy:
                print('Significant difference btw HC, PD, SCH')

        if statistics_type == 'ttest' or statistics_type == 'hutchenson':
            # Print results
            print('num clusters: %02d' %n_cluster)
            print('p and t-value for difference between HC and PD')
            print(p12,t12)
            # compare g2 to g3
            print('p and t-value for difference between PD and SC')
            print(p23,t23)
            # compare g1 to g3
            print('p and t-value for difference between HC and SC')
            print(p13,t13)

            if p12 < significancy:
                print('Significant difference btw HC and PD when looking at %s' %key)
            elif p23 < significancy:
                print('Significant difference btw PD and SCH when looking at %s' %key)
            elif p13 < significancy:
                print('Significant difference btw HC and SCH when looking at %s' %key)
            else:
                print('No significant difference among groups for %s' %key)


    #------------------------------------------------------------------------------
    # Plot Graph Theory Parameteres
    #------------------------------------------------------------------------------
    print('-----------------------------------------------------------------------')
    ind = np.arange(n_groups)
    width = 0.35
    if analysis_type == 'graph_analysis':
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

    elif analysis_type == 'synchrony':
        parameters = results['synchrony_h']
        parameters_s = ['Synchrony']
        parameters_std = results['synchrony_h_std']
        print ('mean synchrony: %s') % str(results['synchrony_h']).strip('[]')
        print ('bold std : %s') % str(results['synchrony_h_std']).strip('[]')

    elif analysis_type == 'BOLD':
        parameters = results['bold_h']
        parameters_s = ['BOLD correlation']
        parameters_std = results['bold_h_std']
        print ('mean bold: %s') % str(results['bold_h']).strip('[]')
        print ('bold std : %s') % str(results['bold_h_std']).strip('[]')

    if analysis_type == 'synchrony' or analysis_type == 'BOLD':
        print('plotting and saving %s' %parameters_s)
        fig, ax = plt.subplots()
        ax.bar(ind, parameters, width, yerr=parameters_std,
            ecolor='black', # black error bar
            alpha=0.5,      # transparency
            align='center')
    elif analysis_type == 'graph_theory':
            print('plotting and saving %s' %parameters_s[element])
            fig, ax = plt.subplots()
            ax.bar(ind, parameters[element], width, yerr=parameters_std[element],
                ecolor='black', # black error bar
                alpha=0.5,      # transparency
                align='center')
    for element in range(len(parameters_s)):
        ax.set_xticklabels(('HC', 'PD', 'SC'))
        # set height of the y-axis
        if analysis_type == 'BOLD':
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
        plt.savefig(os.path.join(base_path, network_comp, 'rand_ind_%02d' %rand_ind,
            'group_comparison', '%02d_clusters' %n_cluster,
            ''.join([parameters_s[element], '_%s.png' %statistics_type])))
        plt.close('all')
