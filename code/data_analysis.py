#!/share/apps/anaconda/bin/python
# -*- coding: ascii -*-
from __future__ import division

import logging
import time
import json
import matplotlib
matplotlib.use('Agg')  # allow generation of images without user interface
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import glob
from scipy.signal import hilbert
from scipy.stats import entropy
from bct import (degrees_und, distance_bin, transitivity_bu, clustering_coef_bu,
                 randmio_und_connected, charpath, clustering, breadthdist, efficiency_bin,
                 community_louvain)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def dump_golden_subjects_json(output_base_path, network_type, subjects, window_size, data_analysis_type):

    parameters_list = {}
    timestamp = time.strftime("%Y%m%d%H%M%S")

    parameters_list['timestamp'] = timestamp
    parameters_list['network_type'] = network_type
    parameters_list['subjects'] = subjects
    parameters_list['window_size'] = window_size
    parameters_list['data_analysis_type'] = data_analysis_type

    with open(os.path.join(output_base_path, 'golden_subjects.json'), 'w') as json_file:
        json.dump(parameters_list, json_file, indent=4)


def calculate_subject_optimal_k(mean_synchrony, indices, k_lower=0.1, k_upper=1.0, k_step=0.01):
    """ Iterate over different threshold (k) to find the optimal value to use a
    threshold. This function finds the optimal threshold that allows the
    trade-off between cost and efficiency to be minimal.

    In order obtain the best threshold for all time points, the mean of the
    synchrony over time is used as the connectivity matrix.

    The here implemented approach was based on Bassett-2009Cognitive

    """
    # obtain the number of regions according to the passed dataset
    n_regions = mean_synchrony.shape[0]

    EC_optima = 0  # cost-efficiency
    k_optima = 0  # threshold
    for k in np.arange(k_lower, k_upper, k_step):
        # Binarise connection matrix according with the threshold
        mean_synchrony_bin = np.zeros((mean_synchrony.shape))
        for index in indices:
            if mean_synchrony[index[0], index[1]] >= k:
                mean_synchrony_bin[index[0], index[1]] = 1
        mean_synchrony_bin = mirror_array(mean_synchrony_bin)

        # calculate the shortest path length between each pair of regions using
        # the geodesic distance
        D = distance_bin(mean_synchrony_bin)
        # calculate cost
        C = estimate_cost(n_regions, mean_synchrony_bin)

        # Calculate the distance for the regional efficiency at the current
        # threshold
        E_reg = np.zeros((n_regions))
        for ii in range(D.shape[0]):
            sum_D = 0
            for jj in range(D.shape[1]):
                # check if the current value is different from inf or 0 and
                # sum it (inf represents the absence of a connection)
                if jj == ii:
                    continue
                elif D[ii, jj] == np.inf:
                    continue
                else:
                    sum_D += 1 / float(D[ii, jj])
            E_reg[ii] = sum_D / float(n_regions - 1)

        # From the regional efficiency calculate the global efficiency for the
        # current threshold
        E = np.mean(E_reg)
        # update the current optimal Efficiency
        if E - C > EC_optima:
            EC_optima = E - C
            k_optima = k
    return k_optima


def estimate_cost(N, G):
    """ Calculate costs using the formula described in Basset-2009Cognitive """
    tmp = 0
    for ii in range(G.shape[0]):
        for jj in range(G.shape[1]):
            if jj == ii:
                continue
            tmp += G[ii, jj]
        cost = tmp / float(N * (N - 1))
    return cost


def calculate_healthy_optimal_k(roi_input_basepath, output_basepath, subjects, network_type, window_size, window_type,
                                data_analysis_type, nclusters, rand_ind):

    # Calculate how many networks keys there are.
    nnetwork_keys = check_number_networks(subjects, roi_input_basepath,
                                          network_type)

    # Calculate the optimal k for each subject's network.
    healthy_k_optima = {key: [] for key in range(nnetwork_keys)}
    for subject in subjects:
        # Load the mean_synchrony for the subject.
        subject_path = data_analysis_subject_basepath(output_basepath,
                                                      network_type,
                                                      window_type,
                                                      data_analysis_type,
                                                      nclusters,
                                                      rand_ind,
                                                      subject)
        dynamic_measures = pickle.load(
            open(os.path.join(subject_path, 'dynamic_measures.pickle'),
                 'rb'))
        if len(dynamic_measures.keys()) != nnetwork_keys:
            raise ValueError('Inconsistent number of networks for ' +
                             'subject %s. In nnetwork_keys: %d. In pickle: %d.' %
                             (subject, nnetwork_keys,
                              len(dynamic_measures.keys())))
        mean_synchrony = {key: dynamic_measures[key]['mean_synchrony'] \
                          for key in range(nnetwork_keys)}
        # Calculate the optimal k for each subject's network.
        for network in range(nnetwork_keys):
            nregions = mean_synchrony[network].shape[0]
            indices = np.tril_indices(nregions)
            indices = zip(indices[0], indices[1])
            healthy_k_optima[network].append(
                calculate_subject_optimal_k(mean_synchrony[network], indices))

    # Find optimal mean of healthy subjects.
    k_optima = {}
    logging.info('')
    logging.info('* OPTIMAL MEAN THRESHOLD:')
    for network in range(nnetwork_keys):
        k_optima[network] = np.mean(healthy_k_optima[network])
        logging.info('Network %d: %3f' % (network, k_optima[network]))
    logging.info('')

    # Dump json file with optimal k and timestamp for the current analysis
    output_path = os.path.split(subject_path)[0]
    with open(os.path.join(output_path, 'optimal_k.json'), 'w') as json_file:
        json.dump(healthy_k_optima, json_file, indent=4)

    dump_golden_subjects_json(output_path, network_type, subjects, window_size, data_analysis_type)

    return


def check_number_networks(subjects, input_basepath, network_type):
    # Calculate how many networks keys there are. The number of networks for within network
    # is defined based on the known extracted ROIs.
    # We use the first subject for this purpose.
    data_path = os.path.join(input_basepath, subjects[0])
    if network_type == 'between_network':
        nnetwork_keys = 1
    elif network_type == 'within_network':
        nnetwork_keys = len(glob.glob1(data_path, "within_network_*.txt"))
    elif network_type == 'full_network':
        nnetwork_keys = 1
    else:
        raise ValueError('Unrecognised network type: %s' % (network_type))
    return nnetwork_keys


def calculate_dynamic_measures(subjects, input_basepath, output_basepath, network_type, window_size, window_type,
                               data_analysis_type, ica_aroma_type, glm_denoise, nclusters, rand_ind, pipeline_call=True):
    # Find number of network for dataset
    nnetwork_keys = check_number_networks(subjects, input_basepath, network_type)

    # Compute synchrony, metastability and mean synchrony for each subject, both
    # globally and pairwise.
    if pipeline_call:
        logging.info('* DYNAMIC MEASURES')

    for subject in subjects:
        if pipeline_call:
            logging.info('Subject ID:        %s' %(subject))

        # Calculate Hilbert transform for the network(s).
        # Import ROI data for each VOI.
        # The actual data depends on the network type.
        hilbert_transforms = {}

        # Obtain correct path for the extracted ROI according with the type of ica_aroma and glm_analysis
        if (ica_aroma_type in ['aggr', 'nonaggr']) and (glm_denoise is False):
            analysis_path = os.path.join('ica', subject, ''.join(['icaroma_', ica_aroma_type]))
        elif (ica_aroma_type in ['aggr', 'nonaggr']) and (glm_denoise is True):
            analysis_path = os.path.join('ica_glm', subject, ''.join(['icaroma_', ica_aroma_type]))
        elif (ica_aroma_type == 'no_ica') and (glm_denoise is True):
            analysis_path = os.path.join('glm', subject)

        if network_type == 'between_network':
            data_path = os.path.join(input_basepath, analysis_path, 'between_network.txt')
            data = np.genfromtxt(data_path)
            hilbert_transforms[0] = compute_hilbert_tranform(data)
        elif network_type == 'within_network':
            for network in range(nnetwork_keys):
                data_path = os.path.join(input_basepath, analysis_path, 'within_network_%d.txt' % network)
                data = np.genfromtxt(data_path)
                hilbert_transforms[network] = compute_hilbert_tranform(data)
        elif network_type == 'full_network':
            data_path = os.path.join(input_basepath, analysis_path, 'full_network.txt')
            data = np.genfromtxt(data_path)
            hilbert_transforms[0] = compute_hilbert_tranform(data)

        # Calculate data synchrony following Hellyer-2015_Cognitive.
        dynamic_measures = {}
        for network in hilbert_transforms:
            # Apply sliding windowing if required.
            hilbert_transform = hilbert_transforms[network]
            if window_type == 'sliding':
                hilbert_transform = apply_sliding_window(hilbert_transform,
                                                         window_size)

            # Calculate synchrony, metastability and mean synchrony.
            synchrony, \
            mean_synchrony, \
            metastability, \
            global_synchrony, \
            global_metastability = calculate_phi(hilbert_transform)

            # Save the results for later dump.
            dynamic_measures[network] = {
                'synchrony': synchrony,
                'metastability': metastability,
                'mean_synchrony': mean_synchrony,
                'global_synchrony': global_synchrony,
                'global_metastability': global_metastability
            }

        # Dump results for all networks, for this subject, into a pickle file.
        subject_path = data_analysis_subject_basepath(output_basepath,
                                                      network_type, window_type,
                                                      data_analysis_type, nclusters, rand_ind,
                                                      subject)
        if not os.path.exists(subject_path):
            os.makedirs(subject_path)
        pickle.dump(dynamic_measures,
                    open(os.path.join(subject_path, 'dynamic_measures.pickle'),
                         'wb'))
        if pipeline_call:
            logging.info('    Done')


def compute_hilbert_tranform(data):
    """ Perform Hilbert Transform on given data. This allows extraction of phase
     information of the empirical data"""

    # Perform Hilbert transform on filtered and demeaned data
    hiltrans = hilbert(data)
    # discard first and last 10 time steps to avoid border effects caused by the
    # Hilbert tranform
    hiltrans = hiltrans[:, 10:-10]
    return hiltrans


def slice_window_avg(array, window_size):
    """ Perform convolution on the specified sliding window. By using the
    'valid' mode the last time points will be discarded. """
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(array, window, 'valid')
    # TODO: use a less conservative approach for convolution ('full' instead of
    # 'valid').


def apply_sliding_window(hilbert_transform, window_size):
    nregions = hilbert_transform.shape[0]
    ntpoints = hilbert_transform.shape[1]
    slided = np.zeros((nregions, ntpoints - window_size + 1), dtype=complex)
    for roi in range(nregions):
        slided[roi, :] = slice_window_avg(hilbert_transform[roi, :],
                                          window_size)
    return slided


def calculate_phi(hiltrans):
    n_regions = hiltrans.shape[0]
    hilbert_t_points = hiltrans.shape[1]

    # Find indices of regions for pairwise comparison and reshuffle them
    # to obtain a tuple for each pairwise comparison.
    # As the comparision is symmetric computation power can be saved by
    # calculating only the lower diagonal matrix.
    indices = np.tril_indices(n_regions)
    indices = zip(indices[0], indices[1])

    phi = np.zeros((n_regions, n_regions, hilbert_t_points), dtype=complex)
    mean_synchrony = np.zeros((n_regions, n_regions))
    pair_metastability = np.zeros((n_regions, n_regions))
    # find the phase angle of the data
    phase_angle = np.angle(hiltrans)

    for index in indices:
        # obtain phase of the data is already saved inside hiltrans
        # calculate pairwise order parameter
        phi[index[0], index[1], :] += np.exp(phase_angle[index[0]] * 1j)
        phi[index[0], index[1], :] += np.exp(phase_angle[index[1]] * 1j)

        # divide the obtained results by the number of regions, which in
        # this case is 2.
        phi[index[0], index[1], :] /= 2
        # each value represent the synchrony between two regions over all time points
        mean_synchrony[index[0], index[1]] = np.mean(abs(phi[index[0], index[1], :]))
        # each value represent the standard deviation of synchrony over the time points
        pair_metastability[index[0], index[1]] = np.std(abs(phi[index[0], index[1], :]))
    synchrony = abs(phi)
    # mirror array so that lower and upper matrix are identical
    # Mirror each time point of synchrony
    for time_p in range(synchrony.shape[2]):
        synchrony[:, :, time_p] = mirror_array(synchrony[:, :, time_p])
    mean_synchrony = mirror_array(mean_synchrony)
    pair_metastability = mirror_array(pair_metastability)
    global_synchrony = np.mean(np.tril(mean_synchrony), -1)
    global_metastability = np.std(global_synchrony)
    return synchrony, mean_synchrony, pair_metastability, \
           global_synchrony, global_metastability


def mirror_array(array):
    """ Mirror results obtained on the lower diagonal to the Upper diagonal """
    return array + np.transpose(array) - np.diag(array.diagonal())



def estimate_small_wordness(synchrony_bin, rand_ind):
    """ Estimate small-wordness coefficient. Every time this function is called,
    a new random network is generated.

    Returns
    --------
    SM: small world coefficient
    Ds: distance matrix. Whith length of the shortest matrix
    """

    G_rand = randmio_und_connected(synchrony_bin, rand_ind)[0]
    # Calculate clustering coefficient for the random and binary
    # synchrony matrix
    CC = clustering_coef_bu(synchrony_bin)
    CC_rand = clustering_coef_bu(G_rand)
    # Estimate characteristic path lenght for random and binary
    # synchrony matrix
    # To calculate the characteristic path lenght the distance between
    # nodes is needed
    Ds = distance_bin(synchrony_bin)
    Ds_rand = distance_bin(G_rand)
    # The first element of the returned array correspond to the
    # characteristic path lenght
    L = charpath(Ds)[0]
    L_rand = charpath(Ds_rand)[0]

    CC_sm = np.divide(CC, CC_rand)
    L_sm = np.divide(L, L_rand)
    SM = np.divide(CC_sm, L_sm)
    return SM, Ds


def bold_plot_threshold(data, n_regions, threshold=1.3):
    """ This function thresholds the BOLD activity using the passed threshold  """
    # Calculate states on raw BOLD data
    z_data = np.zeros((data.shape))
    thr_data = np.zeros((data.shape))
    for VOI in range(n_regions):
        voi_mean = np.mean(data[VOI, :])
        voi_std = np.std(data[VOI, :])
        for t in range(data.shape[1]):
            z_data[VOI, t] = abs(float((data[VOI, t] - voi_mean)) / voi_std)
            # Threshold BOLD at 1.3
            if z_data[VOI, t] > threshold:
                thr_data[VOI, t] = 1
    return thr_data


def data_analysis_subject_basepath(basepath,
                                   network_type,
                                   window_type,
                                   data_analysis_type,
                                   nclusters,
                                   rand_ind,
                                   subject):
    subject_base_path = os.path.join(basepath, network_type, window_type, data_analysis_type,
                                     'nclusters_%d' % nclusters)
    if data_analysis_type == 'graph_analysis':
        return os.path.join(subject_base_path, 'rand_ind_%d' % rand_ind, subject)
    else:
        return os.path.join(subject_base_path, subject)

def data_analysis(subjects,
                  input_basepath,
                  output_basepath,
                  network_type,
                  window_type,
                  data_analysis_type,
                  ica_aroma_type,
                  glm_denoise,
                  nclusters,
                  rand_ind,
                  golden_subjects):
    ''' Compute the main analysis. This function calculates the synchrony,
    metastability and perform the graph analysis.

    Inputs:
        - subjects_id:    A list of subjects_id
        - rand_ind:       Randomisation index -- necessary for generating random
                          matrix
        - analysis_type:  Define type of analysis to be performed. Possbile
                          inputs: synchrony or BOLD.
        - nclusters:      Number of clusters used for k-means
        - sliding_window: Sliding window used to reduce noise of the time serie
        - graph_analysis: Defines if graph_analysis will be performed or not
        - window_size:    Defined size of the sliding window
        - n_time_points:  number ot time points of the data set
        - n_regions:      Define number of regions used in the data set
        - network_comp:   Define type of coparision that will be carried out.
                          between_network = compare BOLD between network
                          within_network = compare BOLD within network
                          full_network = compare BOLD from all regions used in the
                          segmentation
        - n_network:      number of networks of interested (only needed when looking at the
                          within network comparison)
    '''

    if data_analysis_type == 'BOLD' and network_type != 'full_network':
        raise ValueError('The BOLD data analysis only works with ' +
                         'full_network networks.')

    window_size = 5

    logging.info('--------------------------------------------------------------------')
    logging.info(' Data Analysis')
    logging.info('--------------------------------------------------------------------')
    logging.info('')
    logging.info('* PARAMETERS')
    logging.info('Network type:        %s' %(network_type))
    logging.info('Window type:         %s' %(window_type))
    logging.info('Window size:         %d' %(window_size))
    logging.info('Data analysis type:  %s' %(data_analysis_type))
    logging.info('ICA-AROMA type:      %s' %(ica_aroma_type))
    logging.info('Nclusters:           %d' %(nclusters))
    logging.info('Golden subjects      %s' %(golden_subjects))
    logging.info('Rand_ind:            %d' %(rand_ind))
    logging.info('')

    calculate_dynamic_measures(subjects, input_basepath, output_basepath, network_type, window_size, window_type,
                               data_analysis_type, ica_aroma_type, glm_denoise, nclusters, rand_ind)

    # Calculate the optimal k from the healthy subjects only.
    # Note: This is not needed with the BOLD data analysis. The optimal k will
    #       be used later when computing the Shannon entropy measures.
    if data_analysis_type != 'BOLD':
        # Compute optimal threshold.
        # Note: Golden subjects's id are hardcoded inside the json file and are not used for further analysis
        if golden_subjects:
            calculate_healthy_optimal_k(input_basepath, output_basepath, subjects, network_type, window_size, window_type,
                                               data_analysis_type, nclusters, rand_ind)
            return
        else:
            filepath = data_analysis_subject_basepath(output_basepath, network_type, window_type, data_analysis_type,
                                                       nclusters, rand_ind, subjects[0])
            filepath = os.path.join(os.path.split(filepath)[0], 'optimal_k.json')

            with open(filepath) as f:
                k_optima = json.load(f)

    # Calculate the Shannon entropy measures for every subject.
    for subject in subjects:

        subject_path = data_analysis_subject_basepath(output_basepath,
                                                      network_type,
                                                      window_type,
                                                      data_analysis_type,
                                                      nclusters,
                                                      rand_ind,
                                                      subject)
        if not os.path.exists(subject_path):
            os.makedirs(subject_path)

        # Behave differently based on data analysis type.
        if data_analysis_type == 'BOLD':
            # Apply a threshold to the data. We use the 1.3 default value.
            data_path = os.path.join(input_basepath, subject, ica_aroma_type, 'full_network.txt')
            data = np.genfromtxt(data_path)
            nregions = data.shape[0]
            thr_data = bold_plot_threshold(data, nregions, threshold=1.3)

            # Save thresholded image of BOLD.
            fig = plt.figure()
            plt.imshow(thr_data, interpolation='nearest')
            fig.savefig(os.path.join(subject_path, 'bold.png'))
            plt.clf()
            plt.close()

            # Perfom k-means on the BOLD signal.
            # Because BOLD only support one network and for compatibility with results.
            network = 0
            measure = 'BOLD'
            bold_shannon_entropy = {network: {measure: {}}}
            kmeans_bold = KMeans(n_clusters=nclusters)
            kmeans_bold.fit_transform(np.transpose(thr_data))
            kmeans_bold_labels = kmeans_bold.labels_

            # Calculate Shannon Entropy.
            bold_shannon_entropy[network][measure]['entropy'] = entropy(kmeans_bold_labels)
            pickle.dump(bold_shannon_entropy,
                        open(os.path.join(subject_path, 'bold_shannon.pickle'),
                             'wb'))
        else:
            # This first part of the code is common to the synchrony and graph
            # analysis data analysis types.

            # Load synchrony for the subject.
            dynamic_measures = pickle.load(
                open(os.path.join(subject_path, 'dynamic_measures.pickle'),
                     'rb'))
            nnetwork_keys = check_number_networks(subjects, input_basepath, network_type)

            if len(dynamic_measures.keys()) != nnetwork_keys:
                raise ValueError('Inconsistent number of networks for ' +
                                 'subject %s. In nnetwork_keys: %d. In pickle: %d.' %
                                 (subject, nnetwork_keys,
                                  len(dynamic_measures.keys())))
            synchrony = {key: dynamic_measures[key]['synchrony'] \
                         for key in range(nnetwork_keys)}

            # Threshold the synchrony matrix at each time point using the
            # optimal threshold and save the output.
            synchrony_bins = {}
            for network in range(nnetwork_keys):
                nregions = synchrony[network].shape[0]
                ntpoints = synchrony[network].shape[2]
                indices = np.tril_indices(nregions)
                indices = zip(indices[0], indices[1])
                synchrony_bin = np.zeros((nregions, nregions, ntpoints))
                for t in range(ntpoints):
                    for index in indices:
                        if synchrony[network][index[0], index[1], t] >= np.mean(k_optima[str(network)]):
                            synchrony_bin[index[0], index[1], t] = 1
                    synchrony_bin[:, :, t] = mirror_array(synchrony_bin[:, :, t])
                synchrony_bins[network] = synchrony_bin

            # The actual measures we save depend on the data analysis type.
            if data_analysis_type == 'synchrony':
                measure = 'synchrony'
                shannon_entropy_measures = {}
                for network in range(nnetwork_keys):
                    # Flatten the synchrony bin for the current network.
                    nregions = synchrony_bins[network].shape[0]
                    ntpoints = synchrony_bins[network].shape[2]
                    synchrony_bin_flat = np.zeros((ntpoints, nregions * nregions))
                    for t in range(ntpoints):
                        synchrony_bin_flat[t, :] = \
                            np.ndarray.flatten(synchrony_bins[network][:, :, t])

                    # Find the highest variance
                    pca = PCA(n_components=8)
                    reduced_synchrony_bin_flat = pca.fit_transform(synchrony_bin_flat)

                    # Perform k-means on this reduced space and find centroids
                    initial_kmeans = KMeans(n_clusters=nclusters)
                    initial_kmeans.fit_transform(reduced_synchrony_bin_flat)
                    # retransform centroids back to normal space
                    cluster_centres = pca.inverse_transform(initial_kmeans.cluster_centers_)
                    # use this centroids to perform kmeans

                    # Calculate the k means for synchrony.
                    subject_kmeans = KMeans(n_clusters=nclusters, init=cluster_centres)
                    subject_kmeans.fit_transform(synchrony_bin_flat)
                    synchrony_entropy = entropy(subject_kmeans.labels_)

                    # Save the results.
                    shannon_entropy_measures[network] = {}
                    shannon_entropy_measures[network][measure] = {
                        'centroids': subject_kmeans.cluster_centers_,
                        'entropy': synchrony_entropy
                    }

                # Dump the results in a pickle file.
                pickle.dump(shannon_entropy_measures,
                            open(os.path.join(subject_path,
                                              'synchrony_shannon_entropy_measures.pickle'),
                                 'wb'))
            elif data_analysis_type == 'graph_analysis':
                graph_theory_measures = {}
                shannon_entropy_measures = {}
                for network in range(nnetwork_keys):
                    nregions = synchrony_bins[network].shape[0]
                    ntpoints = synchrony_bins[network].shape[2]
                    graph_theory_measures[network] = {}

                    # Modularity/Flexibility:
                    # -------------------
                    # For the fist iteration each node is considered part of a separate community. All following iterations
                    # use previous knowledge to find only the nodes that change communities.
                    community_affiliation = np.arange(nregions) + 1
                    community_0 = 0
                    flexibility_time = np.zeros((ntpoints, nregions), dtype=bool)
                    for t in range(ntpoints):
                        W = synchrony_bins[network][:, :, t]
                        community_t, q = community_louvain(W, ci=community_affiliation)
                        # True: are the elemets that are different between time points
                        flexibility_time[t] = community_t - community_0 != 0
                        community_0 = community_t
                        community_affiliation = community_t

                    # Eliminate first time point
                    flexibility_time = flexibility_time[1:]

                    # calculate flexibility for each node
                    flexibility_regions = np.sum(flexibility_time, axis=0)
                    graph_theory_measures[network]['flexibility'] = flexibility_regions


                    # Note: Because K-means will be performed over time and of the way
                    #  the data is defined all measures will need to transposed.
                    # Degree centrality:
                    # -------------------
                    # Number of links connected to each node
                    degree_centrality = degrees_und(synchrony_bins[network])
                    graph_theory_measures[network]['degree_centrality'] = \
                        np.transpose(degree_centrality)

                    # Cluster Coefficient:
                    # ----------------------
                    # Calculate cluster Coefficient at each time point.
                    cluster_coefficient = np.zeros((nregions, ntpoints))
                    for t in range(ntpoints):
                        cluster_coefficient[:, t] = clustering_coef_bu(synchrony_bins[network][:, :, t])
                    graph_theory_measures[network]['cluster_coefficient'] = np.transpose(cluster_coefficient)


                    # # Shortest path length:
                    # # ----------------------
                    # # Calculate the shortest path length between all nodes. The matrix for eacht time point
                    # # is flattened.
                    # shortest_path = np.zeros((nregions * nregions, ntpoints))
                    # for t in range(ntpoints):
                    #     _, tmp_shortest_path = breadthdist(synchrony_bins[network][:, :, t])
                    #     shortest_path[:, t] = tmp_shortest_path.flatten()
                    # graph_theory_measures[network]['shortest_path'] = np.transpose(shortest_path)

                    # Global efficiency
                    # ----------------------
                    # Returns only a float. For comparision between groups the standard deviation will
                    # be used.
                    networks_global_efficiency = {}
                    global_efficiency = np.zeros(ntpoints)
                    for t in range(ntpoints):
                        global_efficiency[t] = efficiency_bin(synchrony_bins[network][:, :, t])
                    graph_theory_measures[network]['global_efficiency'] = global_efficiency

                    # Weight
                    # -------------------
                    weight = np.zeros((ntpoints, nregions))
                    w = np.multiply(synchrony[network], synchrony_bins[network])
                    for t in range(ntpoints):
                        for roi in range(nregions):
                            weight[t, roi] = np.average(w[:, roi, t])
                    graph_theory_measures[network]['weight'] = weight

                    # Perform K-means and calculate Shannon Entropy for each
                    # graph theory measurement.
                    kmeans = KMeans(n_clusters=nclusters)
                    shannon_entropy_measures[network] = {}
                    # Select only keys that will be used on the analysis
                    kmeans_measures = ['weight', 'cluster_coefficient', 'degree_centrality']
                    for measure in kmeans_measures:
                        # Find the components with the highest variance
                        pca = PCA(n_components=8)
                        reduced_graph_theory = pca.fit_transform(graph_theory_measures[network][measure])

                        # Perform k-means on this reduced space and find centroids
                        initial_kmeans = KMeans(n_clusters=nclusters)
                        initial_kmeans.fit_transform(reduced_graph_theory)
                        # retransform centroids back to normal space
                        cluster_centres = pca.inverse_transform(initial_kmeans.cluster_centers_)
                        # use this centroids to perform kmeans

                        # Calculate the k means for synchrony.
                        subject_kmeans = KMeans(n_clusters=nclusters, init=cluster_centres)
                        subject_kmeans.fit_transform(graph_theory_measures[network][measure])
                        shannon_entropy_measures[network][measure] = {}
                        measures = shannon_entropy_measures[network][measure]


                        # Save the results in the measure-specific dictionary.
                        measures['labels'] = subject_kmeans.labels_
                        measures['entropy'] = entropy(shannon_entropy_measures[network][measure]['labels'])


                # Dump results into three pickle files.
                pickle.dump(graph_theory_measures,
                            open(os.path.join(subject_path,
                                              'graph_analysis_measures.pickle'),
                                 'wb'))
                pickle.dump(shannon_entropy_measures,
                            open(os.path.join(subject_path,
                                              'graph_analysis_shannon_entropy_measures.pickle'),
                                 'wb'))
                pickle.dump(networks_global_efficiency,
                            open(os.path.join(subject_path,
                                              'global_efficiency.pickle'),
                                 'wb'))
            else:
                raise ValueError('Unrecognised data analysis type: %s' %
                                 (data_analysis_type))
