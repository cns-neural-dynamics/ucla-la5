#!/share/apps/anaconda/bin/python
# -*- coding: ascii -*-
from __future__ import division
import matplotlib
matplotlib.use('Agg')  # allow generation of images without user interface
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import glob
from math import log
from nitime.timeseries import TimeSeries
from nitime.analysis import FilterAnalyzer
from scipy.signal import hilbert
from bct import (degrees_und, distance_bin, transitivity_bu, clustering_coef_bu,
                 randmio_und_connected, charpath, clustering)
from sklearn.cluster import KMeans
import nibabel as nib
import progressbar
import pdb


def extract_roi(subjects,
                network_type,
                input_basepath,
                segmented_image_filename,
                segmented_regions_filename,
                output_basepath,
                network_mask_filename=None):
    """
    Iterate over all subjects and all regions (specified by the segmented_image).
     For each region find the correspoding BOLD signal. To reduce the
     dimensionality the signal belonging to the same anatomical regions are
     averaged for each time point. The BOLD signal for each region is then saved
     in a txt file for each subject

     Inputs:
         - subjects_id   : List of subjects id
         - fwhm          : used fwhm
         - input_basepath: Path to datasink
         - preprocessed_image: name of the preprocessed file that will be
                               segmented
         - segmented_image_filename : Path to the image that will be used to segment
         - segmented_regions: List of regions that will be used for the
           segmentation
         - output_basepath   : Path where the BOLD signal will be saved
         - newtork       : Define if segmented regions should be further
                           combined into networks
         - network_path  : Path to the image where the different networks are
                           specified
         - network_comp  : Allow for comparison between networks and inside
                           networks
     """

    # Only full_network does not require a network mask.
    if network_type != 'full_network' and network_mask_filename is None:
        raise ValueError('The %s network type requires a network mask.' %
                         (network_type))

    # Load the segmented regions list.
    segmented_regions = np.genfromtxt(segmented_regions_filename,
        dtype = [('numbers', '<i8'), ('regions', 'S31'), ('labels', 'i4')],
        delimiter=','
    )

    # Load the segmented image.
    segmented_image = nib.load(segmented_image_filename)
    segmented_image_data = segmented_image.get_data()

    # Extract ROI for each subjects.
    preprocessed_image_filename = 'denoised_func_data_nonaggr_filt.nii.gz'
    for subject in subjects:
        print 'Analysing subject: %s.' % subject

        # Generate the output folder.
        subject_path = os.path.join(output_basepath, subject)
        if not os.path.exists(subject_path):
            os.makedirs(subject_path)

        # Load the subject input image.
        image_filename = os.path.join(input_basepath, 'final_image', subject,
                                      preprocessed_image_filename)
        image = nib.load(image_filename)
        image_data = image.get_data()
        ntpoints = image_data.shape[3]

        if network_type == 'full_network':
            # Calculate the average BOLD signal over all regions.
            avg = np.zeros((segmented_regions['labels'].shape[0], ntpoints))
            for region in range(len(segmented_regions)):
                label = segmented_regions['labels'][region]
                boolean_mask = np.where(segmented_image_data == label)
                for t in range(ntpoints):
                    data = image_data[:, :, :, t]
                    data = data[boolean_mask[0], boolean_mask[1], boolean_mask[2]]
                    avg[region, t] = data.mean()

            # Dump the results.
            np.savetxt(os.path.join(subject_path, 'full_network.txt'),
                       avg, delimiter=' ', fmt='%5e')
        else:
            # Load the network mask.
            ntw_image = nib.load(network_mask_filename)
            ntw_data = ntw_image.get_data()
            boolean_ntw = ntw_data > 1.64

            # Find the most likely regions inside the network.
            networks = {key: [] for key in range(ntw_data.shape[3])}
            ntw_filter = np.zeros(ntw_data.shape)
            for region in range(len(segmented_regions)):
                label = segmented_regions['labels'][region]
                boolean_mask = segmented_image_data == label
                networks, ntw_filter = most_likely_roi_network(networks,
                                                               ntw_data,
                                                               ntw_filter,
                                                               boolean_ntw,
                                                               boolean_mask,
                                                               region)

            if network_type == 'within_network':
                # Calculate the BOLD signal for the selected regions in the
                # network. The labels from the original segmentation will be used
                # to identify the regions of interest
                for network in networks:
                    avg = np.zeros((len(networks[network]), ntpoints))
                    for region in range(len(networks[network])):
                        label = segmented_regions['labels'][networks[network][region]]
                        boolean_mask = np.where(segmented_image_data == label)
                        for t in range(ntpoints):
                            data = image_data[:, :, :, t]
                            data = data[boolean_mask[0], boolean_mask[1], boolean_mask[2]]
                            avg[region, t] = data.mean()
                    np.savetxt(os.path.join(subject_path,
                                            'within_network_%d.txt' % network),
                               avg, delimiter=' ', fmt='%5e')
            elif network_type == 'between_network':
                # Calculate the BOLD signal across the selected networks. This
                # procedure is similar to the full network approach, however,
                # the BOLD activity of all regions enclosed in one network is
                # taken into account.
                avg = np.zeros((ntw_data.shape[3], ntpoints))
                for network in range(ntw_data.shape[3]):
                    boolean_mask = np.where(ntw_filter[:, :, :, network] > 0)
                    for t in range(ntpoints):
                        data = image_data[:, :, :, t]
                        data = data[boolean_mask[0], boolean_mask[1], boolean_mask[2]]
                        avg[network, t] = data.mean()
                np.savetxt(os.path.join(subject_path,
                                        'between_network.txt'),
                           avg, delimiter=' ', fmt='%5e')
            else:
                raise ValueError('Unrecognised network type: %s.' % (network_type))


def most_likely_roi_network(netw, ntw_data, net_filter, boolean_ntw, boolean_mask, region):
    """ iterate over each network and find the one with the highest probability of
    including a specific region. Once the best network is found, compute the mean bold from
    that region. The mean bold will be used to compare among regions that belong
    to the same network"""

    p_network = 0
    for n_network in range(ntw_data.shape[3]):
        # find voxels that correspond to that region in the current network
        filtered_mask = np.multiply(boolean_ntw[:, :, :, n_network], boolean_mask)
        tmp = np.sum(filtered_mask) / float(np.sum(boolean_ntw[:, :, :, n_network]))
        if tmp > p_network:
            netw[n_network].append(region)
            net_filter[:, :, :, n_network] = np.add(filtered_mask, net_filter[:, :, :, n_network])
            p_network = tmp
    return netw, net_filter


def compute_hilbert_tranform(data, TR=2, upper_bound=0.07, lower_bound=0.04):
    """ Perform Hilbert Transform on given data. This allows extraction of phase
     information of the empirical data"""
    # Initialise TimeSeries object
    T = TimeSeries(data, sampling_interval=TR)
    # Initialise Filter and set band pass filter to be between 0.04 and 0.07 Hz
    # - as described on the Glerean-2012Functional paper
    F = FilterAnalyzer(T, ub=upper_bound, lb=lower_bound)
    # Obtain Filtered data from the TimeSeries object
    filtered_data = F.filtered_fourier[:]
    # Demean data: Each region (row)  is subtracted to its own mean
    for row in range(filtered_data.shape[0]):
        filtered_data[row] -= filtered_data[row].mean()

    # Perform Hilbert transform on filtered and demeaned data
    hiltrans = hilbert(filtered_data)
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
    pair_synchrony = np.zeros((n_regions, n_regions))
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
        pair_synchrony[index[0], index[1]] = np.mean(abs(phi[index[0], index[1], :]))
        # each value represent the standard deviation of synchrony over the time points
        pair_metastability[index[0], index[1]] = np.std(abs(phi[index[0], index[1], :]))
    synchrony = abs(phi)
    # mirror array so that lower and upper matrix are identical
    # Mirror each time point of synchrony
    for time_p in range(synchrony.shape[2]):
        synchrony[:, :, time_p] = mirror_array(synchrony[:, :, time_p])
    pair_synchrony = mirror_array(pair_synchrony)
    pair_metastability = mirror_array(pair_metastability)
    global_synchrony = np.mean(np.tril(pair_synchrony), -1)
    global_metastability = np.std(global_synchrony)
    return synchrony, pair_synchrony, pair_metastability, \
           global_synchrony, global_metastability


def mirror_array(array):
    """ Mirror results obtained on the lower diagonal to the Upper diagonal """
    return array + np.transpose(array) - np.diag(array.diagonal())


def calculate_optimal_k(mean_synchrony, indices, k_lower=0.1, k_upper=1.0, k_step=0.01):
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


def estimate_small_wordness(synchrony_bin, rand_ind):
    """ Estimate small-wordness coefficient. Every time this function is called,
    a new random network is generated.

    Returns
    --------
    SM: small world coefficient
    Ds: distance matrix. Whith lenght of the shortest matrix
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


def shannon_entropy(labels):
    """ Computes Shannon entropy using the labels distribution """
    n_labels = labels.shape[0]

    # check number of labels and if there is only 1 class return 0
    if n_labels <= 1:
        return 0

    # bincount return the counts in an ascendent format.
    counts = np.bincount(labels)
    probs = counts / float(n_labels)
    n_classes = probs.shape[0]

    if n_classes <= 1:
        return 0

    ent = 0

    # Compute Shannon Entropy
    ss = 0
    sq = 0
    for prob in probs:
        ent -= prob * log(prob, 2)
        ss += prob * (log(prob, 2)) ** 2
        sq += (prob * log(prob, 2)) ** 2
    s2 = ((ss - sq) / float(n_labels)) - ((n_classes - 1) / float(2 *
                                                                  (n_labels) ** 2))
    return ent, s2, n_labels, n_classes  # count is an array you might want to


# return n_classes instead


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
                                   subject):
    return os.path.join(basepath, network_type, window_type, subject)


def data_analysis(subjects,
                  input_basepath,
                  output_basepath,
                  network_type,
                  window_type,
                  data_analysis_type,
                  nclusters,
                  rand_ind,
                  graph_analysis=True, # FIXME
                  window_size=20, # FIXME
                  n_time_points=184):
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

    # Calculate how many networks keys there are.
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

    # Compute synchrony, metastability and mean synchrony for each subject, both
    # globally and pairwise.
    for subject in subjects:
        # Calculate Hilbert transform for the network(s).
        # Import ROI data for each VOI.
        # The actual data depends on the network type.
        hilbert_transforms = {}
        if network_type == 'between_network':
            data_path = os.path.join(input_basepath, subject, 'between_network.txt')
            data = np.genfromtxt(data_path)
            hilbert_transforms[0] = compute_hilbert_tranform(data)
        elif network_type == 'within_network':
            for network in range(nnetwork_keys):
                data_path = os.path.join(input_basepath, subject, 'within_network_%d.txt' % network)
                data = np.genfromtxt(data_path)
                hilbert_transforms[network] = compute_hilbert_tranform(data)
        elif network_type == 'full_network':
            data_path = os.path.join(input_basepath, subject, 'full_network.txt')
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
                                                      subject)
        if not os.path.exists(subject_path):
            os.makedirs(subject_path)
        pickle.dump(dynamic_measures,
                    open(os.path.join(subject_path, 'dynamic_measures.pickle'),
                         'wb'))

    # Calculate the optimal k from the healthy subjects only.
    # Note: This is not needed with the BOLD data analysis. The optimal k will
    #       be used later when computing the Shannon entropy measures.
    if data_analysis_type != 'BOLD':
        # Extract the list of healthy subjects.
        # Note: We assume that the input list contains all healthy subjects
        #       first, then all the schizophrenic ones.
        healthy_subjects = subjects[:(len(subjects) // 2)]

        # Calculate the threshold.
        healthy_k_optima = {key: [] for key in range(nnetwork_keys)}
        for healthy_subject in healthy_subjects:
            # Load the mean_synchrony for the subject.
            subject_path = data_analysis_subject_basepath(output_basepath,
                                                          network_type,
                                                          window_type,
                                                          healthy_subject)
            dynamic_measures = pickle.load(
                open(os.path.join(subject_path, 'dynamic_measures.pickle'),
                     'rb'))
            if len(dynamic_measures.keys()) != nnetwork_keys:
                raise ValueError('Inconsistent number of networks for ' +
                                 'subject %s. In nnetwork_keys: %d. In pickle: %d.' %
                                 (healthy_subject, nnetwork_keys,
                                  len(dynamic_measures.keys())))
            mean_synchrony = {key: dynamic_measures[key]['mean_synchrony'] \
                              for key in range(nnetwork_keys)}

            # Calculate the optimal k for each subject's network.
            for network in range(nnetwork_keys):
                nregions = mean_synchrony[network].shape[0]
                indices = np.tril_indices(nregions)
                indices = zip(indices[0], indices[1])
                healthy_k_optima[network].append(
                    calculate_optimal_k(mean_synchrony[network], indices))

        # Find optimal mean of healthy subjects.
        k_optima = {}
        for network in range(nnetwork_keys):
            k_optima[network] = np.mean(healthy_k_optima[network])

        print ('Optimal mean threshold:')
        for network in range(nnetwork_keys):
            print('Network %d: %3f' % (network, k_optima[network]))

    # Calculate the Shannon entropy measures for every subject.
    for subject in subjects:
        subject_path = data_analysis_subject_basepath(output_basepath,
                                                      network_type,
                                                      window_type,
                                                      subject)
        if not os.path.exists(subject_path):
            os.makedirs(subject_path)

        # Behave differently based on data analysis type.
        if data_analysis_type == 'BOLD':
            # Apply a threshold to the data. We use the 1.3 default value.
            data_path = os.path.join(input_basepath, subject, 'full_network.txt')
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
            bold_shannon_entropy = {}
            kmeans_bold = KMeans(n_clusters=nclusters)
            kmeans_bold.fit_transform(np.transpose(thr_data))
            kmeans_bold_labels = kmeans_bold.labels_

            # Calculate Shannon Entropy.
            bold_shannon_entropy['bold_h'], bold_shannon_entropy['s2'], \
            bold_shannon_entropy['n_labels_bold'], \
            bold_shannon_entropy['n_classes_bold'] = shannon_entropy(kmeans_bold_labels)
            save_path = os.path.join(subject_path, 'nclusters_%d' % (nclusters))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            pickle.dump(bold_shannon_entropy,
                        open(os.path.join(save_path, 'bold_shannon.pickle'),
                             'wb'))
        else:
            # This first part of the code is common to the synchrony and graph
            # analysis data analysis types.

            # Load synchrony for the subject.
            dynamic_measures = pickle.load(
                open(os.path.join(subject_path, 'dynamic_measures.pickle'),
                     'rb'))
            if len(dynamic_measures.keys()) != nnetwork_keys:
                raise ValueError('Inconsistent number of networks for ' +
                                 'subject %s. In nnetwork_keys: %d. In pickle: %d.' %
                                 (healthy_subject, nnetwork_keys,
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
                        if synchrony[network][index[0], index[1], t] >= k_optima[network]:
                            synchrony_bin[index[0], index[1], t] = 1
                    synchrony_bin[:, :, t] = mirror_array(synchrony_bin[:, :, t])
                synchrony_bins[network] = synchrony_bin

            # The actual measures we save depend on the data analysis type.
            if data_analysis_type == 'synchrony':
                shannon_entropy_measures = {}
                for network in range(nnetwork_keys):
                    # Flatten the synchrony bin for the current network.
                    nregions = synchrony_bins[network].shape[0]
                    ntpoints = synchrony_bins[network].shape[2]
                    synchrony_bin_flat = np.zeros((ntpoints, nregions * nregions))
                    for t in range(ntpoints):
                        synchrony_bin_flat[t, :] = \
                            np.ndarray.flatten(synchrony_bins[network][:, :, t])

                    # Calculate the k means for synchrony.
                    kmeans = KMeans(n_clusters=nclusters)
                    kmeans.fit_transform(synchrony_bin_flat)
                    kmeans_labels = kmeans.labels_
                    synchrony_h, \
                    s2, \
                    n_labels_syn, \
                    n_classes_syn = shannon_entropy(kmeans_labels)

                    # Save the results.
                    shannon_entropy_measures[network] = {
                        'centroids': kmeans.cluster_centers_,
                        'synchrony_h': synchrony_h,
                        's2': s2,
                        'n_labels_syn': n_labels_syn,
                        'n_classes_syn': n_classes_syn
                    }

                # Dump the results in a pickle file.
                save_path = os.path.join(subject_path,
                                         'nclusters_%d' % (nclusters))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                pickle.dump(shannon_entropy_measures,
                            open(os.path.join(subject_path,
                                              'synchrony_shannon_entropy_measures.pickle'),
                                 'wb'))
            elif data_analysis_type == 'graph_analysis':
                shannon_entropy_measures = {}
                for network in range(nnetwork_keys):
                    nregions = synchrony_bins[network].shape[0]
                    ntpoints = synchrony_bins[network].shape[2]
                    shannon_entropy_measures[network] = {}

                    # Degree centrality:
                    # -------------------
                    degree_centrality = np.transpose(
                        degrees_und(synchrony_bins[network]))
                    shannon_entropy_measures[network]['degree_centrality'] = \
                        degree_centrality

                    # Iterate over time to obtain different network measurements.
                    weight = np.zeros((ntpoints, nregions))
                    w = np.multiply(synchrony[network], synchrony_bins[network])
                    SM = []
                    Ds = []
                    Ds_flat = []
                    for t in range(ntpoints):
                        # Weight
                        # -------------------
                        # Use the thresholded matrix to calculate the average
                        # weight over all regions.
                        for roi in range(nregions):
                            weight[t, roi] = np.average(w[:, roi, t])
                        shannon_entropy_measures[network]['weight'] = weight

                        # Transitivity:
                        # -------------------
                        transitivity = transitivity_bu(synchrony_bins[network][:, :, t])
                        shannon_entropy_measures[network]['transitivity'] = transitivity

                        # Small-worldness
                        # ------------------
                        ncomponents = clustering.number_of_components(
                            synchrony_bins[network][:, :, t])
                        components = range(1, ncomponents + 1)
                        regions2component_mapping = clustering.get_components(
                            synchrony_bins[network][:, :, t])[0]
                        regions_per_component = np.bincount(regions2component_mapping)

                        # Eliminate all components that only have one region.
                        synchrony_bin_sw = synchrony_bins[network]
                        components_to_delete = []
                        for component in components:
                            if regions_per_component[component] == 1:
                                indices_to_eliminate = np.where(
                                    regions2component_mapping == component)[0][0]
                                print('Node %d was eliminated at timepoint %d' %
                                      (indices_to_eliminate, t))
                                synchrony_bin_sw = np.delete(
                                    np.delete(synchrony_bin_sw[:, :, t],
                                              indices_to_eliminate, 0),
                                    indices_to_eliminate, 1)
                                components_to_delete.append(component)
                        components = list(set(components) -
                                          set(components_to_delete))

                        # Calculate the small-worldness for each remaining
                        # component.
                        SM_component = {}
                        Ds_component = {}
                        for component in components:
                            # Select the portion of the synchrony bin containing
                            # only the regions for this component.
                            # TODO: Use slicing directly.
                            indices_to_keep = np.where(regions2component_mapping == component)[0]
                            all_indices = range(nregions)
                            indices_to_eliminate = np.delete(all_indices,
                                                             indices_to_keep, 0)
                            synchrony_bin_component = np.delete(
                                np.delete(synchrony_bin_sw[:, :, t],
                                          indices_to_eliminate, 0),
                                indices_to_eliminate, 1)

                            # Compute the small worldness for this component.
                            sm, ds = estimate_small_wordness(
                                synchrony_bin_component[:, :], rand_ind)
                            SM_component[component] = sm
                            Ds_component[component] = ds

                            # Flatten the synchrony matrix and path_distance so
                            # that it can be given as argument for the K-means
                            Ds_flat_component = np.ndarray.flatten(ds)
                        SM.append(SM_component)
                        Ds.append(Ds_component)
                        Ds_flat.append(Ds_flat_component)

                    # Save results to a dictionary.
                    shannon_entropy_measures[network]['SM'] = SM
                    shannon_entropy_measures[network]['Ds_flat'] = Ds_flat

                # Dump results into a pickle file.
                save_path = os.path.join(subject_path,
                                         'nclusters_%d' % (nclusters),
                                         'rand_ind_%d' % (rand_ind))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                pickle.dump(shannon_entropy_measures,
                            open(os.path.join(save_path,
                                              'graph_analysis_shannon_entropy_measures.pickle'),
                                 'wb'))

                # ---------------------------------------------------------------------
                # Clustering
                # ---------------------------------------------------------------------
                # Perform K-means and calculate Shannon Entropy for each graph theory
                # measurement
                # TODO: think how you want to transform SM and DM into one single
                # matrix
                kmeans = KMeans(n_clusters=nclusters)
                graph_measures_labels = {}
                for key in graph_measures:
                    pdb.set_trace()
                    kmeans.fit_transform(graph_measures[key])
                    graph_measures_labels[key] = kmeans.labels_
                    graph_measures_labels[key + '_h'], graph_measures_labels[key + 's2'], \
                    graph_measures_labels['n_labels_gm'], \
                    graph_measures_labels['n_classes_gm'] = shannon_entropy(graph_measures_labels[key])

                pickle.dump(graph_measures_labels, open(os.path.join(output_basepath,
                                                                     grouping_type, network_type,
                                                                     'rand_ind_%02d' % rand_ind,
                                                                     '%s' % subject, '%02d_clusters' % nclusters,
                                                                     'graph_measures_labels_shannon_%s.pickle' % (
                                                                     subject)), 'wb'))

                pass
            else:
                raise ValueError('Unrecognised data analysis type: %s' %
                                 (data_analysis_type))


            print('Done!')
            print ('--------------------------------------------------------------')
