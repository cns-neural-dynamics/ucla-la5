#!/share/apps/anaconda/bin/python
# -*- coding: ascii -*-

from __future__ import division
import matplotlib
matplotlib.use('Agg') #allow generation of images without user interface
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import argparse
import sys
from math import log
from nitime.timeseries import TimeSeries
from nitime.analysis import FilterAnalyzer
from scipy.signal import hilbert
from bct import (degrees_und, distance_bin, transitivity_bu, clustering_coef_bu,
                randmio_und_connected, charpath, clustering)
from sklearn.cluster import KMeans
import nibabel as nib
import time
import progressbar
import pdb

def extract_roi(subjects_id, fwhm, data_sink_path, preprocessed_image,
        segmented_image_path, segmented_regions, output_path, network=False,
        network_path=None, network_comp=None):
    """
    Iterate over all subjects and all regions (specified by the segmented_image).
     For each region find the correspoding BOLD signal. To reduce the
     dimensionality the signal belonging to the same anatomical regions are
     averaged for each time point. The BOLD signal for each region is then saved
     in a txt file for each subject

     Inputs:
         - subjects_id   : List of subjects id
         - fwhm          : used fwhm
         - data_sink_path: Path to datasink
         - preprocessed_image: name of the preprocessed file that will be
                               segmented
         - segmented_image_path : Path to the image that will be used to segment
         - segmented_regions: List of regions that will be used for the
           segmentation
         - output_path   : Path where the BOLD signal will be saved
         - newtork       : Define if segmentated regions should be further
                           combined into networks
         - network_path  : Path to the image where the different networks are
                           specified
         - network_comp  : Allow for comparison between networks and inside
                           networks
     """
    # Load segmented image and obtain data from the image
    segmented_image =  nib.load(segmented_image_path)
    # Load the segmented image data from the nii file
    segmented_image_data = segmented_image.get_data()

    for subject_id in subjects_id:
        print('Analysing Subject: %s') %subject_id
        image_path = os.path.join(data_sink_path, 'final_output', subject_id, fwhm,
                preprocessed_image)
        # Load subjects preprocessed image and extract data from the image
        image = nib.load(image_path)
        image_data = image.get_data()
        # Obtain different time points (corrisponding to the image TR)
        time = image_data.shape[3]
        # Initialise matrix where the averaged BOLD signal for each region will be
        # saved
        avg = np.zeros((segmented_regions['labels'].shape[0], time))
        # Find the voxels on the subject's image that correspond to the voxels on
        # the labeled image for each time point and calculate the mean BOLD response
        # Combine the diffferent segmented regions into pre-specified networks
        pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),
               progressbar.Bar()], maxval=len(segmented_regions)).start()
        if network:
            # Load the network maks
            ntw_img = nib.load(network_path)
            ntw_data = ntw_img.get_data()
            boolean_ntw = ntw_data > 1.64
        # Dictionary where the belonging for each network will be saved
        netw = {key: [] for key in range(ntw_data.shape[3])}
        net_filter = np.zeros(ntw_data.shape)
        for region in range(len(segmented_regions)):
            pbar.update(region)
            label = segmented_regions['labels'][region]
            boolean_mask = segmented_image_data == label

            if network:
               netw, net_filter =  most_likely_roi_network(netw, ntw_data,
                       net_filter, boolean_ntw, boolean_mask, region)
            else:
                 for t in range(time):
                     data = image_data[:, :, :, t]
                     boolean_mask = np.where(segmented_image_data == label)
                     data = data[boolean_mask[0], boolean_mask[1], boolean_mask[2]]
                     # for all regions calculate the mean BOLD at each time point
                     avg[region, t] = data.mean()

        pbar.finish()
        # obtain BOLD data
        if network_comp == 'within_network':
            # calculate the bold for each region inside a specific
            # network
            for network in netw:
                avg_bold = np.zeros((len(netw[network]), time))
                for region in range(len(netw[network])):
                    label = segmented_regions['labels'][netw[network][region]]
                    boolean_mask = segmented_image_data == label
                    for t in range(time):
                        data = image_data[ :, :, :, t]
                        boolean_mask = np.where(segmented_image_data == label)
                        data = data[boolean_mask[0], boolean_mask[1], boolean_mask[2]]
                        avg_bold[region, t] = data.mean()
                np.savetxt(os.path.join(output_path , subject_id, 'fwhm_5',
                    'within_network_%d.txt' %network), avg_bold, delimiter=' ', fmt='%5e')
            # save avg_bold
        elif network_comp == 'between_network':
            ntw_avg = np.zeros((ntw_data.shape[3], time))
            # calculate the main bold across networks
            for t in range(time):
                data = image_data[:,:,:,t]
                for network in range(ntw_data.shape[3]):
                    boolean_filtered_data = np.where(net_filter[:,:,:,network]>0)
                    data_filtered = data[boolean_filtered_data[0],
                                         boolean_filtered_data[1],
                                         boolean_filtered_data[2]]
                    ntw_avg[network,t] = data_filtered.mean()
            # save netw_avg
            np.savetxt(os.path.join(output_path , subject_id, 'fwhm_5', 'between_network.txt') , ntw_avg, delimiter=' ', fmt='%5e')
        elif network_comp == 'full_network':
            # save data into a text file
            np.savetxt(os.path.join(output_path, subject_id, 'fwhm5',
                '%s.txt' %subject_id), avg, delimiter=' ', fmt='%5e')

        print 'Subject %s: Done!' % subject_id

def most_likely_roi_network(netw, ntw_data, net_filter, boolean_ntw, boolean_mask, region):
    """ iterate over each network that corresponds to a different volume on the
    ntw_data matrix and find the one with the highest probability of including a
    specific region. Once the best network is found, compute the mean bold from
    that region. The mean bold will be used to compare among regions that belong
    to the same network"""

    p_network = 0
    for n_network in range(ntw_data.shape[3]):
         filtered_mask = np.multiply(boolean_ntw[:,:,:,n_network], boolean_mask)
         tmp = np.sum(filtered_mask)/float(np.sum(boolean_ntw[:,:,:,n_network]))
         if tmp > p_network:
            netw[n_network].append(region)
            net_filter[:,:,:,n_network] = np.add(filtered_mask, net_filter[:,:,:,n_network])
            p_network = tmp
    return netw, net_filter

def min_max_scalling(data):
    """Return matrix scalled by its maximum and minimum."""
    min_d = data.min()
    max_d = data.max()
    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            for volume in range(data.shape[2]):
                if data[row, column, volume] == 0:
                    continue
                else:
                    data[row, column, volume] = \
                float(data[row, column, volume] - min_d)/ (max_d - min_d)
    return data

def hilbert_tranform(data, TR=2, upper_bound=0.07, lower_bound=0.04):
    """ Perform Hilbert Transform on given data. """
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
    """ Perform convolution on the specified sliding window """
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(array, window, 'valid')

def calculate_phi(indices, n_regions, hilbert_t_points, hiltrans):
    phi = np.zeros((n_regions, n_regions, hilbert_t_points), dtype=complex)
    for index in indices:
        # obtain synchrony measure for pairwise data
        phi[index[0], index[1], :] += np.exp(np.angle(hiltrans[index[0]]) * 1j)
        phi[index[0], index[1], :] += np.exp(np.angle(hiltrans[index[1]]) * 1j)
        # divide the obatined results by the number of regions, which in
        # this case is 2.
        phi[index[0], index[1], :] /= 2
    return phi

def calculate_synchronies(indices, n_regions, hilbert_t_points, phi):
    """ This function calculates both the synchrony over all time point as well
    as the mean synchrony value (average over time). The latter will be used for
    the cost-efficiency calculation """
    mean_synchrony = np.zeros((n_regions, n_regions))
    synchrony = np.zeros((n_regions, n_regions, hilbert_t_points))
    for index in indices:
       for t in range(hilbert_t_points):
           synchrony[index[0], index[1], t] = np.mean(abs(phi[index[0], index[1], t]))
       mean_synchrony[index[0], index[1]] = np.mean(synchrony[index[0], index[1], :])
    mean_synchrony = mirror_array(mean_synchrony)
    # mirror synchrony matrix
    for t in range(hilbert_t_points):
        synchrony[:, :, t] = mirror_array(synchrony[:, :, t])
    return synchrony, mean_synchrony

def calculate_metastability(indices, n_regions, phi):
    """ Calculate metastability over all time points """
    metastability = np.zeros((n_regions, n_regions))
    for index in indices:
        metastability[index[0], index[1]] = np.std(abs(phi[index[0], index[1], :]))
    metastability = mirror_array(metastability)
    return metastability

def mirror_array(array):
    """ Mirror results obtained on the lower diagonal to the Upper diagonal """
    return array + np.transpose(array) - np.diag(array.diagonal())

def calculate_optimal_k(mean_synchrony, indices,n_regions=82, k_lower=0.1, k_upper=1.0, k_step=0.01):
    """ Iterate over different threshold (k) to find the optimal value. In
    order obtain the best threshold for all time points, the mean of the
    synchrony over time is used as the connectivity matrix. """
    EC_optima = 0                                         # cost-efficiency
    k_optima = 0                                          # threshold
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
                elif D[ii,jj] == np.inf:
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
            tmp += G[ii,jj]
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

def shanon_entropy(labels):
    """ Computes Shanon entropy using the labels distribution """
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
        ss += prob  * (log(prob,2))**2
        sq += (prob * log(prob,2))**2
    s2 = ((ss - sq) / float(n_labels)) - ((n_classes - 1) / float(2 *
        (n_labels) **2))
    return ent, s2, n_labels, n_classes # count is an array you might want to
# return n_classes isnted
#-------------------------------------------------------------------------------------------------
#                                           Settings
#-------------------------------------------------------------------------------------------------
def data_analysis(subjects_id, rand_ind, n_cluster, pairwise=True,
        sliding_window=True, graph_analysis=True, window_size=20,
        n_time_points=184, network_comp='between_network'):
    ''' Compute the main analysis. This function calculates the synchrony,
    metastability and perform the graph analysis.

    Inputs:
        - subjects_id: A list of subjects_id
        - rand_ind:    Randomisation index -- necessary for generating random
                       matrix
        - n_cluster:   Number of clusters used for k-means
        - pairwise:    Binary input. Is a pairwise comparison going to be
                       performed?
        - sliding_window: Sliding window used to reduce noise of the time serie
        - graph_analysis: Defines if graph_analysis will be performed or not
        - window_size: Defined size of the sliding window
        - n_time_points: number ot time points of the data set
        - n_regions:   Define number of regions used in the data set
        - network_comp: Define type of coparision that will be carried out.
                        between_network = compare BOLD between network
                        within_network = compare BOLD within network
                        full_network = compare BOLD from all regions used in the
                        segmentation
    '''

    hilbert_t_points = n_time_points - window_size
    base_path = os.path.join(os.sep, 'scratch' ,'jdafflon', 'personal', 'data_out')
    in_dir = os.path.join(base_path, 'preprocessing_out', 'extract_vois')
    out_dir = os.path.join(base_path, 'data_analysis' )

    # Initialise matrices for average comparision between regions and index
    # counter
    all_phi = np.zeros((len(subjects_id), 3))
    phi =  np.zeros((hilbert_t_points), dtype=complex)
    idx = 0

    # Iterate over the list of subjects and calculate the pairwise and globabl
    # metastability and synchrony.
    for subject_id in subjects_id:
        #check if graph theory metrics for this subject exists already in case
        # yes, go to next subject
        if graph_analysis:
            print os.path.join(out_dir, 'pairwise_comparison', network_comp,
                'rand_ind_%02d' %rand_ind, '%s' %subject_id, '%02d_clusters'
                %n_cluster, 'graph_measures_labels_shannon_%s.pickle'
                %subject_id)
            if os.path.isfile(os.path.join(out_dir, 'pairwise_comparison', network_comp,
                'rand_ind_%02d' %rand_ind, '%s' %subject_id, '%02d_clusters'
                %n_cluster, 'graph_measures_labels_shannon_%s.pickle' %subject_id)):
                print('Subject %s with randomisation index %02d and cluster %02d already exists' %(subject_id, rand_ind, n_cluster))
                continue
        print(' Calculating graph measures for subject: %s' %subject_id)
        # Check if output folder exists, if it does not then create it
        if not os.path.isdir(os.path.join(out_dir, 'pairwise_comparison', network_comp,
            'rand_ind_%02d' %rand_ind, '%s' %subject_id, 'metastability')):
            os.makedirs(os.path.join(out_dir, 'pairwise_comparison', network_comp,
                'rand_ind_%02d' %rand_ind, '%s' %subject_id, 'metastability'))

        if not os.path.isdir(os.path.join(out_dir, 'pairwise_comparison', network_comp,
            'rand_ind_%02d' %rand_ind, '%s' %subject_id, 'synchrony')):
            os.makedirs(os.path.join(out_dir, 'pairwise_comparison', network_comp,
                'rand_ind_%02d' %rand_ind,'%s' %subject_id, 'synchrony'))

        if not os.path.isdir(os.path.join(out_dir, 'pairwise_comparison', network_comp,
            'rand_ind_%02d' %rand_ind, '%s' %subject_id, 'threshold_matrix')):
            os.makedirs(os.path.join(out_dir, 'pairwise_comparison', network_comp,
                'rand_ind_%02d' %rand_ind, '%s' %subject_id, 'threshold_matrix'))

        if not os.path.isdir(os.path.join(out_dir, 'pairwise_comparison', network_comp,
            'rand_ind_%02d' %rand_ind, '%s' %subject_id, '%02d_clusters' %n_cluster)):
            os.makedirs(os.path.join(out_dir, 'pairwise_comparison', network_comp,
                'rand_ind_%02d' %rand_ind, '%s' %subject_id, '%02d_clusters' %n_cluster))
        # Import BOLD signal for each VOI
        if network_comp == 'between_network':
            data_path = os.path.join(in_dir, subject_id, 'fwhm_5', 'between_network.txt')
            data = np.genfromtxt(data_path)
        else:
            data_path = os.path.join(in_dir, subject_id, 'fwhm_5', '%s.txt'
                    %subject_id)
            data = np.genfromtxt(data_path)
        # The first number corresponds to the number of regions in the dataset
        # used for the segmentation
        n_regions = data.shape[0]

        # Perform hilbert transform
        hiltrans = hilbert_tranform(data)
        if sliding_window:
            hiltrans_sliding_window = np.zeros((n_regions, (hiltrans.shape[1] -
                window_size + 1)), dtype=complex)
            for roi in range(n_regions):
                hiltrans_sliding_window[roi, :] = slice_window_avg(hiltrans[roi, :], window_size)

        # Calculate states on raw BOLD data
        z_data = np.zeros((data.shape))
        thr_data = np.zeros((data.shape))
        for VOI in range(n_regions):
            voi_mean = np.mean(data[VOI,:])
            voi_std = np.std(data[VOI,:])
            for t in range(data.shape[1]):
                z_data[VOI, t] = abs(float((data[VOI, t] - voi_mean)) / voi_std)
                # Threshold BOLD at 1.3
                if z_data[VOI, t] > 1.3:
                    thr_data[VOI, t] = 1
        # Save thresholded image of BOLD
        fig = plt.figure()
        plt.imshow(thr_data, interpolation='nearest')
        fig.savefig(os.path.join(out_dir, 'pairwise_comparison', network_comp,
            'rand_ind_%02d' %rand_ind, '%s' %subject_id,
            'threshold_matrix', '%s_BOLD.png' %(subject_id)))
        plt.clf()
        plt.close()

        bold_shanon_entropy = {}
        # Perfom k-means on the BOLD signal
        kmeans_bold = KMeans(n_clusters=n_cluster)
        kmeans_bold.fit_transform(np.transpose(thr_data))
        kmeans_bold_labels= kmeans_bold.labels_
        # Calculate Shannon Entropy
        bold_shanon_entropy['bold_h'],  bold_shanon_entropy['s2'],       \
        bold_shanon_entropy['n_labels_bold'], \
        bold_shanon_entropy['n_classes_bold'] = shanon_entropy(kmeans_bold_labels)
        pickle.dump(bold_shanon_entropy, open(os.path.join(out_dir,
            'pairwise_comparison', network_comp,  'rand_ind_%02d' %rand_ind,
            '%s' %subject_id,'%02d_clusters' %n_cluster,
            'bold_shannon_%s.pickle' %(subject_id)), 'wb'))

        # Calculate data synchrony following Hellyer-2015_Cognitive
        if pairwise:
            # Find length of time points after Hilbert Transform and/or sliding
            # window
            if sliding_window:
                hilbert_t_points = hiltrans_sliding_window.shape[1]
                # overwrite hiltrans with the data obtained with the sliding window
                hiltrans = hiltrans_sliding_window
            else:
                hilbert_t_points = hiltrans.shape[1]

            # Find indices of regions for pairwise comparison. As the comperision is
            # symetric computation power can be saved by calculating only the lower
            # diagonal matrix.
            indices = np.tril_indices(n_regions)
            # reshuffle index and obtain tuple for each pairwise comparision
            indices = zip(indices[0], indices[1])

            # Calculate phi, metastability, synchrony and mean synchrony  for the specified indices
            phi = calculate_phi(indices, n_regions, hilbert_t_points, hiltrans)
            metastability = calculate_metastability(indices, n_regions, phi)
            # save values for metastability
            pickle.dump(metastability, open(os.path.join(out_dir,
                'pairwise_comparison',  network_comp, 'rand_ind_%02d' %rand_ind, '%s'
                %subject_id, 'metastability', 'mean_metastability.pickle'),
                'wb'))

            synchrony, mean_synchrony = calculate_synchronies(indices, n_regions,
                    hilbert_t_points, phi)
            pickle.dump(synchrony, open(os.path.join(out_dir,
                'pairwise_comparison',  network_comp, 'rand_ind_%02d' %rand_ind, '%s'
                %subject_id, 'synchrony', 'synchrony.pickle'), 'wb'))

            # plot synchrony matrix for each time point
            # TODO: to speed up performace a bit you could implement this method:
            # http://stackoverflow.com/questions/16334588/create-a-figure-that-is-reference-counted/16337909#16337909
            # fig = plt.figure()
            # for t in range(hilbert_t_points):
            #     plt.imshow(synchrony[:, :, t], interpolation='nearest')
            #     print out_dir
            #     fig.savefig(os.path.join(out_dir, 'pairwise_comparison', network_comp,
            #         'rand_ind_%02d' %rand_ind, '%s' %subject_id, 'synchrony',
            #         '%s_%03d.png' %(subject_id, t)))
            #     plt.clf()
            # plt.close()

            # ---------------------------------------------------------------------
            # Graph Theory Measurements
            # ---------------------------------------------------------------------
            k_optima = calculate_optimal_k(mean_synchrony, indices)
            print ('Optimal mean threshold: %3f' %k_optima)

            # Threshold the synchrony matrix at each time point using the just found optimal
            # threshold and save the output
            synchrony_bin = np.zeros((n_regions, n_regions, hilbert_t_points))
            # fig = plt.figure()
            for t in range(hilbert_t_points):
                for index in indices:
                    if synchrony[index[0], index[1], t] >= k_optima:
                        synchrony_bin[index[0], index[1], t] = 1

                synchrony_bin[:,:,t] = mirror_array(synchrony_bin[:,:,t])
                # plt.imshow(synchrony_bin[:,:,t], interpolation='nearest')
                # fig.savefig(os.path.join(out_dir, 'pairwise_comparison', network_comp,
                #     'rand_ind_%02d' %rand_ind, '%s' %subject_id,
                #     'threshold_matrix', '%s_%03d.png' %(subject_id, t)))
                # plt.clf()
            # plt.close()
            # pickle.dump(synchrony_bin, open(os.path.join(out_dir,
            #     'pairwise_comparison',  network_comp, 'rand_ind_%02d' %rand_ind, '%s'
            #     %subject_id, 'metastability', 'synchrony_bin.pickle'),
            #     'wb'))

            if graph_analysis == True:
                print('Calculating Graph Theory Measurements')
                # Degree centrality:
                #-------------------
                degree_centrality = np.transpose(degrees_und(synchrony_bin))

                weight = np.zeros(( hilbert_t_points, n_regions))
                w = np.multiply(synchrony, synchrony_bin)
                # Initialise flatten array so that you have time by regions (140 x
                # 6724), this will strucutre is necessary in order to perform
                # K-means
                # Ds_flat = np.zeros((hilbert_t_points, (synchrony_bin.shape[0])**2))
                Ds_flat = {}
                SM = {}
                Ds = {}
                # SM = np.zeros((hilbert_t_points, n_regions))
                # Ds = np.zeros((n_regions, n_regions, hilbert_t_points))
                if network_comp == 'between_network':
                    network = range(n_regions)
                    network_list = {}
                    for t in range(hilbert_t_points):
                        network_list[t] = network

                # # check where the network has more then one component
                # for t in range(hilbert_t_points):
                #     n_components = clustering.number_of_components(synchrony_bin[:,:,t])
                #     if len(np.where(clustering.get_components(synchrony_bin[:,:,t])[0]>1)[0])>1:
                #         print t
                # Iterate over time to obtain different complex network measurements.
                for t in range(hilbert_t_points):

                    # Weight
                    #-------------------
                    # Use the thresholded matrix to calcualte the average of weight over all regions
                    for roi in range(n_regions):
                        weight[t, roi] = np.average(w[:, roi, t])

                    # Transitivity:
                    #-------------------
                    transitivity = transitivity_bu(synchrony_bin[:, :, t])

                    # Small-worldness
                    #------------------
                    # Every time this function is called a new random network is
                    # generated
                    print(t)
                    n_components = clustering.number_of_components(synchrony_bin[:,:,t])
                    if n_components > 1:
                        components = dict([(key, []) for key in range(n_components)])
                        # Get all components and transform numpy array into a python list
                        # list_components = clustering.get_components(synchrony_bin[:,:,t])[0].tolist()
                        list_components = clustering.get_components(synchrony_bin[:,:,t])[0]
                        # Check if all components are composed of more then one region.
                        # If so, divide the current network into the corresponding
                        # components, otherwise eliminate the lonely component
                        # get_components()[0]:  ensure that only the vector of
                        # component assignments for each node is returned
                        # if len(np.where(clustering.get_components(synchrony_bin[:,:,t])[0]>1)[0]) == 1:
                        for component in range(1, n_components + 1):
                            if np.bincount(list_components)[component] == 1:
                                # transform list into np.array to use np.where
                                # list_components = np.array(list_components)
                                index_to_eliminate = np.where(list_components == component)[0][0]
                                # Eliminate first the specified row and then the
                                # specified column from the thresholded synchrony matrix
                                tmp = np.delete(synchrony_bin[:,:,t],
                                        index_to_eliminate, 0)
                                tmp2 = np.delete(tmp, index_to_eliminate, 1)
                                # eliminate the specific network form the network list.
                                network_list[t] = np.delete(network_list[t],
                                        index_to_eliminate, 0)
                                print('Node #%d was eliminated at timepoint %d') %(index_to_eliminate, t)
                                SM[str(t)], Ds[str(t)] = estimate_small_wordness(tmp2, rand_ind)
                                # Flatten the synchrony matrix and path_distance so that it can be given as
                                # argument for the K-means
                                Ds_flat[str(t)] = np.ndarray.flatten(Ds[str(t)])

                            elif 2 <= np.bincount(list_components)[component] < n_regions - 1:
                                # check if there is more then one component
                                print('More then one component found at timepoint %d') %(t)
                                # As all components start from one, iteration should
                                # start from 1 and not 0.
                                # find index of the elements belonging to this
                                # component
                                all_indices = range(n_regions)
                                # find indices for each component
                                indices = np.where(list_components == component)[0]
                                # obtain indices for elements that will be
                                # discarted
                                indices_2_eliminate = np.delete(all_indices, indices, 0)
                                tmp  = np.delete(synchrony_bin[:,:,t], indices_2_eliminate, 1)
                                # final matrix where the binary synchrony values
                                # are saved
                                components[component] = np.delete(tmp, indices_2_eliminate, 0)
                                # Estimate small-wordness for each component
                                element = ''.join((str(t), '_', str(component)))
                                SM[element], Ds[element] = estimate_small_wordness(components[component], rand_ind)
                                # Flatten the synchrony matrix and path_distance so that it can be given as
                                # argument for the K-means
                                Ds_flat[element] = np.ndarray.flatten(Ds[element])
                            else:
                                continue
                    else:
                        SM[str(t)], Ds[str(t)] = estimate_small_wordness(synchrony_bin[:,:,t], rand_ind)
                        # Flatten the synchrony matrix and path_distance so that it can be given as
                        # argument for the K-means
                        Ds_flat[str(t)] = np.ndarray.flatten(Ds[str(t)])

                # Save picke for with graph measurements for each subject
                graph_measures = {'weight': weight,
                                  'small_wordness': SM,
                                  'degree_centrality': degree_centrality,
                                  'path_distance': Ds_flat
                                  }
                pickle.dump(graph_measures, open(os.path.join(out_dir,
                    'pairwise_comparison',  network_comp, 'rand_ind_%02d' %rand_ind,
                    '%s' %subject_id,'%02d_clusters' %n_cluster,
                    'graph_measures_%s.pickle' %(subject_id)), 'wb'))

                # ---------------------------------------------------------------------
                # Clustering
                # ---------------------------------------------------------------------
                # Perform K-means and calculate Shannon Entropy for each graph theory
                # measurement
                #TODO: think how you want to transform SM and DM into one single
                # matrix
                kmeans = KMeans(n_clusters=n_cluster)
                graph_measures_labels = {}
                for key in graph_measures:
                    pdb.set_trace()
                    kmeans.fit_transform(graph_measures[key])
                    graph_measures_labels[key] = kmeans.labels_
                    graph_measures_labels[key + '_h'], graph_measures_labels[key + 's2'], \
                    graph_measures_labels['n_labels_gm'],                                 \
                    graph_measures_labels['n_classes_gm']  = shanon_entropy(graph_measures_labels[key])

                pickle.dump(graph_measures_labels, open(os.path.join(out_dir,
                    'pairwise_comparison',  network_comp, 'rand_ind_%02d' %rand_ind,
                    '%s' %subject_id, '%02d_clusters' %n_cluster,
                    'graph_measures_labels_shannon_%s.pickle' %(subject_id)), 'wb'))

            # ---------------------------------------------------------------------
            # Clustering
            # ---------------------------------------------------------------------
            # Calculate the K-means clusters
            if graph_analysis == False:
                synchrony_bin_flat = np.zeros((hilbert_t_points, (synchrony_bin.shape[0])**2))
                total_entropy = {}
                cluster_centroids = {}
                for t in range(hilbert_t_points):
                    synchrony_bin_flat[t, :] = np.ndarray.flatten(synchrony_bin[:, :, t])
                kmeans = KMeans(n_clusters=n_cluster)
                kmeans.fit_transform(synchrony_bin_flat)
                kmeans_labels = kmeans.labels_
                centroids = kmeans.cluster_centers_
                cluster_centroids['centroids'] = centroids
                total_entropy['synchrony_h'], total_entropy['s2'], \
                total_entropy['n_labels_syn'],                     \
                total_entropy['n_classes_syn'] = shanon_entropy(kmeans_labels)
                pickle.dump(cluster_centroids, open(os.path.join(out_dir,
                    'pairwise_comparison',  network_comp, 'rand_ind_%02d' %rand_ind,
                    '%s' %subject_id, '%02d_clusters' %n_cluster,
                    '%s_cluster_centroid.pickle' %(subject_id)), 'wb'))
                pickle.dump(total_entropy, open(os.path.join(out_dir,
                    'pairwise_comparison',  network_comp, 'rand_ind_%02d' %rand_ind,
                    '%s' %subject_id, '%02d_clusters' %n_cluster,
                    '%s_total_shannon_entropy.pickle' %(subject_id)), 'wb'))

                # # save plot of centroids
                # n_centroids = centroids.shape[0]
                # for ncentroid in range(n_centroids):
                #     state = numpy.zeros((n_regions,n_regions))
                #     n_centroid = centroids[ncentroid]
                #     for ii in range(82):
                #         states[ii,:] = n_centroid[82*ii:82*(ii+1)]
                #         plt.imshow(states, interpolation='nearest')
                #         fig.savefig('state_%s.png' %ncentroid)
                #         plt.clf()

            print('Done!')
            print ('--------------------------------------------------------------')
        else:
            # iterate over all analysed regions
            for voi in range(n_regions):
                phi += np.exp(np.angle(hiltrans[voi, :]) * 1j)
            # normalise over the number of regions
            phi *= float(1)/n_regions
            pickle.dump(phi, open(os.path.join(out_dir,
                'pairwise_comparison',  network_comp, 'rand_ind_%02d' %rand_ind,
                '%s' %subject_id,'%02d_clusters' %n_cluster,
                'synchrony_data_%s.pickle' %(subject_id)), 'wb'))

            pdb.set_trace()

            # Reshuffle data so that it can be saved
            all_phi[idx, 0] = int(subject_id.strip('sub'))
            all_phi[idx, 1] = np.mean(abs(phi))
            all_phi[idx, 2] = np.std(abs(phi))
            idx += 1

            np.savetxt(os.path.join(out_dir, 'global_comparison', 'synchrony_and_metastability.txt'), all_phi, fmt='%3d %1.4f %1.4f', delimiter=' ', header='subject_id, mean synchrony, metastability')

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--subjects_id', required=True, nargs='*', type=str)
#     parser.add_argument('--rand_ind', required=True, type=int)
#     parser.add_argument('--n_cluster', required=True, type=int)
#     args = parser.parse_args()

#     # parse arguments to pass to function
#     subjects_id = args.subjects_id
#     rand_ind =  args.rand_ind
#     n_cluster = args.n_cluster

#     data_analysis(subjects_id, rand_ind, n_cluster)

