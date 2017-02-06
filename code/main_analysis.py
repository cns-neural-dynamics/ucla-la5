#!/share/apps/anaconda/bin/python
import nibabel as nib
import numpy as np
import os
from data_analysis import *
import subprocess
import threading
import shlex
from preprocessing_workflow import preprocessing_pipeline
# from data_analysis import data_analysis
# from group_analysis_pairwise import group_analysis_pairwise

#------------------------------------------------------------------------------
# Define which analysis to perform
do_preprocessing = True
do_extract_roi   = False
do_data_analysis = False

# Define if analysis will be carried out by comparing different regions inside
# of a predefined network  (within_network) or among different networks
# (between_networks - you will end up with a 10x10 matrix)
network_comp = 'between_network'
# network_comp = 'within_network'
# network_comp = 'full_network'
fwhm = 'fwhm_5'
subject_list = [['sub-10429', 'sub-10438', 'sub-10440']]
# sub_list = ['sub-10171', 'sub-10189', 'sub-10193', 'sub-10206',
#         'sub-10217', 'sub-10225', 'sub-10227', 'sub-10228', 'sub-10235',
#         'sub-10249', 'sub-10269', 'sub-10271', 'sub-10273', 'sub-10274',
#         'sub-10280', 'sub-10290', 'sub-10292', 'sub-10299', 'sub-10304',
#         'sub-10316', 'sub-10321', 'sub-10325', 'sub-10329', 'sub-10339',
#         'sub-10340', 'sub-10345', 'sub-10347', 'sub-10356', 'sub-10361',
#         'sub-10365', 'sub-10376', 'sub-10377', 'sub-10388', 'sub-10429',
#         'sub-10438', 'sub-10440']
# sub_list = ['sub-10448',
#         'sub-10455', 'sub-10460', 'sub-10471', 'sub-10478', 'sub-10487',
#         'sub-10492', 'sub-10501', 'sub-10506', 'sub-10517', 'sub-10523',
#         'sub-10524', 'sub-10525', 'sub-10527', 'sub-10530', 'sub-10557',
#         'sub-10565', 'sub-10570', 'sub-10575', 'sub-10624', 'sub-10629',
#         'sub-10631', 'sub-10638', 'sub-10668', 'sub-10672', 'sub-10674',
#         'sub-10678', 'sub-10680', 'sub-10686', 'sub-10692', 'sub-10696',
#         'sub-10697', 'sub-10704', 'sub-10707', 'sub-10708', 'sub-10719',
#         'sub-10724', 'sub-10746', 'sub-10762', 'sub-10779', 'sub-10785',
#         'sub-10788', 'sub-10844', 'sub-10855', 'sub-10871', 'sub-10877',
#         'sub-10882', 'sub-10891', 'sub-10893', 'sub-10912', 'sub-10934',
#         'sub-10940', 'sub-10948', 'sub-10949', 'sub-10958', 'sub-10963',
#         'sub-10968', 'sub-10975', 'sub-10977', 'sub-10987',
#         'sub-10998', 'sub-11019', 'sub-11030', 'sub-11044', 'sub-11050',
#         'sub-11052', 'sub-11059', 'sub-11061', 'sub-11062', 'sub-11066',
#         'sub-11067', 'sub-11068', 'sub-11077', 'sub-11082', 'sub-11088',
#         'sub-11090', 'sub-11097', 'sub-11098', 'sub-11104', 'sub-11105',
#         'sub-11106', 'sub-11108', 'sub-11112', 'sub-11122',
#         'sub-11128', 'sub-11131', 'sub-11142', 'sub-11143', 'sub-11149',
#         'sub-11156']
# sub_list = ['sub-50004', 'sub-50005', 'sub-50006', 'sub-50007', 'sub-50008',
#         'sub-50010', 'sub-50013', 'sub-50014', 'sub-50015', 'sub-50016',
#         'sub-50020', 'sub-50021', 'sub-50022', 'sub-50023', 'sub-50025',
#         'sub-50027', 'sub-50029', 'sub-50032', 'sub-50033', 'sub-50034',
#         'sub-50035', 'sub-50036', 'sub-50038', 'sub-50043', 'sub-50047',
#         'sub-50048', 'sub-50049', 'sub-50050', 'sub-50051', 'sub-50052',
#         'sub-50053', 'sub-50054', 'sub-50055', 'sub-50056', 'sub-50058',
#         'sub-50059', 'sub-50060', 'sub-50061', 'sub-50064', 'sub-50066',
#         'sub-50067', 'sub-50069', 'sub-50073', 'sub-50075', 'sub-50076',
#         'sub-50077', 'sub-50080', 'sub-50081', 'sub-50083', 'sub-50085']
# sub_list = ['sub-50043', 'sub-50047', 'sub-50048', 'sub-50049', 'sub-50050',
#         'sub-50051', 'sub-50052', 'sub-50053', 'sub-50054', 'sub-50055',
#         'sub-50056', 'sub-50058', 'sub-50059']
base_path = os.path.join(os.path.sep, 'home', 'jdafflon', 'scratch', 'personal')

#------------------------------------------------------------------------------
#                           call pre-processing
#------------------------------------------------------------------------------
if do_preprocessing:
    for sub in subject_list:
        preprocessing_pipeline(sub)
    print('preprocessing %s' %sub)
    print('-----------------------------------------------------------------------')
#------------------------------------------------------------------------------
#                               extract ROIs
#------------------------------------------------------------------------------
if do_extract_roi:
    print('extracting ROIs')
    # File name of the final preprocessed image that will be segmented
    preprocessed_image = 'denoised_func_data_nonaggr_filt_regfilt.nii.gz'

    data_sink_path = os.path.join(base_path, 'data_out', 'preprocessing_out')
    output_path = os.path.join(base_path, 'data_out', 'preprocessing_out', 'extract_vois')

    # Define path to the image that will be used as a segmentation mask and load
    segmented_image_path = os.path.join(base_path, 'data_in', 'voi_extraction', 'seg_aparc_82roi_2mm.nii.gz')
    segmented_regions_path = os.path.join(base_path, 'data_in', 'voi_extraction',
            'LookupTable')
    segmented_regions = np.genfromtxt(segmented_regions_path, dtype = [('numbers',
        '<i8'), ('regions', 'S31'), ('labels', 'i4')], delimiter=',')

    # Path to image where the different networks are specified
    network_path = os.path.join(base_path, 'data_in', 'voi_extraction',
                   'PNAS_Smith09_rsn10.nii')
    # get subjects_id from a list of lists in subject_list
    subjects_id = subject_list[0]
    # Extract ROIs
    extract_roi(subjects_id, fwhm, data_sink_path, preprocessed_image,
                segmented_image_path, segmented_regions, output_path, network=True,
                network_path=network_path, network_comp=network_comp)

#------------------------------------------------------------------------------
#                              data analysis
#------------------------------------------------------------------------------
if do_data_analysis:
    # get subjects_id from a list of lists in subject_list
    subjects_id = subject_list[0]

    print('performing data analysis')
    bold_path = os.path.join(base_path, 'data_out', 'data_analysis', 'pairwise_comparison')
    # perform data_analysis with different valus for the n_cluster
    n_clusters = [25] # 15, 20, 25, 30]
    rand_inds = [15] #, 45]     # randomisation index necessary for randmio_und. The higher
                         # this parameter the more random will the generated matrix be
    n_groups = 2
    analysis_type = 'synchrony'
    # analysis_type = 'BOLD'
    # analysis_type = 'graph_analysis'
    # statistics_type = 'hutchenson'
    statistics_type = 'ttest'
    # statistics_type = '1ANOVA'
    threads = []

    def thread_wrapper_function(rand_ind=None, n_cluster=None):
        print 'Running thread: ' + str(rand_ind) + ' ' + str(n_cluster)
        # data_analysis(subjects_id, rand_ind, n_cluster, graph_analysis=True)
        # when calling the between network comparision
        data_analysis(subjects_id, rand_ind, n_cluster, pairwise=True,
                      sliding_window=True, graph_analysis=True,
                      network_comp=network_comp)
        group_analysis_pairwise(subjects_id, n_cluster, n_groups, bold_path,
                rand_ind, analysis_type, statistics_type,
                network_comp=network_comp)

    for rand_ind in rand_inds:
        for n_cluster in n_clusters:
            t = threading.Thread(target=thread_wrapper_function,
                kwargs={
                    'rand_ind': rand_ind,
                    'n_cluster': n_cluster
                }
            )
            threads.append(t)
            t.start()


    # Wait for all threads to complete.
    for thread in threads:
        thread.join()
