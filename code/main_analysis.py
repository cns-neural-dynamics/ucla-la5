#!/share/apps/anaconda/bin/python

import os
import numpy as np
from argparse import ArgumentParser

# FIXME: Move to extract_roi.py
from data_analysis import extract_roi, data_analysis
from group_analysis_pairwise import group_analysis_pairwise
from subjects import load_subjects

################################################################################
# File paths
################################################################################

# Base path for all input and output data.
base_path = os.path.join(os.path.sep, 'home', 'jdafflon', 'scratch', 'personal')

# Subjects
# FIXME: Move to data_in folder.
subjects_filename = 'subjects.json'

# Pre-processing
# TODO: Add input paths.
# This folder contains all subjects' folders for the pre-processing phase.
preprocessing_output_basepath = os.path.join(base_path, 'data_out', 'ucla_la5', 'preprocessing_out')
# Each subject folder will have its own output file, with the same name.
preprocessing_output_filename = 'denoised_func_data_nonaggr_filt.nii.gz'

# ROI extraction
# Input image for ROI extraction
roi_input_segmented_image_filename = os.path.join(base_path, 'data_in', 'voi_extraction', 'seg_aparc_82roi_2mm.nii.gz')
# Input region list for ROI extraction
roi_input_segmented_regions_filename = os.path.join(base_path, 'data_in', 'voi_extraction', 'LookupTable')
# Image where between_network and within_network are specified.
roi_input_network_filename = os.path.join(base_path, 'data_in', 'voi_extraction', 'PNAS_Smith09_rsn10.nii')
# For each subject, this folder will contain a folder with the extracted ROI.
roi_output_basepath = os.path.join(preprocessing_output_basepath, 'extract_vois')

# Group data analysis
group_analysis_output_basepath = os.path.join(base_path, 'data_out', 'ucla_la5', 'data_analysis', 'pairwise_comparison')

################################################################################
# Parameters
################################################################################

roi_fwhm = 'fwhm_5'
network_types = ['between_network', 'within_network', 'full_network']
data_analysis_types = ['BOLD', 'synchrony', 'graph_analysis']
group_analysis_types = ['hutchenson', 'ttest', '1ANOVA']

# When called from the command line:
if __name__ == '__main__':

    ############################################################################
    # Argument parsing
    ############################################################################
    parser = ArgumentParser(
        description='Analyse the subjects.'
    )
    # Number of subjects.
    parser.add_argument(
        '-n', '--nsubjects',
        type=int, dest='nsubjects', metavar='NSUBJECTS', default=None,
        help='Number of healthy and schizophrenic subjects.'
    )
    # Possible activities/phases.
    parser.add_argument(
        '-p', '--preprocess',
        action='store_true', dest='preprocessing',
        help='Perform pre-processing of the data.'
    )
    parser.add_argument(
        '-r', '--extract_roi',
        action='store_true', dest='extract_roi',
        help='Perform ROI extraction.'
    )
    parser.add_argument(
        '-a', '--analyse-data',
        action='store_true', dest='analyse_data',
        help='Perform data analysis.'
    )
    parser.add_argument(
        '-g', '--analyse-data-group',
        action='store_true', dest='analyse_data_group',
        help='Perform group data analysis.'
    )
    # Options to pass to phases.
    # Note: Not all options apply to all phases.
    parser.add_argument(
        '--network-type',
        dest='network_type', metavar='NETWORK_TYPE',
        choices=network_types,
        help='Network type. Choose from: ' + ', '.join(network_types)
    )
    parser.add_argument(
        '--roi-network',
        type=int, dest='roi_network', metavar='ROI_NETWORK_BOOL',
        choices=[0,1],
        help='Use pre-defined network (1) or not (0) at ROI extraction.'
    )
    parser.add_argument(
        '--data-analysis-type',
        dest='data_analysis_type', metavar='DATA_ANALYSIS_TYPE',
        choices=data_analysis_types,
        help='Data analysis type. Choose from: ' + ', '.join(data_analysis_types)
    )
    parser.add_argument(
        '--nclusters',
        type=int, dest='nclusters', metavar='NCLUSTERS',
        help='Number of clusters to use in data analysis.'
    )
    parser.add_argument(
        '--rand-ind',
        type=int, dest='rand_ind', metavar='RAND_IND',
        help='Random index to use in data analysis (graph_analysis only).'
    )
    parser.add_argument(
        '--group-analysis-type',
        dest='group_analysis_type', metavar='GROUP_ANALYSIS_TYPE',
        choices=group_analysis_types,
        help='Group analysis type. Choose from: ' + ', '.join(group_analysis_types)
    )
    parser.add_argument(
        '--ngroups',
        type=int, dest='ngroups', metavar='NGROUPS',
        help='Number of groups to use in group data analysis.'
    )
    args = parser.parse_args()

    # Load subjects.
    # They must always be loaded, no matter the type of the analysis.
    # Note: If the user didn't specify nsubjects, we take all subjects (still
    #       balanced).
    subjects = load_subjects(subjects_filename, args.nsubjects)

    ############################################################################
    # Pre-processing
    ############################################################################
    if args.preprocess != False:
        # TODO: Implement pre-processing.
        # TODO: Pre-procesing wants subjects to be a list of list.
        print('Pre-processing.')
        pass

    ############################################################################
    # ROI extraction
    ############################################################################
    if args.extract_roi:
        if args.network_type is None or \
           args.roi_network is None:
            parser.error('You must specify --network_type and --roi-network.')

        # Extract ROIs.
        print('Extract ROI. Type: %s. Network: %s' % (args.network_type,
                                                      bool(args.roi_network)))
        extract_roi(subjects,
                    roi_fwhm,
                    preprocessing_output_basepath,
                    preprocessing_output_filename,
                    roi_input_segmented_image_filename,
                    roi_input_segmented_regions_filename,
                    roi_output_basepath,
                    network=bool(args.roi_network),
                    network_path=roi_input_network_filename,
                    network_comp=args.network_type)

    ############################################################################
    # Data analysis
    ############################################################################
    if args.analyse_data:
        if args.data_analyse_type is None or \
           args.network_type is None or \
           args.nclusters is None or \
           args.rand_ind is None:
            parser.error('You must specify: ' + \
                         '--data-analysis-type, ' + \
                         '--network_type, ' + \
                         '--nclusters, ' + \
                         '--rand-ind.')

        # Analyse data.
        print('Data analysis. Type: %s. Clusters: %d. Rand index: %d' %
              (args.data_analysis_type, args.nclusters, args.rand_ind))
        data_analysis(subjects,
                      args.rand_ind,
                      args.nclusters,
                      args.data_analysis_type,
                      pairwise=True, # TODO: Parametrize
                      sliding_window=True, # TODO: Parametrize
                      graph_analysis=False, # TODO: Parametrize
                      network_comp=args.network_type,
                      n_network=9
        )

    ############################################################################
    # Group analysis
    ############################################################################
    if args.analyse_data_group:
        if args.group_analysis_type is None or \
           args.network_type is None or \
           args.nclusters is None or \
           args.rand_ind is None or \
           args.ngroups is None:
            parser.error('You must specify: ' + \
                         '--group-analysis-type, ' + \
                         '--network_type, ' + \
                         '--nclusters, ' + \
                         '--rand-ind' + \
                         '--ngroups.')

        print('Group analysis. Type: %s.' % (args.group_analysis_type))
        group_analysis_pairwise(
            subjects,
            args.nclusters,
            args.ngroups,
            group_analysis_output_basepath,
            args.rand_ind,
            args.data_analysis_type,
            args.group_analysis_type,
            network_comp=args.network_type
        )
