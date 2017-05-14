#!/bin/env python

################################################################################
# Global imports
################################################################################
import os
import sys
import time

################################################################################
# Parameters
################################################################################

network_types = ['between_network', 'within_network', 'full_network']
window_types = ['non-sliding', 'sliding']
data_analysis_types = ['BOLD', 'synchrony', 'graph_analysis']
group_analysis_types = ['hutchenson', 'ttest', '1ANOVA']
analysis_types = ['rest', 'task']

############################################################################
# Argument parsing
############################################################################
# This needs to be first, because even path settings depend on the
# parameters passed to the command line.
from argparse import ArgumentParser
parser = ArgumentParser(
    description='Analyse the subjects.'
)
# Analysis type
parser.add_argument(
    '--analysis-type', required=True,
    dest='analysis_type', metavar='ANALYSIS_TYPE',
    choices=analysis_types,
    help='Analysis type. Choose from: ' + ', '.join(analysis_types)
)
parser.add_argument(
    '-c' '--golden-subjects',
    action='store_true', dest='golden_subjects',
    help='Perform analysis with subset of healthy subjects'
)
# Number of subjects
parser.add_argument(
    '-n', '--nsubjects',
    type=int, dest='nsubjects', metavar='NSUBJECTS', default=None,
    help='Number of healthy and schizophrenic subjects.'
)
# Possible activities/phases
parser.add_argument(
    '-p', '--preprocess',
    action='store_true', dest='preprocess',
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
# Options to pass to phases
# Note: Not all options apply to all phases.
parser.add_argument(
    '--network-type',
    dest='network_type', metavar='NETWORK_TYPE',
    choices=network_types,
    help='Network type. Choose from: ' + ', '.join(network_types)
)
parser.add_argument(
    '--window-type',
    dest='window_type', metavar='WINDOW_TYPE',
    choices=window_types,
    help='Window type. Choose from: ' + ', '.join(window_types)
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
args = parser.parse_args()

################################################################################
# Path settings
################################################################################
# Base path for all input and output data.
base_path = os.path.join(os.path.sep, 'group', 'scz_dynamics', 'ucla-la5')
base_path_in = os.path.join(base_path, 'data_in')
base_path_out = os.path.join(base_path, 'data_out', args.analysis_type)

# Pre-processing
# This folder contains all subjects' folders for the pre-processing phase.
# FIXME: This has never been tested.
preprocessing_input_basepath = os.path.join(base_path_in, 'reconall_data')
preprocessing_output_basepath = os.path.join(base_path_out, 'preprocessing_out')

# ROI extraction
# Input image for ROI extraction
roi_input_segmented_image_filename = os.path.join(base_path_in, 'voi_extraction', 'seg_aparc_82roi_2mm.nii.gz')
# Input region list for ROI extraction
roi_input_segmented_regions_filename = os.path.join(base_path_in, 'voi_extraction', 'LookupTable')
# Image where between_network and within_network are specified.
roi_input_network_filename = os.path.join(base_path_in, 'voi_extraction', 'PNAS_Smith09_rsn10.nii')
roi_input_basepath = preprocessing_output_basepath
roi_output_basepath = os.path.join(base_path_out, 'extract_roi')

# Data analysis
data_analysis_input_basepath = roi_output_basepath
data_analysis_output_basepath = os.path.join(base_path_out, 'data_analysis')

# Group data analysis
group_analysis_input_basepath = data_analysis_output_basepath
group_analysis_output_basepath = os.path.join(base_path_out, 'group_analysis')

# Subjects
# FIXME: Move to data_in folder.
subjects_filename = 'subjects.json'

################################################################################
# Global logging
################################################################################
# Note: This needs to be setup before other local modules are imported and
#       and before any local code is executed.
import logging
timestamp = time.strftime("%Y%m%d%H%M%S")
log_filename = os.path.join(base_path_out, '%s_ucla5.log' % timestamp)
formatter = logging.Formatter('%(message)s')
log = logging.getLogger('')
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_filename)
fh.setFormatter(formatter)
log.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
log.addHandler(ch)

# logging.basicConfig(filename=log_filename,
#                     level=logging.INFO,
#                     format='%(message)s',
#                     stream=sys.stdout)

################################################################################
# Local imports
################################################################################
# FIXME: Move extract_roi to extract_roi.py
from subjects import load_subjects
from preprocessing_workflow import preprocess_data
from extract_roi import extract_roi
from data_analysis import data_analysis
from group_analysis_pairwise import group_analysis_pairwise

################################################################################
# Load subjects.
################################################################################
# They must always be loaded, no matter the type of the analysis.
# Note: If the user didn't specify nsubjects, we take all subjects (still
#       balanced).
subjects = load_subjects(subjects_filename, args.golden_subjects, args.nsubjects)

############################################################################
# Pre-processing
############################################################################
if args.preprocess:
    # Note: Preprocessing is only running on the old cluster
    print('Preprocessing')
    preprocess_data(subjects,
                    preprocessing_input_basepath,
                    preprocessing_output_basepath)
    print('Pre-processing.')

############################################################################
# ROI extraction
############################################################################
if args.extract_roi:
    if args.network_type is None:
        parser.error('You must specify: --network_type.')

    # Extract ROIs.
    extract_roi(subjects,
                args.network_type,
                roi_input_basepath,
                roi_input_segmented_image_filename,
                roi_input_segmented_regions_filename,
                roi_output_basepath,
                args.golden_subjects,
                network_mask_filename=roi_input_network_filename)

############################################################################
# Data analysis
############################################################################
if args.analyse_data:
    if args.network_type is None or \
       args.window_type is None or \
       args.data_analysis_type is None or \
       args.nclusters is None or \
       args.rand_ind is None:
        parser.error('You must specify: ' + \
                     '--network-type, ' + \
                     '--window-type, ' + \
                     '--data-analysis-type, ' + \
                     '--nclusters, ' + \
                     '--rand-ind.')

    # Analyse data.
    data_analysis(subjects,
                  data_analysis_input_basepath,
                  data_analysis_output_basepath,
                  args.network_type,
                  args.window_type,
                  args.data_analysis_type,
                  args.nclusters,
                  args.rand_ind,
                  args.golden_subjects)


############################################################################
# Group analysis
############################################################################
if args.analyse_data_group:
    if args.network_type is None or \
       args.window_type is None or \
       args.data_analysis_type is None or \
       args.group_analysis_type is None or \
       args.nclusters is None or \
       args.rand_ind is None:
        parser.error('You must specify: ' + \
                     '--network-type, ' + \
                     '--window-type, ' + \
                     '--data-analysis-type, ' + \
                     '--group-analysis-type, ' + \
                     '--nclusters, ' + \
                     '--rand-ind.')

    group_analysis_pairwise(subjects,
                            group_analysis_input_basepath,
                            group_analysis_output_basepath,
                            args.network_type,
                            args.window_type,
                            args.data_analysis_type,
                            args.group_analysis_type,
                            args.nclusters,
                            args.rand_ind)

# remember to close the handlers
for handler in log.handlers:
    handler.close()
    log.removeFilter(handler)