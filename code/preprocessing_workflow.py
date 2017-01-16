#!/share/apps/anaconda/bin/python
import os
import sys
from nipype.interfaces.fsl import Info, FSLCommand, MeanImage
from nipype.interfaces.freesurfer import BBRegister, MRIConvert
from nipype.interfaces.ants import Registration, ApplyTransforms
from nipype.interfaces.c3 import C3dAffineTool
from nipype.interfaces.utility import Merge
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.io import SelectFiles, DataSink, FreeSurferSource, DataGrabber
from nipype.interfaces.utility import IdentityInterface, Function
from nipypext import nipype_wrapper
import argparse

#------------------------------------------------------------------------------
#                              Specify Variabless
#------------------------------------------------------------------------------
# all outputs will ge generated in compressed nifti format
FSLCommand.set_default_output_type('NIFTI_GZ')
# location of template file
template = Info.standard_image('MNI152_T1_2mm_brain.nii.gz')

subjects_list=['sub-10159']
# subjects_list = ['sub-10159', 'sub-10171', 'sub-10189', 'sub-10193', 'sub-10206',
#         'sub-10217', 'sub-10225', 'sub-10227', 'sub-10228', 'sub-10235',
#         'sub-10249', 'sub-10269', 'sub-10271', 'sub-10273', 'sub-10274',
#         'sub-10280', 'sub-10290', 'sub-10292', 'sub-10299', 'sub-10304',
#         'sub-10316', 'sub-10321', 'sub-10325', 'sub-10329', 'sub-10339',
#         'sub-10340', 'sub-10345', 'sub-10347', 'sub-10356', 'sub-10361',
#         'sub-10365', 'sub-10376', 'sub-10377', 'sub-10388']

# Data Location
base_path = os.path.join(os.sep, 'home', 'jdafflon', 'scratch', 'personal')
data_in_dir  = os.path.join(base_path, 'data_in', 'ucla_la5', 'ds000030')
data_out_dir = os.path.join(base_path, 'data_out', 'ucla_la5')


# Get functional image
datasource = Node(interface=DataGrabber(infields=['subject_id'],
        outfields=['epi', 't1']), name='datasource')
datasource.inputs.base_directory = data_in_dir
datasource.inputs.template = '*'
datasource.inputs.sort_filelist = True
datasource.inputs.field_template = dict(epi=os.path.join('%s', 'func', '%s_task-stopsignal_bold.nii.gz'),
                                        t1=os.path.join('%s', 'anat', '%s_T1w.nii.gz' ))
datasource.inputs.template_args = dict(epi=[['subject_id', 'subject_id']],
                                       t1=[['subject_id', 'subject_id']])
datasource.inputs.subject_id = subjects_list
#------------------------------------------------------------------------------
# Use data grabber specific for FreeSurfer data
fslsource = Node(FreeSurferSource(), name='getFslData')
fslsource.inputs.subjects_dir = data_in_dir

# Generate mean image - only for the EPI image
mean_image = Node(MeanImage(), name = 'Mean_Image')
# mean_image.inputs.out_file = 'MeanImage.nii.gz'

# convert FreeSurfer's MGZ format to nii.gz format
mgz2nii = Node(MRIConvert(), name='mri_convert')
mgz2nii.inputs.out_type = 'niigz'

# bet = Node(BET(), name='bet')
# bet.output_file = 'T1_brain.nii.gz'
# bet.inputs.mask = True

# Registration:  T1 - MNI
antsreg = Node(Registration(), name='antsreg')
# TODO: check validity of this values
# antsreg.inputs.args = '--float'
# antsreg.inputs.collapse_output_transforms = True
# antsreg.inputs.fixed_image = template
# antsreg.inputs.initial_moving_transform_com = True
# antsreg.inputs.output_warped_image = True
# antsreg.inputs.sigma_units=['vox']*3
# antsreg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
# antsreg.inputs.metric = ['MI', 'MI', 'CC']
# antsreg.inputs.terminal_output = 'file'
# antsreg.inputs.winsorize_lower_quantile=0.005
# antsreg.inputs.winsorize_upper_quantile=0.995
# antsreg.inputs.convergence_threshold = [1e-08, 1e-08, -0.01]
# antsreg.inputs.convergence_window_size = [20,20,5]
# antsreg.inputs.metric=['Mattes', 'Mattes', ['Mattes', 'CC']]
# antsreg.inputs.metric_weight = [1.0, 1.0, [0.5, 0.5]]
# antsreg.inputs.number_of_iterations = [[1000, 500, 250, 100],                                                                             [1000, 500, 250, 100],
#                                        [100,   70,  50,  20]]
# antsreg.inputs.radius_or_number_of_bins = [32,32,[32,4]]
# antsreg.inputs.sampling_percentage=[0.3, 0.3, [None, None]]
# antsreg.inputs.sampling_strategy = ['Regular', 'Regular', [None, None]]
# antsreg.inputs.shrink_factors = [[3, 2, 1],
#                           [3, 2, 1],
#                           [4, 2, 1]]
# antsreg.inputs.smoothing_sigmas = [[4.0, 2.0, 1.0],
#                             [4.0, 2.0, 1.0],
#                             [1.0, 0.5, 0.0]]
# antsreg.inputs.transform_parameters=[(0.1,), (0.1,), (0.2, 3.0, 0.0)]
# antsreg.inputs.use_estimate_learning_rate_once = [True]*3
# antsreg.inputs.use_histogram_matching= [False, False, True]
# antsreg.inputs.write_composite_transform = True

# parameters from:
# http://miykael.github.io/nipype-beginner-s-guide/normalize.html
antsreg.inputs.args = '--float'
antsreg.inputs.collapse_output_transforms = True
antsreg.inputs.fixed_image = template
antsreg.inputs.initial_moving_transform_com = True
antsreg.inputs.num_threads = 1
antsreg.inputs.output_warped_image = True
antsreg.inputs.output_inverse_warped_image = True
antsreg.inputs.sigma_units=['vox']*3
antsreg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
antsreg.inputs.terminal_output = 'file'
antsreg.inputs.winsorize_lower_quantile=0.005
antsreg.inputs.winsorize_upper_quantile=0.995
antsreg.inputs.convergence_threshold = [1e-06]
antsreg.inputs.metric=['MI', 'MI', 'CC']
antsreg.inputs.metric_weight = [1.0]*3
antsreg.inputs.number_of_iterations = [[1000, 500, 250, 100],
                                       [1000, 500, 250, 100],
                                       [100,   70,  50,  20]]
antsreg.inputs.radius_or_number_of_bins = [32,32,4]
antsreg.inputs.sampling_percentage=[0.25, 0.25, 1]
antsreg.inputs.sampling_strategy = ['Regular', 'Regular', 'None']
antsreg.inputs.shrink_factors = [[8,4,2,1]]*3
antsreg.inputs.smoothing_sigmas = [[3, 2, 1, 0]]*3
antsreg.inputs.transform_parameters=[(0.1,), (0.1,), (0.1, 3.0, 0.0)]
antsreg.inputs.use_histogram_matching= True
antsreg.inputs.write_composite_transform = True

#TODO: check how to check the number of nodes

# Corregister the median to surface
bbreg = Node(BBRegister(), name='bbRegister')
bbreg.inputs.init = 'fsl'
bbreg.inputs.contrast_type = 't2'
bbreg.inputs.out_fsl_file = True
bbreg.inputs.subjects_dir = data_in_dir

# convert BBRegister transformation to ANTS ITK format. Necessary to convert
# fsl.style affine registration into ANTS itk format
convert2itk = Node(C3dAffineTool(), name='convert2itk')
convert2itk.inputs.fsl2ras = True
convert2itk.inputs.itk_transform = True

# concatenate BBRegister's and ANTS's transform into a list
merge = Node(Merge(2), interfield=['in2'], name='AntsBBregisterMerge')

### normalise anatomical and functional image
# transfrom EPI, first to anatomical and then to MNI
warpall = MapNode(ApplyTransforms(), name='warpall', iterfield=['input_image'])
warpall.inputs.args = '--float'
warpall.inputs.reference_image = template # reference image space that you wish to warp INTO
warpall.inputs.input_image_type = 3 # define input image type scalar(1), vector(2), tensor(3)
warpall.inputs.interpolation='Linear'
# TODO:check num thread
warpall.inputs.num_threads = 1
warpall.inputs.terminal_output = 'file' # writes output to file
warpall.inputs.invert_transform_flags = [False, False]
#------------------------------------------------------------------------------
#                             Set up Workflow
#------------------------------------------------------------------------------

# Specify input and output Node
# ------------------------------
# Define Infosource, which is the input Node. Information from subject_id are
# obtained from this Node
infosource = Node(interface=IdentityInterface(fields = ['subject_id']), name = 'InfoSource')
infosource.iterables = ('subject_id', subjects_list)

# Define DataSink, where all data will be saved
data_sink = Node(DataSink(), name = 'DataSink')
data_sink.inputs.base_directory = os.path.join(data_out_dir, 'preprocessing_out')
# data_sink.inputs.container = '{subject_id}'
substitutions = [('_subject_id_', ''), ('_fwhm', 'fwhm')]
data_sink.inputs.substitutions = substitutions

# Define workflow name and where output will be saved
preproc = Workflow(name = 'preprocessing')
preproc.base_dir = data_out_dir

# Define connection between nodes
preproc.connect([
       # iterate over epi and t1 files
       (infosource,          fslsource,     [('subject_id'     ,'subject_id'     )] ),
       (infosource,          datasource,    [('subject_id'     , 'subject_id'    )] ),
       (datasource,          mean_image,     [('epi'            , 'in_file'       )] ),
       # (mean_image,          data_sink       [('out_file'       , 'meanimage'     )] ),
       (fslsource,           mgz2nii,        [('T1'             , 'in_file'       )] ),
       (mgz2nii,             convert2itk,    [('out_file'       , 'reference_file')] ),
       (mean_image,          convert2itk,    [('out_file'       , 'source_file'   )] ),
       (infosource,          bbreg,          [('subject_id'     , 'subject_id'    )] ),
       (mean_image,          bbreg,          [('out_file'       , 'source_file'   )] ),
       (bbreg,               convert2itk,    [('out_fsl_file'   , 'transform_file')] ),
       (datasource,          antsreg,        [('t1'             , 'moving_image'  )] ),
       (antsreg,             merge,          [('composite_transform', 'in1'       )] ),
       (antsreg,             data_sink,      [('warped_image'   ,
                                                            'antsreg.warped_image'),
                                              ('inverse_warped_image',
                                                    'antsreg.inverse_warped_image'),
                                              ('composite_transform',
                                                              'antsreg.transform'),
                                              ('inverse_composite_transform',
                                                      'antsreg.inverse_transform')] ),
       (convert2itk,         merge,          [('itk_transform'  , 'in2'           )] ),
       (mean_image,          warpall,        [('out_file'       , 'input_image'   )] ),
       (merge,               warpall,        [('out'            , 'transforms'    )] ),
       (warpall,             data_sink,        [('output_image'  ,
                                                          'warp_complete.warpall')] ),
       ])

# save graph of the workflow into the workflow_graph folder
preproc.write_graph(os.path.join(data_out_dir, 'preprocessing_out', 'workflow_graph',
    'workflow_graph.dot'))
preproc.run()
# preproc.run('MultiProc', plugin_args={'n_procs': 8})
