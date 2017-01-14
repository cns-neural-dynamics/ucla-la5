#!/share/apps/anaconda/bin/python
import os
import sys
from nipype.interfaces.fsl import BET, FSLCommand, MCFLIRT, utils, MeanImage, ApplyWarp, TemporalFilter, IsotropicSmooth
from nipype.interfaces.freesurfer import BBRegister, ReconAll, MRIConvert
from nipype.interfaces.ants import Registration, ApplyTransforms
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.io import SelectFiles, DataSink, DataGrabber
from nipype.interfaces.utility import IdentityInterface, Function
from nipypext import nipype_wrapper
import argparse

#------------------------------------------------------------------------------
#                                Settings
#------------------------------------------------------------------------------
# all outputs will ge generated in compressed nifti format
FSLCommand.set_default_output_type('NIFTI_GZ')

subjects_list=['sub-10159']
# subjects_list = ['sub-10159', 'sub-10171', 'sub-10189', 'sub-10193', 'sub-10206',
#         'sub-10217', 'sub-10225', 'sub-10227', 'sub-10228', 'sub-10235',
#         'sub-10249', 'sub-10269', 'sub-10271', 'sub-10273', 'sub-10274',
#         'sub-10280', 'sub-10290', 'sub-10292', 'sub-10299', 'sub-10304',
#         'sub-10316', 'sub-10321', 'sub-10325', 'sub-10329', 'sub-10339',
#         'sub-10340', 'sub-10345', 'sub-10347', 'sub-10356', 'sub-10361',
#         'sub-10365', 'sub-10376', 'sub-10377', 'sub-10388']
# Specify Variables
#-----------------------------------------------------------------------------
# Data Location
base_path = os.path.join(os.sep, 'home', 'jdafflon', 'scratch', 'personal')
data_in_dir  = os.path.join(base_path, 'data_in', 'ucla_la5', 'ds000030')
data_out_dir = os.path.join(base_path, 'data_out', 'ucla_la5')

# Define path for functional (epi image) and T1 image and create a node
datasource = Node(interface=DataGrabber(infields=['subject_id'],
    outfields=['epi', 't1']), name='datasource')
datasource.inputs.base_directory = data_in_dir
datasource.inputs.template = '*'
datasource.inputs.sort_filelist = True

datasource.inputs.field_template = dict(epi=os.path.join('%s', 'func', '%s_task-stopsignal_bold.nii.gz'),
                                       t1=os.path.join('%s', 'mri', 'T1.mgz' ))
datasource.inputs.template_args = dict(epi=[['subject_id', 'subject_id']],
                                         t1=[['subject_id']])
datasource.inputs.subject_id = subjects_list

# convert files fromm .nii to gu
mriconv = Node(MRIConvert(), name='mri_convert')
mriconv.inputs.out_type = 'niigz'

# bet = Node(BET(), name='bet')
# bet.output_file = 'T1_brain.nii.gz'
# bet.inputs.mask = True

# registration T1-MNI
bbreg = Node(BBRegister(), name='bbRegister')
bbreg.inputs.init = 'fsl'
bbreg.inputs.contrast_type = 't1'
bbreg.inputs.out_fsl_file = True
bbreg.inputs.subjects_dir = data_in_dir

# Define path for the MNI template without skull
reg_mni_reference_brain = os.path.join(os.environ['FSLDIR'], 'data', 'standard',
        'MNI152_T1_2mm_brain.nii.gz')

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
       # Define iterables and ipnut files
       (infosource,          datasource,     [('subject_id'     ,'subject_id'  )] ),
       # (datasource,          bet,            [('t1'           , 'in_file'    )] ),
       # (bet,                 data_sink,      [('out_file'     , 'bet'        ),
       #                                        ('mask_file'    , 'bet.mask'   )] ),
       (datasource,          mriconv,        [('t1'             , 'in_file'    )] ),
       (infosource,          bbreg,          [('subject_id'     , 'subject_id' )] ),
       (mriconv,             bbreg,          [('out_file'       , 'source_file')] ),
       (bbreg,               data_sink,      [('out_fsl_file'   , 'bbreg.fsl'  ),
                                              ('out_reg_file'   , 'bbreg.reg'  ),
       ])

# save graph of the workflow into the workflow_graph folder
preproc.write_graph(os.path.join(data_out_dir, 'preprocessing_out', 'workflow_graph',
    'workflow_graph.dot'))
preproc.run()
