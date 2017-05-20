
import os
from nipype.interfaces.fsl import Info, FSLCommand, MCFLIRT, MeanImage, TemporalFilter, IsotropicSmooth, BET, GLM, BinaryMaths
from nipype.interfaces.freesurfer import BBRegister, MRIConvert
from nipype.interfaces.ants import Registration, ApplyTransforms
from nipype.interfaces.c3 import C3dAffineTool
from nipype.interfaces.utility import Merge
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.io import DataSink, FreeSurferSource, DataGrabber
from nipype.interfaces.utility import IdentityInterface, Function
from argparse import ArgumentParser

from nipypext import nipype_wrapper
from extract_roi import extract_roi

def get_file(in_file):
    """
    ApplyTransforms ouptu is a list. This function gets the path to warped file
    from the import generated list
    """
    path2file = in_file[0]
    return path2file


def get_VOIs(preprocessed_image, segmented_image_path, segmented_regions_path,
             subject_id):
    '''Extract VOIs from preprocesssed image using the provided atlas and the
    corresponding Lookup table (where the different regions are labled)'''

    import os
    import numpy as np
    import nibabel as nib

    # Load segmented image and obtain data from the image
    segmented_image = nib.load(segmented_image_path)
    segmented_image_data = segmented_image.get_data()
    # Load the Lookup Table which assigns each region to one specific pixel
    # intensity
    segmented_regions = np.genfromtxt(segmented_regions_path, dtype=[('numbers',
                                                                        '<i8'), ('regions', 'S31'), ('labels', 'i4')], delimiter=',')

    # Load subjects preprocessed image and extract data from the image
    image = nib.load(preprocessed_image)
    image_data = image.get_data()
    # Obtain different time points (corrisponding to the image TR)
    time = image_data.shape[3]
    # Initialise matrix where the averaged BOLD signal for each region will be
    # saved
    avg = np.zeros((segmented_regions['labels'].shape[0], time))
    # Find the voxels on the subject's image that correspond to the voxels on
    # the labeled image for each time point and calculate the mean BOLD response
    for region in range(len(segmented_regions)):
        label = segmented_regions['labels'][region]
        for t in range(time):
            data = image_data[:, :, :, t]
            boolean_mask = np.where(segmented_image_data == label)
            data = data[boolean_mask[0], boolean_mask[1], boolean_mask[2]]
            # for all regions calculate the mean BOLD at each time point
            avg[region, t] = data.mean()
    # save data into a text file
    file_name = '%s.txt' % subject_id
    np.savetxt(file_name, avg, delimiter=' ', fmt='%5e')
    return os.path.abspath(file_name)

def get_lookuptable(segmented_regions_file):
    import numpy as np
    lookuptable = np.genfromtxt(segmented_regions_file,
                                names='intensity, regions, numbers',
                                dtype=None,
                                delimiter=','
                                )
    return lookuptable

def preprocessing_pipeline(subject, base_path, preprocessing_type=None):
    '''
    The second argument specify the type of preprocessing
    '''
    # Note: Subjects need to be passed as a list
    subjects_list = [subject]

    if preprocessing_type == None:
        raise ValueError('Pass type of image you want to be preprocessessed')
    #------------------------------------------------------------------------------
    #                              Specify Variabless
    #------------------------------------------------------------------------------
    # all outputs will ge generated in compressed nifti format
    FSLCommand.set_default_output_type('NIFTI_GZ')
    # location of template file
    template = Info.standard_image('MNI152_T1_2mm.nii.gz')

    # Data Location
    voi_in_dir = os.path.join(base_path, 'data_in', 'voi_extraction')
    data_in_dir = os.path.join(base_path, 'data_in', 'reconall_data')
    data_out_dir = os.path.join(base_path, 'data_out', preprocessing_type)

    # Get functional image
    datasource = Node(interface=DataGrabber(infields=['subject_id'],
            outfields=['epi', 't1', 'aseg_auto']), name='datasource')
    datasource.inputs.base_directory = data_in_dir
    datasource.inputs.template = '*'
    datasource.inputs.sort_filelist = True
    if preprocessing_type == 'rest':
        datasource.inputs.field_template = dict(epi=os.path.join('%s', 'func', '%s_task-rest_bold.nii.gz'),
                                                t1=os.path.join('%s', 'anat', '%s_T1w.nii.gz'),
                                                aseg_auto=os.path.join('%s', 'mri', 'aseg.auto.mgz'))
    elif preprocessing_type == 'task':
        datasource.inputs.field_template = dict(epi=os.path.join('%s', 'func', '%s_task-stopsignal_bold.nii.gz'),
                                                t1=os.path.join('%s', 'anat', '%s_T1w.nii.gz'),
                                                aseg_auto=os.path.join('%s', 'mri', 'aseg.auto.mgz'))
    # this specifies the variables for the field_templates
    datasource.inputs.template_args = dict(epi=[['subject_id', 'subject_id']],
                                           t1=[['subject_id', 'subject_id']],
                                           aseg_auto=[['subject_id']])
    datasource.inputs.subject_id = subjects_list
    #------------------------------------------------------------------------------
    # Use data grabber specific for FreeSurfer data
    fslsource = Node(FreeSurferSource(), name='getFslData')
    fslsource.inputs.subjects_dir = data_in_dir

    # Generate mean image - only for the EPI image
    mean_image = Node(MeanImage(), name='Mean_Image')
    # mean_image.inputs.out_file = 'MeanImage.nii.gz'

    # motion correction
    mot_par = Node(MCFLIRT(), name='motion_correction')
    mot_par.inputs.mean_vol = True
    mot_par.inputs.save_rms = True
    mot_par.inputs.save_plots = True

    # convert FreeSurfer's MGZ format to nii.gz format
    mgz2nii = Node(MRIConvert(), name='mri_convert')
    mgz2nii.inputs.out_type = 'niigz'

    bet = Node(BET(), name='bet')
    bet.inputs.frac = 0.3 # recommended by ICA-Aroma manual
    bet.inputs.mask = True

    # parameters from:
    # http://miykael.github.io/nipype-beginner-s-guide/normalize.html
    antsreg = Node(Registration(), name='antsreg')
    antsreg.inputs.args = '--float'
    # antsreg.inputs.collapse_output_transforms = True
    antsreg.inputs.fixed_image = template
    antsreg.inputs.initial_moving_transform_com = True
    antsreg.inputs.num_threads = 8
    antsreg.inputs.output_warped_image = True
    antsreg.inputs.output_inverse_warped_image = True
    antsreg.inputs.sigma_units = ['vox']*3
    antsreg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    antsreg.inputs.terminal_output = 'file'
    antsreg.inputs.winsorize_lower_quantile = 0.005
    antsreg.inputs.winsorize_upper_quantile = 0.995
    antsreg.inputs.convergence_threshold = [1e-06]
    antsreg.inputs.metric = ['MI', 'MI', 'CC']
    antsreg.inputs.metric_weight = [1.0]*3
    antsreg.inputs.number_of_iterations = [[1000, 500, 250, 100],
                                           [1000, 500, 250, 100],
                                           [100,   70,  50,  20]]
    antsreg.inputs.radius_or_number_of_bins = [32,32,4]
    antsreg.inputs.sampling_percentage = [0.25, 0.25, 1]
    antsreg.inputs.sampling_strategy = ['Regular', 'Regular', 'None']
    antsreg.inputs.shrink_factors = [[8,4,2,1]]*3
    antsreg.inputs.smoothing_sigmas = [[3, 2, 1, 0]]*3
    antsreg.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3.0, 0.0)]
    antsreg.inputs.use_histogram_matching = True
    antsreg.inputs.write_composite_transform = True
    antsreg.inputs.save_state = 'savestate.mat'

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
    warpmean = MapNode(ApplyTransforms(), name='warpmean', iterfield=['input_image'])
    warpmean.inputs.args = '--float'
    warpmean.inputs.reference_image = template # reference image space that you wish to warp INTO
    warpmean.inputs.input_image_type = 3 # define input image type scalar(1), vector(2), tensor(3)
    warpmean.inputs.interpolation = 'Linear'
    warpmean.inputs.num_threads = 1
    warpmean.inputs.terminal_output = 'file' # writes output to file
    warpmean.inputs.invert_transform_flags = [False, False]

    warpall = MapNode(ApplyTransforms(), name='warpall', iterfield=['input_image'])
    warpall.inputs.args = '--float'
    warpall.inputs.reference_image = template
    warpall.inputs.input_image_type = 3
    warpall.inputs.interpolation = 'Linear'
    warpall.inputs.num_threads = 1
    warpall.inputs.terminal_output = 'file' # writes output to file
    warpall.inputs.invert_transform_flags = [False, False]

    warpaseg = MapNode(ApplyTransforms(), name='warpaseg', iterfield=['input_image'])
    warpaseg.inputs.args = '--float'
    warpaseg.inputs.reference_image = template
    warpaseg.inputs.input_image_type = 3
    warpaseg.inputs.interpolation = 'Linear'
    warpaseg.inputs.num_threads = 1
    warpaseg.inputs.terminal_output = 'file' # writes output to file

    warpasegnii = Node(MRIConvert(), name='warpaseg_mgz2nii')
    warpasegnii.inputs.out_type = 'niigz'

    # get path from warp file
    warpall2file = Node(name='warpmean2file', interface=Function(input_names=['in_file'],
        output_names=['out_file'], function=get_file))
    warpmean2file = Node(name='warpall2file', interface=Function(input_names=['in_file'],
        output_names=['out_file'], function=get_file))
    warpaseg2file = Node(name='warpaseg2file', interface=Function(input_names=['in_file'],
        output_names=['out_file'], function=get_file))

    # Perform ICA to find components related to motion (implemented on ICA-Aroma)
    # inputs for the ICA-aroma function
    ica_aroma = Node(name='ICA_aroma',
                     interface=Function(input_names=['inFile', 'outDir',
                     'mc', 'subject_id', 'mask', 'denType'],
                                        output_names=['output_file', 'denType'],
                                        function=nipype_wrapper.get_ica_aroma))
    ica_aroma.inputs.outDir = os.path.join(data_out_dir, 'preprocessing_out', 'ica_aroma')
    ica_aroma.iterables = ('denType', ['aggr', 'nonaggr'])

    # Generate mean image of ICA-AROMA output
    mean_ica = Node(MeanImage(), name='Mean_Image_ICA')

    # Extract the desing matrix
    glm_design = Node(name='GLM_Design_Matrix',
                      interface=Function(input_names=['subjects', 'network_type', 'extract_csf_wm',
                                                      'input_basepath', 'input_file', 'segmented_image', 'lookuptable',
                                                      'output_basepath', 'ica_aroma_type', 'network_mask_filename'],
                                         output_names=['design_matrix'],
                                         function=extract_roi))
    glm_design.inputs.network_type = 'full_network'
    glm_design.inputs.extract_csf_wm = True
    glm_design.inputs.network_mask_filename = None
    glm_design.inputs.input_basepath = os.path.join(data_out_dir, 'preprocessing_out')
    segmented_region_path = os.path.join(voi_in_dir, 'csf_wm_LookupTable')
    #FIXME: rename this input
    glm_design.inputs.lookuptable = get_lookuptable(segmented_region_path)
    glm_design.inputs.output_basepath = os.path.join(data_out_dir, 'preprocessing_out', 'final_image_wm_csf')

    glm = Node(GLM(), name='GLM_Nuissance')
    glm.inputs.demean = True
    glm.inputs.out_res_name = 'denoised_func_data_filt_wm_csf_extracted.nii.gz'
    glm.inputs.out_file = 'glm_betas.nii.gz'

    # spatial filtering
    iso_smooth_all = Node(IsotropicSmooth(), name='SpatialFilterAll')
    iso_smooth_all.inputs.fwhm = 5
    # spatial filtering
    iso_smooth_mean = Node(IsotropicSmooth(), name='SpatialFilterMean')
    iso_smooth_mean.inputs.fwhm = 5

    # temporal filtering
    # note: TR for this experiment is 2 and we are setting a filter of 100s.
    # Therefore, fwhm = 0.5 Hrz/0.01 Hrz = 50.
    # The function here, however, requires the fwhm (aka sigma) of this value, hence, its half.
    temp_filt = Node(TemporalFilter(), name='TemporalFilter')
    temp_filt.inputs.highpass_sigma = 25

    final_mean = Node(BinaryMaths(), name='AddMean')
    final_mean.inputs.operation = 'add'
    final_mean.inputs.terminal_output = 'file'

    #------------------------------------------------------------------------------
    #                             Set up Workflow
    #------------------------------------------------------------------------------

    # Specify input and output Node
    # ------------------------------
    # Define Infosource, which is the input Node. Information from subject_id are
    # obtained from this Node
    infosource = Node(interface=IdentityInterface(fields=['subject_id']), name='InfoSource')
    infosource.iterables = ('subject_id', subjects_list)

    # Define DataSink, where all data will be saved
    data_sink = Node(DataSink(), name='DataSink')
    data_sink.inputs.base_directory = os.path.join(data_out_dir, 'preprocessing_out')
    # data_sink.inputs.container = '{subject_id}'
    substitutions = [('_subject_id_', ''),
                     ('_fwhm', 'fwhm'),
                     ('_warpmean0', 'warpmean'),
                     ('_warpall0', 'warpall'),
                     ('_warpaseg0', 'warpaseg'),
                     ('_denType_aggr', 'icaroma_aggr'),
                     ('_denType_nonaggr', 'icaroma_nonaggr')]
    data_sink.inputs.substitutions = substitutions

    # Define workflow name and where output will be saved
    preproc = Workflow(name='preprocessing')
    preproc.base_dir = data_out_dir

    # Define connection between nodes
    preproc.connect([
        # iterate over epi and t1 files
        (infosource,          fslsource,      [('subject_id'     , 'subject_id'    )] ),
        (infosource,          datasource,     [('subject_id'     , 'subject_id'    )] ),
        # get motion parameters
        (datasource,          mot_par,        [('epi'            , 'in_file'       )] ),
        (mot_par,             data_sink,      [('par_file'       ,
                                                                 'mcflirt.par_file')] ),
        # get mean image (functional data)
        (mot_par,             mean_image,     [('out_file'       , 'in_file'       )] ),
        # Co-register T1 and functional image
        (fslsource,           mgz2nii,        [('T1'             , 'in_file'       )] ),
        (mgz2nii,             convert2itk,    [('out_file'       , 'reference_file')] ),
        (mean_image,          convert2itk,    [('out_file'       , 'source_file'   )] ),
        (infosource,          bbreg,          [('subject_id'     , 'subject_id'    )] ),
        (mean_image,          bbreg,          [('out_file'       , 'source_file'   )] ),
        (bbreg,               convert2itk,    [('out_fsl_file'   , 'transform_file')] ),
        # Normalise T1 to MNI
        (datasource,          antsreg,        [('t1'             , 'moving_image'  )] ),
        # concatenate affine and ants transforms into a list
        (antsreg,             merge,          [('composite_transform', 'in1'       )] ),
        (convert2itk,         merge,          [('itk_transform'  , 'in2'           )] ),
        (antsreg,             data_sink,      [('warped_image'   ,
                                                            'antsreg.warped_image'),
                                              ('inverse_warped_image',
                                                    'antsreg.inverse_warped_image'),
                                              ('composite_transform',
                                                              'antsreg.transform' ),
                                              ('inverse_composite_transform',
                                                      'antsreg.inverse_transform'  )] ),
        # Use T1 transfomration to register mean functional image to MNI space
        (mean_image,          warpmean,        [('out_file'       , 'input_image'  )] ),
        (merge,               warpmean,         [('out'           , 'transforms'   )] ),
        (warpmean,             data_sink,      [('output_image'   ,
                                                          'warp_complete.warpmean' )] ),
        # Use T1 transformation to register the functional image to MNI space
        (datasource,          warpall,         [('epi'            , 'input_image'  )] ),
        (merge,               warpall,         [('out'            , 'transforms'   )] ),
        (warpall,             data_sink,       [('output_image'   ,
                                                          'warp_complete.warpall.@')] ),
        # need to convert list of path given by warp all into path
        (warpall,             warpall2file,    [('output_image'   , 'in_file'      )] ),
        (warpmean,            warpmean2file,  [('output_image'    , 'in_file'      )] ),
        # register aseg.auto image with WM and CSF segmentation
        (datasource,          warpaseg,        [('aseg_auto'      , 'input_image'  )] ),
        (antsreg,             warpaseg,        [('composite_transform',
                                                                     'transforms'  )] ),
        (warpaseg,            data_sink,       [('output_image'   ,
                                                         'warp_complete.warpaseg.@')] ),
        (warpaseg,            warpaseg2file,   [('output_image'   ,  'in_file'     )] ),
        (warpaseg2file,       warpasegnii,     [('out_file'       ,  'in_file'     )] ),
        (warpasegnii,         data_sink,       [('out_file'       ,
                                                         'warp_complete.warpaseg.@nii')]),
        # skull strip EPI for ICA-AROMA
        (warpall2file,        bet,             [('out_file'       , 'in_file'      )] ),
        (bet,                 data_sink,       [('mask_file'      , 'bet.mask'     )] ),
        # do spatial filtering (mean functional data)
        (warpmean2file,        iso_smooth_mean,[('out_file'       , 'in_file'      )] ),
        (iso_smooth_mean,      data_sink,      [('out_file'       ,
                                                             'spatial_filter.mean' )] ),
        # do spatial filtering (functional data)
        (warpall2file,        iso_smooth_all,  [('out_file'       , 'in_file'      )] ),
        (iso_smooth_all,      data_sink,       [('out_file'       ,
                                                              'spatial_filter.all' )] ),
        # ICA-AROMA
        # run ICA using normalised image
        (iso_smooth_all,      ica_aroma,      [('out_file'        , 'inFile'       )] ),
        (infosource,          ica_aroma,      [('subject_id'      , 'subject_id'   )] ),
        (mot_par,             ica_aroma,      [('par_file'        , 'mc'           )] ),
        (bet,                 ica_aroma,      [('mask_file'       , 'mask'         )] ),
        # ICA-AROMA mean image
        (ica_aroma,           mean_ica,        [('output_file'    , 'in_file'      )] ),
        # Extract CSF + WM
        (infosource,          glm_design,      [('subject_id'     , 'subjects'     )] ),
        (warpasegnii,         glm_design,      [('out_file'       ,
                                                                'segmented_image'  )] ),
        (ica_aroma,           glm_design,      [('output_file'    , 'input_file'   )] ),
        (ica_aroma,           glm_design,      [('denType'        ,
                                                                 'ica_aroma_type'  )] ),
        (ica_aroma,           glm,             [('output_file'    , 'in_file'      )] ),
        (glm_design,          glm,             [('design_matrix'   ,  'design'     )] ),
        # Apply temporal filtering
        (glm,                 temp_filt,       [('out_res'        , 'in_file'      )] ),
        (temp_filt,           data_sink,       [('out_file'       , 'temp_filt'    )] ),
        # Add mean to the dataset
        (temp_filt,           final_mean,      [('out_file'       , 'in_file'      )] ),
        (mean_ica,            final_mean,      [('out_file'       , 'operand_file' )] ),
        (final_mean,          data_sink,       [('out_file'       , 'final_image'  )] ),
    ])

    # save graph of the workflow into the workflow_graph folder
    preproc.write_graph(os.path.join(data_out_dir, 'preprocessing_out', 'workflow_graph',
        'workflow_graph.dot'))
    preproc.run()
    # preproc.run(plugin = 'SGEGraph', plugin_args = '-q short.q')
    # preproc.run('MultiProc', plugin_args={'n_procs': 8})

if __name__ == '__main__':

    parser = ArgumentParser(
            description='Call preprocessing pipeline for each subject'
            )
    parser.add_argument(
            '-s', '--subID',
            dest='subject',
            help='Subject ID'
            )
    parser.add_argument(
            '-t', '--analysis-type', dest='preprocessing_type',
            help='Type of analysis to be performed'
            )
    parser.add_argument(
            '-p' '--base-path', dest='base_path',
            help='Data base path'
            )
    parser.add_argument(
            '-w', '--extract_csf_wm', dest='extract_csf_wm',
            action='store_true',
            help='Perform extraction of CSF and WM'
    )

    args = parser.parse_args()
    # Call preproecessing function
    preprocessing_pipeline(args.subject, args.base_path, args.preprocessing_type)
