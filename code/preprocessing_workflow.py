import os
from nipype.interfaces.fsl import Info, FSLCommand, MCFLIRT, MeanImage, TemporalFilter, IsotropicSmooth, BET
from nipype.interfaces.freesurfer import BBRegister, MRIConvert
from nipype.interfaces.ants import Registration, ApplyTransforms
from nipype.interfaces.c3 import C3dAffineTool
from nipype.interfaces.utility import Merge
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.io import DataSink, FreeSurferSource, DataGrabber
from nipype.interfaces.utility import IdentityInterface, Function


from nipypext import nipype_wrapper


# FIXME: mriconvert and ants_reg not working on new cluster
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
    segmented_image =  nib.load(segmented_image_path)
    segmented_image_data = segmented_image.get_data()
    # Load the Lookup Table which assigns each region to one specific pixel
    # intensity
    segmented_regions = np.genfromtxt(segmented_regions_path, dtype = [('numbers',
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


def preprocess_data(subjects,
                    preprocessing_input_basepath,
                    preprocessing_output_basepath):

    #------------------------------------------------------------------------------
    #                              Specify Variabless
    #------------------------------------------------------------------------------
    # all outputs will ge generated in compressed nifti format
    FSLCommand.set_default_output_type('NIFTI_GZ')
    # location of template file
    template = Info.standard_image('MNI152_T1_2mm.nii.gz')

    # Get functional image
    datasource = Node(interface=DataGrabber(infields=['subject_id'],
            outfields=['epi', 't1']), name='datasource')
    datasource.inputs.base_directory = preprocessing_input_basepath
    datasource.inputs.template = '*'
    datasource.inputs.sort_filelist = True
    datasource.inputs.field_template = dict(epi=os.path.join('%s', 'func', '%s_task-stopsignal_bold.nii.gz'),
                                            t1=os.path.join('%s', 'anat', '%s_T1w.nii.gz' ))
    datasource.inputs.template_args = dict(epi=[['subject_id', 'subject_id']],
                                           t1=[['subject_id', 'subject_id']])
    datasource.inputs.subject_id = subjects
    #------------------------------------------------------------------------------
    # Use data grabber specific for FreeSurfer data
    fslsource = Node(FreeSurferSource(), name='getFslData')
    fslsource.inputs.subjects_dir = preprocessing_input_basepath

    # Generate mean image - only for the EPI image
    mean_image = Node(MeanImage(), name = 'Mean_Image')
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
    bbreg.inputs.subjects_dir = preprocessing_input_basepath

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

    # get path from warp file
    warpall2file = Node(name='warpmean2file', interface=Function(input_names=['in_file'],
        output_names=['out_file'], function=get_file))
    warpmean2file = Node(name='warpall2file', interface=Function(input_names=['in_file'],
        output_names=['out_file'], function=get_file))

    # Perform ICA to find components related to motion (implemented on ICA-Aroma)
    # inputs for the ICA-aroma function
    ica_aroma = Node(name='ICA_aroma',
                     interface=Function(input_names=['inFile', 'outDir',
                     'mc', 'subject_id', 'mask'],
                                        output_names=['output_file'],
                                        function=nipype_wrapper.get_ica_aroma))
    outDir = os.path.join(preprocessing_output_basepath,'preprocessing_out', 'ica_aroma')
    ica_aroma.inputs.outDir = outDir

    # spatial filtering
    iso_smooth_all = Node(IsotropicSmooth(), name = 'SpatialFilterAll')
    iso_smooth_all.inputs.fwhm = 5
    # spatial filtering
    iso_smooth_mean = Node(IsotropicSmooth(), name = 'SpatialFilterMean')
    iso_smooth_mean.inputs.fwhm = 5

    # temporal filtering
    # note: TR for this experiment is 2 and we are setting a filter of 100s.
    # Therefore, fwhm = 0.5 Hrz/0.01 Hrz = 50.
    # The function here, however, requires the fwhm (aka sigma) of this value, hence, its half.
    temp_filt = Node(TemporalFilter(), name='TemporalFilter')
    temp_filt.inputs.highpass_sigma = 25

    # # Extract VOIs
    # #----------------
    # extract_vois = Node(name='extract_VOIs',
    #                          interface = Function(input_names  =
    #                                                             ['preprocessed_image',
    #                                                               'segmented_image_path',
    #                                                               'segmented_regions_path',
    #                                                               'subject_id'],
    #                                               output_names = ['output_file'],
    #                                               function     = get_VOIs))
    # extract_vois.inputs.segmented_image_path = os.path.join(base_path, 'data_in',
    #         'voi_extraction', 'seg_aparc_82roi_2mm.nii.gz')
    # extract_vois.inputs.segmented_regions_path = os.path.join(base_path, 'data_in', 'voi_extraction',
    #         'LookupTable')
    #------------------------------------------------------------------------------
    #                             Set up Workflow
    #------------------------------------------------------------------------------

    # Specify input and output Node
    # ------------------------------
    # Define Infosource, which is the input Node. Information from subject_id are
    # obtained from this Node
    infosource = Node(interface=IdentityInterface(fields = ['subject_id']), name = 'InfoSource')
    infosource.iterables = ('subject_id', subjects)

    # Define DataSink, where all data will be saved
    data_sink = Node(DataSink(), name = 'DataSink')
    data_sink.inputs.base_directory = os.path.join(preprocessing_output_basepath, 'preprocessing_out')
    # data_sink.inputs.container = '{subject_id}'
    substitutions = [('_subject_id_', ''),
                     ('_fwhm', 'fwhm'),
                     ('_warpmean', 'warpmean')]
    data_sink.inputs.substitutions = substitutions

    # Define workflow name and where output will be saved
    preproc = Workflow(name = 'preprocessing')
    preproc.base_dir = preprocessing_output_basepath

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
                                                          'antsreg.inverse_transform' )] ),
           # Use T1 transfomration to register mean functional image to MNI space
           (mean_image,          warpmean,        [('out_file'       , 'input_image'  )] ),
           (merge,               warpmean,         [('out'            , 'transforms'   )] ),
           (warpmean,             data_sink,      [('output_image'   ,
                                                              'warp_complete.warpmean')] ),
           # Use T1 transformation to register the functional image to MNI space
           (datasource,          warpall,         [('epi'            , 'input_image'  )] ),
           (merge,               warpall,         [('out'            , 'transforms'   )] ),
           (warpall,             data_sink,       [('output_image'   ,
                                                              'warp_complete.warpall' )] ),
           # need to convert list of path given by warp all into path
           (warpall,             warpall2file,    [('output_image'   , 'in_file'      )] ),
           (warpmean,            warpmean2file,  [('output_image'   , 'in_file'      )] ),
           # skull strip EPI for ICA-AROMA
           (warpall2file,        bet,             [('out_file'   , 'in_file'       )] ),
           (bet,                 data_sink,       [('mask_file'      , 'bet.mask'     )] ),
           # do spatial filtering (mean functional data)
           (warpmean2file,        iso_smooth_mean,[('out_file'       , 'in_file'      )] ),
           (iso_smooth_mean,      data_sink,      [('out_file'       ,
                                                                 'spatial_filter.mean')] ),
           # do spatial filtering (functional data)
           (warpall2file,        iso_smooth_all,  [('out_file'       , 'in_file'      )] ),
           (iso_smooth_all,      data_sink,       [('out_file'       ,
                                                                  'spatial_filter.all')] ),
           # ICA-AROMA
           # run ICA using normalised image
           (iso_smooth_all,     ica_aroma,       [('out_file'         , 'inFile'        )] ),
           (infosource,          ica_aroma,      [('subject_id'     , 'subject_id'     )] ),
           (mot_par,             ica_aroma,      [('par_file'       , 'mc'             )] ),
           (bet,                 ica_aroma,      [('mask_file'      , 'mask'           )] ),
           # Apply temporal filtering
           (ica_aroma,           temp_filt,      [('output_file'    , 'in_file'       )] ),
           (temp_filt,           data_sink,      [('out_file'       , 'final_image'   )] ),
           # (temp_filt,           extract_vois,   [('out_file'       ,
           #                                                        'preprocessed_image')] ),
           # (infosource,          extract_vois,   [('subject_id'     , 'subject_id'    )] ),
           # (extract_vois,        data_sink,      [('output_file'    ,   'extract_vois')] ),
           ])

    # save graph of the workflow into the workflow_graph folder
    preproc.write_graph(os.path.join(preprocessing_output_basepath, 'preprocessing_out', 'workflow_graph',
        'workflow_graph.dot'))
    preproc.run()
    # preproc.run(plugin = 'SGEGraph', plugin_args = '-q short.q')
    # preproc.run('MultiProc', plugin_args={'n_procs': 8})
