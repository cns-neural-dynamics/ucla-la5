import os
import logging
import numpy as np
import nibabel as nib
import time
import json
import nipype.interfaces.fsl as fsl


def nipype_nuisance_regression(input_image, subject_output_path, design_output_file, ica_aroma_type):
    """Perform Regression with nuisance regressors"""
    # FIXME: pycharm cant find the command fsl_glm
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    logging.info('                   Calculating GLM with Nuisance Regressors')
    glm = fsl.GLM()
    glm.inputs.in_file = input_image
    glm.inputs.design = design_output_file
    glm.inputs.out_res_name = os.path.join(subject_output_path, 'denoised_func_data_%s_filt_wm_csf_extracted.nii.gz' %ica_aroma_type)
    glm.inputs.out_file = os.path.join(subject_output_path, 'glm_betas.nii.gz')
    glm.run()

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


def dump_extract_roi_json_(output_base_path, network_type, subjects, ica_aroma_type, segmented_image_filename):
    output_path = os.path.join(output_base_path)

    parameters_list = {}
    timestamp = time.strftime("%Y%m%d%H%M%S")

    parameters_list['timestamp'] = timestamp
    parameters_list['network_type'] = network_type
    parameters_list['subjects'] = subjects
    parameters_list['ica_aroma_type'] = ica_aroma_type
    parameters_list['segmentation_image'] = segmented_image_filename

    # Dump json file.
    with open(os.path.join(output_path, 'extract_roi.json'), 'w') as json_file:
        json.dump(parameters_list, json_file, indent=4)


def extract_roi(subjects,
                network_type,
                extract_csf_wm,
                input_basepath,
                segmented_image,
                segmented_regions_path,
                output_basepath,
                ica_aroma_type,
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
         - segmented_image : Path to the image that will be used to segment
         - lookuptable: List of regions that will be used for the
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

    if network_type != 'full_network' and extract_csf_wm == 'csf_wm':
        raise ValueError('CSF and white matter can only be extracted with the full network method')

    if extract_csf_wm:
        segmented_regions_filename = os.path.join(segmented_regions_path, 'csf_wm_LookupTable')
        lookuptable = np.genfromtxt(segmented_regions_filename,
                                    dtype=[('intensity', '<i8'), ('regions', 'S31'), ('numbers', 'S10')],
                                    delimiter=','
                                    )
    else:
        # Load the segmented regions list.
        segmented_regions_filename = os.path.join(segmented_regions_path, 'LookupTable')
        lookuptable = np.genfromtxt(segmented_regions_filename,
                                    dtype=[('numbers', '<i8'), ('regions', 'S31'), ('intensity', 'i4')],
                                    delimiter=','
                                    )


    # Load the segmented image.
    if not extract_csf_wm:
        segmented_image_nib = nib.load(segmented_image)
        segmented_image_data = segmented_image_nib.get_data()

    # Extract ROI for each subjects.

    preprocessed_image_filename = 'denoised_func_data_%s_filt.nii.gz' % ica_aroma_type
    wm_csf_extracted_image_filename = 'denoised_func_data_%s_filt_wm_csf_extracted.nii.gz' %ica_aroma_type

    logging.info('--------------------------------------------------------------------')
    logging.info(' Extract ROI')
    logging.info('--------------------------------------------------------------------')
    logging.info('')
    logging.info('* PARAMETERS')
    logging.info('network type:      %s' %(network_type))
    logging.info('ica aroma type:    %s' %(ica_aroma_type))
    logging.info('extract CSF/WM:    %s' %(extract_csf_wm))

    for subject in subjects:
        logging.info('Subject ID:        %s' %(subject))
        logging.info('')

        # Load image to segment. It is subject specific if the analysis is csf and wm extraction
        # FIXME
        # Note: This step assumes that the aseg.auto image has already been registered to standart space and converted to
        # nifti file
        if extract_csf_wm:
            segmented_image_filepath = os.path.join(input_basepath, 'warp_complete', 'warpaseg', subject, 'aseg.auto_trans_out.nii.gz')
            segmented_image_nib = nib.load(segmented_image_filepath)
            segmented_image_data = segmented_image_nib.get_data()

        # Generate the output folder.
        subject_path = os.path.join(output_basepath, subject,''.join(['icaroma_', ica_aroma_type]))
        if not os.path.exists(subject_path):
            os.makedirs(subject_path)

        # Check if ROIs has been extracted in case yes, early exit
        if network_type == 'full_network' or 'between_network':
            if os.path.exists(os.path.join(subject_path, network_type + '.txt')):
                logging.info('Time course for this subject was already extracted')
                continue
        elif network_type == 'within_network':
            if os.path.exists(os.path.join(subject_path, network_type + '_9.txt')):
                logging.info('Time course for this subject was already extracted')
                continue

        # Load the subject input image.
        if extract_csf_wm:
            image_filename = os.path.join(input_basepath, 'final_image', subject, ''.join(['icaroma_', ica_aroma_type]),
                                          preprocessed_image_filename)
        else:
            image_filename = os.path.join(input_basepath, 'final_image_wm_csf_extracted', subject, ''.join(['icaroma_', ica_aroma_type]),
                                          wm_csf_extracted_image_filename)

        image = nib.load(image_filename)
        image_data = image.get_data()
        ntpoints = image_data.shape[3]

        if network_type == 'full_network':
            # Calculate the average BOLD signal over all regions.
            avg = np.zeros((lookuptable['intensity'].shape[0], ntpoints))
            for region in range(len(lookuptable)):
                intensity = lookuptable['intensity'][region]
                # Note: Not all intensity values are integers on the csf/wm segmentaton image are int. Therefore, we use
                # the np.isclose function to find all values that are in a similar range. This lead to the inclusion of
                # a few regions.
                boolean_mask = np.where(np.isclose(segmented_image_data, intensity, atol=.9))
                for t in range(ntpoints):
                    data = image_data[:, :, :, t]
                    data = data[boolean_mask[0], boolean_mask[1], boolean_mask[2]]
                    avg[region, t] = data.mean()

            # Dump the results.
            if extract_csf_wm:
                # Fsl GLM design matrix requires (time x regressor) and demeaned data
                design = np.transpose(avg)
                design_mean = np.mean(design, axis=0)
                design -= design_mean
                design_output_file = os.path.join(subject_path, ''.join(['wm_csf_time_course', '.txt']))
                np.savetxt(design_output_file, design, delimiter=' ', fmt='%5e')
            else:
                np.savetxt(os.path.join(subject_path, 'full_network.txt'),
                           avg, delimiter=' ', fmt='%5e')


            # perform GLM with WM and CSF ans nuisance regressors
            if extract_csf_wm:
                nipype_nuisance_regression(image_filename, subject_path, design_output_file, ica_aroma_type)

        else:
            # Load the network mask.
            ntw_image = nib.load(network_mask_filename)
            ntw_data = ntw_image.get_data()
            boolean_ntw = ntw_data > 1.64

            # Find the most likely regions inside the network.
            networks = {key: [] for key in range(ntw_data.shape[3])}
            ntw_filter = np.zeros(ntw_data.shape)
            for region in range(len(lookuptable)):
                intensity = lookuptable['intensity'][region]
                boolean_mask = segmented_image_data == intensity
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
                        intensity = lookuptable['intensity'][networks[network][region]]
                        boolean_mask = np.where(segmented_image_data == intensity)
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

    # Dump json with parameters of the roi extraction.
    dump_extract_roi_json_(output_basepath, network_type, subjects, ica_aroma_type, segmented_image)

