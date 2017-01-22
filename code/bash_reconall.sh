#!/bin/bash
# This script strat the recon-all command on all subjects
# 1. Force bash
#$ -S /bin/bash
# 6. Set the name of the job.
#$ -N array
# output were saved on your home folder

base_path="/scratch/jdafflon/personal/data_in/ucla_la5/ds000030"
# sub_list=" sub-10171, sub-10189, sub-10193, sub-10206,
#            sub-10217, sub-10225, sub-10227, sub-10228, sub-10235,
#            sub-10249, sub-10269, sub-10271, sub-10273, sub-10274,
#            sub-10280, sub-10290, sub-10292, sub-10299, sub-10304,
#            sub-10316, sub-10321, sub-10325, sub-10329, sub-10339,
#            sub-10340, sub-10345, sub-10347, sub-10356, sub-10361,
#            sub-10365, sub-10376, sub-10377, sub-10388"
sub_list="sub-10189"
# iterate over subjects
for sub in $sub_list;
do
    file_path=$base_path"/"${sub}"/anat/"${sub}_"T1w.nii.gz"
    reconall="recon-all -all -i "$file_path" -subjid "${sub}" -sd "$base_path"/reconall_data"
    echo $reconall
    fsl_sub "$reconall"
done
