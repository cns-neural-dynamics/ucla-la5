import os
import subprocess
import pickle


base_path = "/home/jdafflon/data/ucla-la5/data_in/ds000030"
sub_list = [
        'sub-10159', 'sub-10171', 'sub-10189', 'sub-10193', 'sub-10206',
        'sub-10217', 'sub-10225', 'sub-10227', 'sub-10228', 'sub-10235',
        'sub-10249', 'sub-10269', 'sub-10271', 'sub-10273', 'sub-10274',
        'sub-10280', 'sub-10290', 'sub-10292', 'sub-10299']

        # ['sub-10304',
        # 'sub-10316', 'sub-10321', 'sub-10325', 'sub-10329', 'sub-10339',
        # 'sub-10340', 'sub-10345', 'sub-10347', 'sub-10356', 'sub-10361',
        # 'sub-10365', 'sub-10376', 'sub-10377', 'sub-10388']
# sub_list = ['sub-10429', 'sub-10438', 'sub-10440', 'sub-10448',
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
pickle_file = os.path.join(base_path, 'reconall_data', 'sub2id.pickle')
# check if pickle file exists
if os.path.exists(pickle_file):
    with open(pickle_file, 'rb') as rfp:
        sub2id = pickle.load(rfp)
else:
    sub2id = {}

for sub in sub_list:
    file_path = os.path.join(base_path, sub, 'anat', ''.join([sub, '_T1w.nii.gz']))
    # FIXME: This command is not running when calling directly from terminal
    cmd = 'srun /apps/software/freesurfer/5.3.0/bin/recon-all -all -i {0} -subjid {1} -sd {2}/reconall_data'.format(file_path, sub,base_path)
    job_id = subprocess.check_output(cmd, shell=True)
    sub2id[sub] = job_id

# save pickle with the corresponding job-id for each subject
with open(pickle_file, 'wb') as wfp:
    pickle.dump(sub2id, wfp)

