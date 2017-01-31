# This script copies original data to the recon-all folder.
import subprocess
import os
# subjects_list = ['sub-10290', 'sub-10292', 'sub-10299', 'sub-10304',
#         'sub-10316', 'sub-10321', 'sub-10325', 'sub-10329', 'sub-10339',
#         'sub-10340', 'sub-10345', 'sub-10347', 'sub-10356', 'sub-10361',
#         'sub-10365', 'sub-10376', 'sub-10377', 'sub-10388']

subjects_list = ['sub-50004', 'sub-50005', 'sub-50006', 'sub-50007', 'sub-50008',
    'sub-50010', 'sub-50013', 'sub-50014', 'sub-50015', 'sub-50016',
    'sub-50020', 'sub-50021', 'sub-50022', 'sub-50023', 'sub-50025',
    'sub-50027', 'sub-50029', 'sub-50032', 'sub-50033', 'sub-50034',
    'sub-50035', 'sub-50036', 'sub-50038', 'sub-50060', 'sub-50061',
    'sub-50064', 'sub-50066', 'sub-50067', 'sub-50069', 'sub-50073',
    'sub-50075', 'sub-50076', 'sub-50077', 'sub-50080', 'sub-50081',
    'sub-50083', 'sub-50085']
base_data = '/home/jdafflon/scratch/personal/data_in/ucla_la5/ds000030'
for sub in subjects_list:
    anat_data = os.path.join(base_data, sub, 'anat')
    beh_data  = os.path.join(base_data, sub, 'beh')
    dwi_data  = os.path.join(base_data, sub, 'dwi')
    func_data = os.path.join(base_data, sub, 'func')
    end_path  = os.path.join(base_data, 'reconall_data', sub)
    cmd = 'cp -r {0} {1} {2} {3} {4}'.format(anat_data, beh_data, dwi_data,
            func_data, end_path)
    subprocess.call(cmd, shell=True)
    print ('Done subject{}'.format(sub))
