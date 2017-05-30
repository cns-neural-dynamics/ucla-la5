import subprocess

subjects = ['sub-10225', 'sub-10227',
            'sub-10228', 'sub-10235', 'sub-10249', 'sub-10269', 'sub-10271',
            'sub-10273', 'sub-10274', 'sub-10290', 'sub-10292', 'sub-10304',
            'sub-10316', 'sub-10321', 'sub-10325', 'sub-10329', 'sub-10339',
            'sub-10340', 'sub-10345', 'sub-10347', 'sub-10356', 'sub-10361',
            'sub-10365', 'sub-10376', 'sub-10377', 'sub-10388', 'sub-10429',
            'sub-10438', 'sub-10440', 'sub-50005', 'sub-50006',
            'sub-50007', 'sub-50008', 'sub-50010', 'sub-50013', 'sub-50014',
            'sub-50015', 'sub-50016', 'sub-50020', 'sub-50021', 'sub-50022',
            'sub-50023', 'sub-50025', 'sub-50027', 'sub-50029', 'sub-50032',
            'sub-50033', 'sub-50034', 'sub-50035', 'sub-50036', 'sub-50038',
            'sub-50043', 'sub-50047', 'sub-50048', 'sub-50049', 'sub-50050',
            'sub-50051', 'sub-50052', 'sub-50053', 'sub-50054', 'sub-50055',
            'sub-50056', 'sub-50058', 'sub-50059', 'sub-50060', 'sub-50061',
            'sub-50064', 'sub-50066', 'sub-50067', 'sub-50069', 'sub-50073',
            'sub-50075', 'sub-50076', 'sub-50077', 'sub-50080', 'sub-50081',
            'sub-50083', 'sub-50085']

for subject in subjects:
    cmd = "tmux new-session -d -s {0} 'python preprocessing_workflow.py -s {0} -p /group/dynamics/scz_dynamics/ucla-la5 -t task'".format(subject)
    ob_id = subprocess.check_output(cmd, shell=True)
    print 'Done {0}'.format(subject)
