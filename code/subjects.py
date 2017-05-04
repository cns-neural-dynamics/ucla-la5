import json

def load_subjects(filename, nsubjects=None):
    # Load the JSON file.
    with open(filename) as f:
        dct = json.load(f)
    subjects_healthy = dct['subjects']['healthy']
    subjects_schizo = dct['subjects']['schizo']

    # Return a balanced amount of healthy and schizophrenic patients.
    if nsubjects is None:
        nsubjects = len(subjects_healthy)
    nsubjects = min(len(subjects_healthy),
                    len(subjects_schizo),
                    nsubjects)
    print('--------------------------------------------------------------------')
    print(' Subjects')
    print('--------------------------------------------------------------------')
    print('')
    print('TOT number of subjects:        %d'  %(2 * nsubjects))
    print('')
    return subjects_healthy[:nsubjects] + subjects_schizo[:nsubjects]
