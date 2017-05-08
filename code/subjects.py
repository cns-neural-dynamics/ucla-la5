import json
import logging


def load_subjects(filename, golden_subjects, nsubjects=None):
    logging.info('')
    logging.info('--------------------------------------------------------------------')
    logging.info(' Subjects')
    logging.info('--------------------------------------------------------------------')
    logging.info('')

    # Load the JSON file.
    with open(filename) as f:
        dct = json.load(f)
    subjects_healthy = dct['subjects']['healthy']
    subjects_schizo = dct['subjects']['schizo']
    subjects_golden = dct['subjects']['golden']

    # Return a balanced amount of healthy and schizophrenic patients.
    if golden_subjects:
        nsubjects = len(subjects_golden)
        logging.info('TOT number of golden subjects: %d' % (nsubjects))
        logging.info('')

        return subjects_golden
    else:
        if nsubjects is None:
            nsubjects = len(subjects_healthy)
        nsubjects = min(len(subjects_healthy),
                    len(subjects_schizo),
                    nsubjects)

        logging.info('TOT number of subjects: %d' % (2 * nsubjects))
        logging.info('')
        return subjects_healthy[:nsubjects] + subjects_schizo[:nsubjects]
