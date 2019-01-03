import os
import logging


logger = logging.getLogger(__name__)


def batch_run(list_jobs, processing):
    '''Running batch processing

    Allow batch processing to resume, 
    in case it's interrupted in the middle

    Haven't dealt with pending operations yet

    Parameters:

    list_jobs: list of dictionary
        each dictionary contains
            ['input']:  input arguments to processing function
            ['lock']: absolute path to the lock file

    processing: a function that run the jobs

    '''

    for job in list_jobs:
        if _lock_exists(job['lock']):
            continue

        try:
            _lock_pending(job['lock'])
            processing(job['input'])
            _lock_done(job['lock'])
        except Exception:
            logger.exception("Error during batch processing")


def _lock_pending(lock_path):
    if _lock_exists(lock_path):
        raise ValueError('Lock exists')

    with open(lock_path,'a+') as f:
        f.write("pending")


def _lock_done(lock_path):
    with open(lock_path, 'w') as f:
        f.write("done")


def _lock_exists(lock_path):
    if os.path.isfile(lock_path):
        return True
    else:
        return False
