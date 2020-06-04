import pickle
import os
import string

class IntermediateResultsDALPickle(object):
    VALID_CHARS = frozenset("-_%s%s" % (string.ascii_letters, string.digits))

    def __init__(self, work_dir):

        # Only allow valid filenames in tags/ids (Based on: https://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename )

        self._work_dir = work_dir
        IntermediateResultsDALPickle.create_dir(self._work_dir)

    def _make_canonical_fname(self, result_type, result_id):
        result_type = ''.join(c for c in result_type if c in IntermediateResultsDALPickle.VALID_CHARS)
        result_id = ''.join(c for c in result_id if c in IntermediateResultsDALPickle.VALID_CHARS)
        return '{}.pkl'.format(os.path.join(self._work_dir, result_type, result_id))

    def _make_result_type_dir(self, result_type):
        result_type = ''.join(c for c in result_type if c in IntermediateResultsDALPickle.VALID_CHARS)
        result_type_dir = os.path.join(self._work_dir, result_type)
        IntermediateResultsDALPickle.create_dir(result_type_dir)


    def load_prev_results(self, result_type, result_id):
        fname = self._make_canonical_fname(result_type, result_id)

        return IntermediateResultsDALPickle.load_prev_single_file_results(fname)

    def store_result(self, result_type, result_id, contents):
        # ensure that the result type folder exists
        self._make_result_type_dir(result_type)

        fname = self._make_canonical_fname(result_type, result_id)
        fname_partial = '{}_partial'.format(fname)

        # Store the contents in a partial file and then change it's name
        with open(fname_partial, 'wb') as out_f:
            pickle.dump(contents, out_f, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(fname_partial, fname)

    @staticmethod
    def create_dir(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    @staticmethod
    def load_prev_single_file_results(fname):
        # Check that the prev result file exists
        if not os.path.exists(fname):
            return False, None

        # load the contents, and return it
        with open(fname, 'rb') as in_f:
            contents = pickle.load(in_f)

        return True, contents



