import time
import sys
import logging
import numpy as np
import os
import scipy.spatial.distance as ds


def get_logger(filename):
    """Return a logger instance that writes in filename

    Args:
        filename: (string) path to log.txt

    Returns:
        logger: (instance of logger)

    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    return logger


def get_dir(dataname, filename):
    """ get the file path based on this data set."""
    indir = "Data/" + dataname + "/intermediate"
    return os.path.join(indir, filename)


def read_type_id(filename):
    """
    Construct a dictionary with type name as the key,
    while the type id is the value
    """
    with open(filename,'r') as f:
        type_id_dic = {}
        for line in f:
            splited = line.rstrip().split()
            type_id_dic[splited[0]] = int(splited[1])
        return type_id_dic


def read_id_type(filename):
    """
    Construct a dictionary with type id as the key,
    while the type name is the value
    """
    with open(filename,'r') as f:
        id_type_dic = {}
        for line in f:
            splited = line.rstrip().split()
            id_type_dic[int(splited[1])] = splited[0]
        return id_type_dic


def get_glove_vec(glove_file):
    """
    get the whole embedding dictionary
    """
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec[curr_word] = np.array(line[1:], dtype=np.float32)
    return word_to_vec


def replace_nan(X):
    ''' replace nan and inf to 0 '''
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    return X


def Compute_Sim(sig_y1, sig_y2, sim_scale):
    ''' compute class label similarity '''
    dist = ds.cdist(sig_y1, sig_y2, 'euclidean')
    dist = dist.astype(np.float32)
    Sim = np.exp(-np.square(dist) * sim_scale);
    s = np.sum(Sim, axis=1)
    Sim = replace_nan(Sim / s[:, None])
    return Sim

def write_file(filename, x, y):
    with open(filename, "w") as f:
        if isinstance(x, list):
            for item, item2 in zip(x, y):
                f.write(str(item) + "," + str(set(np.where(item2 == 1)[0]))+'\n')
        f.close()

class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)


