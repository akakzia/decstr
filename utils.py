import numpy as np
from datetime import datetime
import os
import json
import subprocess
import os.path
import sys


def generate_all_goals_in_goal_space():
    goals = []
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                for d in [0, 1]:
                    for e in [0, 1]:
                        for f in [0, 1]:
                            for g in [0, 1]:
                                for h in [0, 1]:
                                    for i in [0, 1]:
                                        goals.append([a, b, c, d, e, f, g, h, i])

    return np.array(goals)


def generate_goals():
    # Returns expert-defined buckets
    buckets = {0: [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                   (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],

                1: [(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],

                2: [(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                    (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)],

                3: [(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                    (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0), (1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                     (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)],

                4:  [(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0), (0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
                     (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                     (1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                     ]}
    return buckets

def get_instruction():
    buckets = generate_goals()

    all_goals = generate_all_goals_in_goal_space().astype(np.float32)
    valid_goals = []
    for k in buckets.keys():
        # if k < 4:
        valid_goals += buckets[k]
    valid_goals = np.array(valid_goals)
    all_goals = np.array(all_goals)
    num_goals = all_goals.shape[0]
    all_goals_str = [str(g) for g in all_goals]
    valid_goals_str = [str(vg) for vg in valid_goals]

    # initialize dict to convert from the oracle id to goals and vice versa.
    # oracle id is position in the all_goal array
    g_str_to_oracle_id = dict(zip(all_goals_str, range(num_goals)))

    instructions = ['Bring blocks away_from each_other',
                    'Bring blue close_to green and red far',
                    'Bring blue close_to red and green far',
                    'Bring green close_to red and blue far',
                    'Bring blue close_to red and green',
                    'Bring green close_to red and blue',
                    'Bring red close_to green and blue',
                    'Bring all blocks close',
                    'Stack blue on green and red far',
                    'Stack green on blue and red far',
                    'Stack blue on red and green far',
                    'Stack red on blue and green far',
                    'Stack green on red and blue far',
                    'Stack red on green and blue far',
                    'Stack blue on green and red close_from green',
                    'Stack green on blue and red close_from blue',
                    'Stack blue on red and green close_from red',
                    'Stack red on blue and green close_from blue',
                    'Stack green on red and blue close_from red',
                    'Stack red on green and blue close_from green',
                    'Stack blue on green and red close_from both',
                    'Stack green on blue and red close_from both',
                    'Stack blue on red and green close_from both',
                    'Stack red on blue and green close_from both',
                    'Stack green on red and blue close_from both',
                    'Stack red on green and blue close_from both',
                    'Stack green on red and blue',
                    'Stack red on green and blue',
                    'Stack blue on green and red',
                    'Stack green on blue and blue on red',
                    'Stack red on blue and blue on green',
                    'Stack blue on green and green on red',
                    'Stack red on green and green on blue',
                    'Stack green on red and red on blue',
                    'Stack blue on red and red on green',
                    ]
    words = ['stack', 'green', 'blue', 'on', 'red', 'and', 'close_from', 'both', 'far', 'close', 'all', 'bring', 'blocks', 'away_from', 'close_to']
    length = set()
    for s in instructions:
        if len(s) not in length:
            length.add(len(s.split(' ')))


    oracle_id_to_inst = dict()
    g_str_to_inst = dict()
    for g_str, oracle_id in g_str_to_oracle_id.items():
        if g_str in valid_goals_str:
            inst = instructions[valid_goals_str.index(g_str)]
        else:
            inst = ' '.join(np.random.choice(words, size=np.random.choice(list(length))))
        g_str_to_inst[g_str] = inst
        oracle_id_to_inst[g_str] = inst

    return oracle_id_to_inst, g_str_to_inst

def init_storage(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    # path to save the model
    if args.curriculum_learning:
        logdir = os.path.join(args.save_dir, '{}_curriculum_{}'.format(datetime.now(), args.architecture))
    else:
        logdir = os.path.join(args.save_dir, '{}_no_curriculum_{}'.format(datetime.now(), args.architecture))
    if args.symmetry_trick:
        logdir += '_sym'
    if args.biased_init:
        logdir += '_biased_init'
    if args.automatic_buckets:
        logdir += '_auto'
    # path to save evaluations
    model_path = os.path.join(logdir, 'models')
    bucket_path = os.path.join(logdir, 'buckets')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(bucket_path):
        os.mkdir(bucket_path)
    with open(os.path.join(logdir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    return logdir, model_path, bucket_path


def get_stat_func(line='mean', err='std'):

    if line == 'mean':
        def line_f(a):
            return np.nanmean(a, axis=0)
    elif line == 'median':
        def line_f(a):
            return np.nanmedian(a, axis=0)
    else:
        raise NotImplementedError

    if err == 'std':
        def err_plus(a):
            return line_f(a) + np.nanstd(a, axis=0)
        def err_minus(a):
            return line_f(a) - np.nanstd(a, axis=0)
    elif err == 'sem':
        def err_plus(a):
            return line_f(a) + np.nanstd(a, axis=0) / np.sqrt(a.shape[0])
        def err_minus(a):
            return line_f(a) - np.nanstd(a, axis=0) / np.sqrt(a.shape[0])
    elif err == 'range':
        def err_plus(a):
            return np.nanmax(a, axis=0)
        def err_minus(a):
            return np.nanmin(a, axis=0)
    elif err == 'interquartile':
        def err_plus(a):
            return np.nanpercentile(a, q=75, axis=0)
        def err_minus(a):
            return np.nanpercentile(a, q=25, axis=0)
    else:
        raise NotImplementedError

    return line_f, err_minus, err_plus


class CompressPDF:
    """
    author: Pure Python
    url: https://www.purepython.org
    copyright: CC BY-NC 4.0
    Forked date: 2018-01-07 / First version MIT license -- free to use as you want, cheers.
    Original Author: Sylvain Carlioz, 6/03/2017
    Simple python wrapper script to use ghoscript function to compress PDF files.
    With this class you can compress and or fix a folder with (corrupt) PDF files.
    You can also use this class within your own scripts just do a
    import CompressPDF
    Compression levels:
        0: default
        1: prepress
        2: printer
        3: ebook
        4: screen
    Dependency: Ghostscript.
    On MacOSX install via command line `brew install ghostscript`.
    """
    def __init__(self, compress_level=0, show_info=False):
        self.compress_level = compress_level

        self.quality = {
            0: '/default',
            1: '/prepress',
            2: '/printer',
            3: '/ebook',
            4: '/screen'
        }

        self.show_compress_info = show_info

    def compress(self, file=None, new_file=None):
        """
        Function to compress PDF via Ghostscript command line interface
        :param file: old file that needs to be compressed
        :param new_file: new file that is commpressed
        :return: True or False, to do a cleanup when needed
        """
        try:
            if not os.path.isfile(file):
                print("Error: invalid path for input PDF file")
                sys.exit(1)

            # Check if file is a PDF by extension
            filename, file_extension = os.path.splitext(file)
            if file_extension != '.pdf':
                raise Exception("Error: input file is not a PDF")
                return False

            if self.show_compress_info:
                initial_size = os.path.getsize(file)

            subprocess.call(['gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
                            '-dPDFSETTINGS={}'.format(self.quality[self.compress_level]),
                            '-dNOPAUSE', '-dQUIET', '-dBATCH',
                            '-sOutputFile={}'.format(new_file),
                             file]
            )


            if self.show_compress_info:
                final_size = os.path.getsize(new_file)
                ratio = 1 - (final_size / initial_size)
                print("Compression by {0:.0%}.".format(ratio))
                print("Final file size is {0:.1f}MB".format(final_size / 1000000))

            return True
        except Exception as error:
            print('Caught this error: ' + repr(error))
        except subprocess.CalledProcessError as e:
            print("Unexpected error:".format(e.output))
            return False


