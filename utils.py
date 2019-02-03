import random
import numpy as np
import torch
import datetime
import collections
import sys
import io
import matplotlib.pyplot as plt
import json
import itertools
import os
from sklearn.model_selection import ParameterGrid


def json_pretty_dump(*args, **kwargs):
    """Call json.dump with some pretty-print options."""
    json.dump(*args, sort_keys=True, indent=4, separators=(',', ': '), **kwargs)


def seed_all(seed=None):
    """Set seed for random, numpy.random, torch and torch.cuda."""
    random.seed(seed)
    np.random.seed(seed)
    if seed is None:
        torch.manual_seed(np.random.randint(1e6))
        torch.cuda.manual_seed(np.random.randint(1e6))
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def seed_torch(seed=None):
    """Set seed for torch and torch.cuda."""
    if seed is None:
        torch.manual_seed(np.random.randint(1e6))
        torch.cuda.manual_seed(np.random.randint(1e6))
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def count_parameters(model):
    """Return the number of trainable (!) parameters in the model."""
    return sum(p.numel() for p in filter(lambda p: p.requires_grad, model.parameters()) if p.requires_grad)


def get_timestamp():
    """Get a string timestamp."""
    return str(datetime.datetime.now())


class StringLogger:
    """
    Write stdout to a string buffer in addition to the terminal.
    Adapted from https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    """

    def __init__(self):
        self.terminal = sys.stdout
        self.string_io = io.StringIO()

    def write(self, message):
        self.terminal.write(message)
        self.string_io.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    def get_output(self):
        return self.string_io.getvalue()

    def attach(self):
        sys.stdout = self
        return self

    def detach(self):
        sys.stdout = self.terminal
        return self


class History:
    """Store metrics for each epoch."""

    def __init__(self, desc='', params=None, log_stdout=False):
        super(History, self).__init__()
        self.values = collections.defaultdict(list)
        self.timestamp = get_timestamp()
        self.desc = desc
        self.params = params
        self.stdout = ''
        self.log_stdout = log_stdout
        # TODO: Test the log_stdout feature properly.
        if log_stdout:
            self.string_logger = StringLogger().attach()

    def _fetch_stdout(self):
        if self.log_stdout:
            self.stdout = self.string_logger.get_output()

    def __repr__(self):
        return 'History from {} with metrics {}'.format(self.timestamp, self.values.keys())

    def log(self, name, value, val_value=None, print_=False):
        self.values[name].append(value)
        if val_value is not None:
            val_name = 'val_' + name
            self.values[val_name].append(val_value)
            if print_:
                print(name + '    :', value)
                print(val_name + ':', val_value)
                print()
        elif print_:
            print(name + ':', value)
            print()

    def last(self, name=None):
        if name is not None:
            return self.values[name][-1]
        else:
            return {name: values[-1] for name, values in self.values.items()}

    def mean(self, name=None):
        if name is not None:
            return np.mean(self.values[name])
        else:
            return {name: np.mean(values) for name, values in self.values.items()}

    def plot(self, names=None, figsize=None, xlim=None, compare_to=None):
        if names is None:
            names = self.values.keys()

        # Remove names that start with val_ and test_ (these will be printed dotted later).
        cleaned_names = []
        for name in names:
            if name.startswith('val_'):
                if name[4:] not in cleaned_names:
                    cleaned_names.append(name[4:])
            elif name.startswith('test_'):
                if name[5:] not in cleaned_names:
                    cleaned_names.append(name[5:])
            elif name not in cleaned_names:
                cleaned_names.append(name)
        names = cleaned_names

        if compare_to is None:
            compare_to = []
        else:
            # If one of the elements in compare_to is a filename, load it as a History object.
            for i in range(len(compare_to)):
                try:
                    compare_to[i] = History.load(compare_to[i])
                except:
                    compare_to[i] = None
            compare_to = list(filter(lambda x: x is not None, compare_to))

        histories = [self] + compare_to
        lines = []

        fig, axes = plt.subplots(len(names), sharex=True, figsize=figsize)

        for name, ax in zip(names, axes):
            plt.sca(ax)
            plt.grid()
            plt.ylabel(name)
            if xlim:
                plt.xlim(*xlim)

            color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])  # default mpl colors
            for h in histories:
                color = next(color_cycle)
                if name in h.values:
                    lines.append(plt.plot(h.values[name], color=color)[0])
                if 'val_' + name in h.values:
                    plt.plot(h.values['val_' + name], color=color, linestyle='--')
                if 'test_' + name in h.values:
                    plt.plot(h.values['test_' + name], color=color, linestyle=':')

        fig.legend(lines, [h.params for h in histories])

        axes[-1].set_xlabel('Epoch')

    @staticmethod
    def plot_from_file(filenames, **kwargs):
        print(filenames)
        History.load(filenames[0]).plot(compare_to=filenames[1:], **kwargs)

    def save(self, filename):
        if self.log_stdout:
            self._fetch_stdout()
        with open(filename, 'w') as f:
            json_pretty_dump({k: v for k, v in self.__dict__.items() if k in ['values', 'timestamp', 'desc', 'params', 'stdout', 'log_stdout']}, f)

    @staticmethod
    def load(filename):
        with open(filename) as f:
            contents = json.load(f)
        h = History()#log_stdout=contents['log_stdout'])
        h.timestamp = contents['timestamp']
        h.desc = contents['desc']
        h.values.update(contents['values'])  # use update here so that `values` is a defaultdict
        h.params = contents['params']
        #h.stdout = contents['stdout']
        return h


class GridSearch:

    # TODO: Add multiple runs per parameter config.
    def __init__(self, log_dir, param_grid=None, resume=True, log_stdout=False):
        self.log_dir = log_dir
        self.log_stdout = log_stdout
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if resume and os.path.exists(self._get_jobs_filename()):  # resume
            with open(self._get_jobs_filename()) as f:
                self.jobs = json.load(f)
        else:  # init a new run
            if param_grid is None:
                raise ValueError('Could not resume a previous run, param_grid must not be None')
            self.jobs = []
            for i, params in enumerate(ParameterGrid(param_grid)):
                # Each job is a dict with fields `index`, `params` (dict of
                # hyperparameters), `status` (one of: 'not started', 'running', 'done').
                self.jobs.append({'index': i, 'params': params, 'status': 'not started', 'history': ''})
            self._write_jobs_file()

    def __len__(self):
        return len(self.jobs)

    def __repr__(self):
        return 'GridSearch with {} jobs:\n{}'.format(len(self.jobs), '\n'.join(str(job) for job in self.jobs))

    def _write_jobs_file(self):
        with open(self._get_jobs_filename(), 'w') as f:
            json_pretty_dump(self.jobs, f)

    def _get_history_filename(self, job):
        return os.path.join(self.log_dir, str(job['index']) + '.json')

    def load_all_histories(self):
        """Loads the History objects of all jobs and returns a dict of job index: history pairs."""
        # TODO: Maybe integrate this with the jobs array better.
        # TODO: Maybe make an extra class Histories, which holds multiple History objects and has a function plot, max/min, etc.
        # TODO: Maybe store path to history and best model in the jobs object, so that it's easier to access and load them.
        histories = {}
        for job in self.jobs:
            try:
                histories[job['index']] = History.load(self._get_history_filename(job))
            except IOError:
                pass
        return histories

    def get_best_value(self, metric):
        """Find the run with the best value for a given metric. Return the index of this run, the epoch with the best value and the best value itself."""
        # TODO: Add parameter mode, which can be 'max' or 'min'.

        best_value = None
        best_index = None
        best_epoch = None

        for index, history in self.load_all_histories().iteritems():
            value = np.max(history.values[metric])
            if best_value is None or value > best_value:
                best_value = value
                best_index = index
                best_epoch = np.argmax(history.values[metric])

        return best_index, best_epoch, best_value

    def _get_jobs_filename(self):
        return os.path.join(self.log_dir, 'jobs.json')

    def run(self, func):
        print('Starting hyperparameter optimization (logging to {})'.format(self.log_dir))
        print('=' * 80)
        for job in self.jobs:
            if job['status'] in ['done', 'running']:
                print('Skipping run {} with parameters {}, is already done or running'.format(job['index'], job['params']))
            else:
                if job['status'] == 'not started':
                    print('Starting run {} with parameters {}'.format(job['index'], job['params']))
                #else:
                #    print('Restarting run {} with parameters {}'.format(job['index'], job['params']))
                #    print('WARNING: This run was already started before, so files will be overwritten')
                print('=' * 80)

                history = History(desc='Run {} of hyperparameter search'.format(job['index']), params=job['params'], log_stdout=self.log_stdout)
                job['status'] = 'running'
                self._write_jobs_file()
                func(job['params'], job['index'], history)
                history.save(self._get_history_filename((job)))
                job['status'] = 'done'
                self._write_jobs_file()

    def plot(self, filter_params=None, **kwargs):
        jobs_to_plot = filter(lambda job: job['status'] in ['done', 'running'], self.jobs)
        if filter_params is not None:
            # TODO: Make it so that values can be either a single value or an iterable of values.s
            for param, values in filter_params.items():
                jobs_to_plot = filter(lambda job: job['params'][param] in values, jobs_to_plot)
        History.plot_from_file(list(map(self._get_history_filename, jobs_to_plot)), **kwargs)
