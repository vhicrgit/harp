import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from scipy.stats import mstats
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
from os.path import dirname, abspath, exists, join, isfile, expanduser
from os import makedirs, system, environ
from socket import gethostname
from collections import OrderedDict
import klepto
import subprocess
from threading import Timer
from time import time
import datetime, pytz
import re
import requests
import random
import pickle
import signal
import numpy as np
import scipy.sparse as sp
import sys
from torch_geometric.nn import GlobalAttention
from nn_att import MultiMyGlobalAttention
import json
import shutil


def check_nx_version():
    nxvg = '2.2'
    nxva = nx.__version__
    if nxvg != nxva:
        raise RuntimeError(
            'Wrong networkx version! Need {} instead of {}'.format(nxvg, nxva))

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# Always check the version first.
# check_nx_version()


def get_root_path():
    return dirname(dirname(abspath(__file__)))


def get_save_path():
    return join(get_root_path(), 'save')


def get_src_path():
    return join(get_root_path(), 'src')


def create_dir_if_not_exists(dir):
    if not exists(dir):
        makedirs(dir)

def _get_y(data):
    return getattr(data, FLAGS.target.replace('-', '_'))

def _get_y_with_target(data, target):
    return getattr(data, target.replace('-', '_'))

def _get_y_multi_obj(data):
    assert(isinstance(FLAGS.target, list))
    y_list = [getattr(data, t.replace('-', '_')) for t in FLAGS.target]
    return y_list

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def save(obj, filepath, print_msg=True, use_klepto=True):
    if type(obj) is not dict and type(obj) is not OrderedDict:
        raise ValueError('Can only save a dict or OrderedDict'
                         ' NOT {}'.format(type(obj)))
    fp = proc_filepath(filepath, ext='.klepto' if use_klepto else '.pickle')
    if use_klepto:
        create_dir_if_not_exists(dirname(filepath))
        save_klepto(obj, fp, print_msg)
    else:
        save_pickle(obj, fp, print_msg)


def load(filepath, print_msg=True, saver=None):
    fp = proc_filepath(filepath)
    if isfile(fp):
        return load_klepto(fp, print_msg, saver=saver)
    elif print_msg:
        print('Trying to load but no file {}'.format(fp))


def save_klepto(dic, filepath, print_msg):
    if print_msg:
        print('Saving to {}'.format(filepath))
    klepto.archives.file_archive(filepath, dict=dic).dump()


def load_klepto(filepath, print_msg, saver=None):
    rtn = klepto.archives.file_archive(filepath)
    rtn.load()
    if print_msg:
        print('Loaded from {}'.format(filepath))
        if saver:
            saver.log_info('Loaded from {}'.format(filepath))
    return rtn


def save_pickle(dic, filepath, print_msg=True):
    if print_msg:
        print('Saving to {}'.format(filepath))
    with open(filepath, 'wb') as handle:
        if sys.version_info.major < 3:  # python 2
            pickle.dump(dic, handle)
        elif sys.version_info >= (3, 4):  # qilin & feilong --> 3.4
            pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise NotImplementedError()


def load_pickle(filepath, print_msg=True):
    fp = proc_filepath(filepath, '.pickle')
    if isfile(fp):
        with open(fp, 'rb') as handle:
            pickle_data = pickle.load(handle)
            return pickle_data
    elif print_msg:
        print('No file {}'.format(fp))
        



def proc_filepath(filepath, ext='.klepto'):
    if type(filepath) is not str:
        raise RuntimeError('Did you pass a file path to this function?')
    return append_ext_to_filepath(ext, filepath)


def append_ext_to_filepath(ext, fp):
    if not fp.endswith(ext):
        fp += ext
    return fp


def sorted_nicely(l, reverse=False):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        if type(s) is not str:
            raise ValueError('{} must be a string in l: {}'.format(s, l))
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    rtn = sorted(l, key=alphanum_key)
    if reverse:
        rtn = reversed(rtn)
    return rtn


global_exec_print = True


def exec_turnoff_print():
    global global_exec_print
    global_exec_print = False


def exec_turnon_print():
    global global_exec_print
    global_exec_print = True


def global_turnoff_print():
    import sys, os
    sys.stdout = open(os.devnull, 'w')


def global_turnon_print():
    import sys
    sys.stdout = sys.__stdout__


def exec_cmd(cmd, timeout=None, exec_print=True):
    '''
    TODO: take a look at

        def _run_prog(self, prog='nop', args=''):
        """Apply graphviz program to graph and return the result as a string.

        >>> A = AGraph()
        >>> s = A._run_prog() # doctest: +SKIP
        >>> s = A._run_prog(prog='acyclic') # doctest: +SKIP

        Use keyword args to add additional arguments to graphviz programs.
        """
        runprog = r'"%s"' % self._get_prog(prog)
        cmd = ' '.join([runprog, args])
        dotargs = shlex.split(cmd)
        p = subprocess.Popen(dotargs,
                             shell=False,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             close_fds=False)
        (child_stdin,
         child_stdout,
         child_stderr) = (p.stdin, p.stdout, p.stderr)
        # Use threading to avoid blocking
        data = []
        errors = []
        threads = [PipeReader(data, child_stdout),
                   PipeReader(errors, child_stderr)]
        for t in threads:
            t.start()

        self.write(child_stdin)
        child_stdin.close()

        for t in threads:
            t.join()
        p.wait()

        if not data:
            raise IOError(b"".join(errors).decode(self.encoding))

        if len(errors) > 0:
            warnings.warn(b"".join(errors).decode(self.encoding), RuntimeWarning)
        return b"".join(data)

        taken from /home/yba/.local/lib/python3.7/site-packages/pygraphviz/agraph.py
    '''
    global global_exec_print
    if not timeout:
        if global_exec_print and exec_print:
            print(cmd)
        else:
            cmd += ' > /dev/null'
        system(cmd)
        return True  # finished
    else:
        def kill_proc(proc, timeout_dict):
            timeout_dict["value"] = True
            proc.kill()

        def run(cmd, timeout_sec):
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            timeout_dict = {"value": False}
            timer = Timer(timeout_sec, kill_proc, [proc, timeout_dict])
            timer.start()
            stdout, stderr = proc.communicate()
            timer.cancel()
            return proc.returncode, stdout.decode("utf-8"), \
                   stderr.decode("utf-8"), timeout_dict["value"]

        if global_exec_print and exec_print:
            print('Timed cmd {} sec(s) {}'.format(timeout, cmd))
        _, _, _, timeout_happened = run(cmd, timeout)
        if global_exec_print and exec_print:
            print('timeout_happened?', timeout_happened)
        return not timeout_happened


tstamp = None


def get_ts():
    global tstamp
    if not tstamp:
        tstamp = get_current_ts()
    return tstamp


def get_current_ts(zone='US/Pacific'):
    return datetime.datetime.now(pytz.timezone(zone)).strftime(
        '%Y-%m-%dT%H-%M-%S.%f')


class timeout:
    """
    https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
    """

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def get_user():
    try:
        home_user = expanduser("~").split('/')[-1]
    except:
        home_user = 'user'
    return home_user


def get_host():
    host = environ.get('HOSTNAME')
    if host is not None:
        return host
    return gethostname()


def slack_notify(message):
    # posts to slack channel #chris-notify-test
    url = 'https://hooks.slack.com/services/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'
    data = {'text': message}
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    requests.post(url, headers=headers, json=data)


def assert_valid_nid(nid, g):
    assert type(nid) is int and (0 <= nid < g.number_of_nodes())


def assert_0_based_nids(g):
    for i, (n, ndata) in enumerate(sorted(g.nodes(data=True))):
        assert_valid_nid(n, g)
        assert i == n  # 0-based consecutive node ids


def format_str_list(sl):
    assert type(sl) is list
    if not sl:
        return 'None'
    else:
        return ','.join(sl)




class OurTimer(object):
    def __init__(self):
        self.t = time()
        self.durations_log = OrderedDict()

    def time_and_clear(self, log_str='', only_seconds=False):
        duration = self._get_duration_and_reset()
        if log_str:
            if log_str in self.durations_log:
                raise ValueError('log_str {} already in log {}'.format(
                    log_str, self.durations_log))
            self.durations_log[log_str] = duration
        if only_seconds:
            rtn = duration
        else:
            rtn = format_seconds(duration)
        print(log_str, '\t\t', rtn)
        return rtn

    def start_timing(self):
        self.t = time()

    def print_durations_log(self):
        print('Timer log', '*' * 50)
        rtn = []
        tot_duration = sum([sec for sec in self.durations_log.values()])
        print('Total duration:', format_seconds(tot_duration))
        lss = np.max([len(s) for s in self.durations_log.keys()])
        for log_str, duration in self.durations_log.items():
            s = '{0}{1} : {2} ({3:.2%})'.format(
                log_str, ' ' * (lss - len(log_str)), format_seconds(duration),
                         duration / tot_duration)
            rtn.append(s)
            print(s)
        print('Timer log', '*' * 50)
        self.durations_log = OrderedDict()  # reset
        return rtn

    def _get_duration_and_reset(self):
        now = time()
        duration = now - self.t
        self.t = now
        return duration

    def get_duration(self):
        now = time()
        duration = now - self.t
        return duration

    def reset(self):
        self.t = time()


def format_seconds(seconds):
    """
    https://stackoverflow.com/questions/538666/python-format-timedelta-to-string
    """
    periods = [
        ('year', 60 * 60 * 24 * 365),
        ('month', 60 * 60 * 24 * 30),
        ('day', 60 * 60 * 24),
        ('hour', 60 * 60),
        ('min', 60),
        ('sec', 1)
    ]

    if seconds <= 1:
        return '{:.3f} msecs'.format(seconds * 1000)

    strings = []
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            if period_name == 'sec':
                period_value = seconds
                has_s = 's'
            else:
                period_value, seconds = divmod(seconds, period_seconds)
                has_s = 's' if period_value > 1 else ''
            strings.append('{:.3f} {}{}'.format(period_value, period_name, has_s))

    return ', '.join(strings)


def random_w_replacement(input_list, k=1):
    return [random.choice(input_list) for _ in range(k)]


def get_sparse_mat(a2b, a2idx, b2idx):
    n = len(a2idx)
    m = len(b2idx)
    assoc = np.zeros((n, m))
    for a, b_assoc in a2b.items():
        if a not in a2idx:
            continue
        for b in b_assoc:
            if b not in b2idx:
                continue
            if n == m:
                assoc[a2idx[a], b2idx[b]] = assoc[b2idx[b], a2idx[a]] = 1.
            else:
                assoc[a2idx[a], b2idx[b]] = 1
    assoc = sp.csr_matrix(assoc)
    return assoc


def prompt(str, options=None):
    while True:
        t = input(str + ' ')
        if options:
            if t in options:
                return t
        else:
            return t


def prompt_get_cpu():
    from os import cpu_count
    while True:
        num_cpu = prompt(
            '{} cpus available. How many do you want?'.format( \
                cpu_count()))
        num_cpu = parse_as_int(num_cpu)
        if num_cpu and num_cpu <= cpu_count():
            return num_cpu


def parse_as_int(s):
    try:
        rtn = int(s)
        return rtn
    except ValueError:
        return None


computer_name = None


def prompt_get_computer_name():
    global computer_name
    if not computer_name:
        computer_name = prompt('What is the computer name?')
    return computer_name


def node_has_type_attrib(g):
    for (n, d) in g.nodes(data=True):
        if 'type' in d:  # TODO: needs to be fixed
            return True
    return False


def print_g(label, g):
    print(f'{label} {g.number_of_nodes()} nodes {g.number_of_edges()} edges')


class MLP(nn.Module):
    '''mlp can specify number of hidden layers and hidden layer channels'''

    def __init__(self, input_dim, output_dim, activation_type='relu', num_hidden_lyr=2,
                 hidden_channels=None, bn=False):
        super().__init__()
        self.out_dim = output_dim
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the lengh of hidden_channels")
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.activation = create_act(activation_type)
        self.layers = nn.ModuleList(list(
            map(self.weight_init, [nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                                   for i in range(len(self.layer_channels) - 1)])))
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(output_dim)

    def weight_init(self, m):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        return m

    def forward(self, x):
        layer_inputs = [x]
        for layer in self.layers:
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                layer_inputs.append(self.activation(layer(input)))
        # model.store_layer_output(self, layer_inputs[-1])
        if self.bn:
            layer_inputs[-1] = self.bn(layer_inputs[-1])
        return layer_inputs[-1]
    
    
class MLP_multi_objective(nn.Module):
    '''mlp can specify number of hidden layers and hidden layer channels'''

    def __init__(self, input_dim, output_dim, activation_type='relu', objectives=None, num_common_lyr=0,
                 hidden_channels=None, bn=False):
        super().__init__()
        self.out_dim = output_dim

        if hidden_channels:
            self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        else:
            self.layer_channels = [input_dim] + [output_dim]
        self.activation = create_act(activation_type)
        self.num_common_lyr = num_common_lyr
        self.layers_common = nn.ModuleList(list(
            map(self.weight_init, [nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                                   for i in range(num_common_lyr)])
            ))
        self.MLP_heads = nn.ModuleDict()
        self.objectives = objectives
        for obj in self.objectives:
            self.MLP_heads[obj] = nn.ModuleList(list(
                map(self.weight_init, [nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                                    for i in range(self.num_common_lyr, len(self.layer_channels) - 1)])
                ))
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(output_dim)

    def weight_init(self, m):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        return m

    def forward(self, x):
        layer_inputs = [x]
        for layer in self.layers_common:
            input = layer_inputs[-1]
            ## always apply activation on common layers
            layer_inputs.append(self.activation(layer(input)))
        out_common_layers = layer_inputs[-1]
        out_MLP = {}
        for obj in self.objectives:
            out_MLP[obj] = out_common_layers
            for layer_ind, layer in enumerate(self.MLP_heads[obj]):
                if layer_ind + self.num_common_lyr == len(self.layer_channels) - 1:
                    out_MLP[obj] = layer(out_MLP[obj])
                    if self.bn:
                        out_MLP[obj] = self.bn(out_MLP[obj])
                else:
                    out_MLP[obj] = self.activation(layer(out_MLP[obj]))
        # model.store_layer_output(self, layer_inputs[-1])
        
        return out_MLP
    
    
class MultiAttension_MLP_multi_objective(nn.Module):
    '''mlp can specify number of hidden layers and hidden layer channels'''

    def __init__(self, input_dim, output_dim, gate_nn_P, gate_nn_pseudo_B, activation_type='relu', objectives=None, 
                 hidden_channels=None, bn=False):
        super().__init__()
        self.gate_nn = Sequential(Linear(64, 64), ReLU(), Linear(64, 1))
        self.out_dim = output_dim

        if hidden_channels:
            self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        else:
            self.layer_channels = [input_dim] + [output_dim]
        self.activation = create_act(activation_type)

        self.MLP_heads = nn.ModuleDict()
        self.objectives = objectives
        for obj in self.objectives:
            self.MLP_heads[obj] = nn.ModuleList()
            self.MLP_heads[obj].append(MultiMyGlobalAttention(gate_nn_P, gate_nn_pseudo_B))
            self.MLP_heads[obj].extend(
                map(self.weight_init, [
                    nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                    for i in range(len(self.layer_channels) - 1)
                ])
            )
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(output_dim)

    def weight_init(self, m):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        return m

    def forward(self, x, batch, X_pseudonids=None):
        # 注意力可视化
        att_dict = dict()
        out_MLP = {}
        for obj in self.objectives:
            out_MLP[obj] = x
            att_dict[obj] = dict()
            for layer_ind, layer in enumerate(self.MLP_heads[obj]):
                if layer_ind == 0:
                    # 第一层是GlobalAttention
                    out_MLP[obj], pseudo_att, P_att = layer(out_MLP[obj], batch, X_pseudonids)
                    att_dict[obj]['pseudo'] = pseudo_att
                    att_dict[obj]['P'] = P_att
                    if self.bn:
                        out_MLP[obj] = self.bn(out_MLP[obj]) 
                elif layer_ind == len(self.layer_channels) - 1:
                    out_MLP[obj] = layer(out_MLP[obj])
                    if self.bn:
                        out_MLP[obj] = self.bn(out_MLP[obj])
                else:
                    out_MLP[obj] = self.activation(layer(out_MLP[obj]))
        # model.store_layer_output(self, layer_inputs[-1])
        
        return out_MLP, att_dict


def create_act(act, num_parameters=None):
    if act == 'relu' or act == 'ReLU':
        return nn.ReLU()
    elif act == 'prelu':
        return nn.PReLU(num_parameters)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'identity' or act == 'None':
        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    if act == 'elu' or act == 'elu+1':
        return nn.ELU()
    else:
        raise ValueError('Unknown activation function {}'.format(act))


def print_stats(li, name, saver=None):
    stats = OrderedDict()
    stats['#'] = len(li)
    stats['Avg'] = np.mean(li)
    stats['Std'] = np.std(li)
    stats['Min'] = np.min(li)
    stats['Max'] = np.max(li)
    
    if saver: saver.log_info(name)
    else: print(name)
    for k, v in stats.items():
        if saver: saver.log_info(f'\t{k}:\t{v}')
        else: print(f'\t{k}:\t{v}')


def plot_dist(data, label, save_dir, saver=None, analyze_dist=True, bins=None):
    if analyze_dist:
        _analyze_dist(saver, label, data)
    fn = f'distribution_{label}.png'
    plt.figure()
    sns.set()
    ax = sns.distplot(data, bins=bins, axlabel=label, kde=False, norm_hist=False)
    plt.xlabel(label)
    ax.figure.savefig(join(save_dir, fn))
    plt.close()


def _analyze_dist(saver, label, data):
    if saver is None:
        func = print
    else:
        func = saver.log_info
    func(f'--- Analyzing distribution of {label} (len={len(data)})')
    if np.isnan(np.sum(data)):
        func(f'{label} has nan')
    probs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999]
    quantiles = mstats.mquantiles(data, prob=probs)
    func(f'{label} {len(data)}')
    s = '\t'.join([str(x) for x in probs])
    func(f'\tprob     \t {s}')
    s = '\t'.join(['{:.2f}'.format(x) for x in quantiles])
    func(f'\tquantiles\t {s}')
    func(f'\tnp.min(data)\t {np.min(data)}')
    func(f'\tnp.max(data)\t {np.max(data)}')
    func(f'\tnp.mean(data)\t {np.mean(data)}')
    func(f'\tnp.std(data)\t {np.std(data)}')


def get_model_info_as_str(FLAGS):
    rtn = []
    d = vars(FLAGS)
    for k in d.keys():
        v = str(d[k])
        if k == 'dataset_list':
            s = '{0:26} : {1}'.format(k, v)
            rtn.append(s)
        else:
            vsplit = v.split(',')
            assert len(vsplit) >= 1
            for i, vs in enumerate(vsplit):
                if i == 0:
                    ks = k
                else:
                    ks = ''
                if i != len(vsplit) - 1:
                    vs = vs + ','
                s = '{0:26} : {1}'.format(ks, vs)
                rtn.append(s)
    rtn.append('{0:26} : {1}'.format('ts', get_ts()))
    return '\n'.join(rtn)


def extract_config_code():
    with open(join(get_src_path(), 'config.py')) as f:
        return f.read()


def plot_scatter_line(data_dict, label, save_dir):
    fn = f'scatter_{label}_iterations.png'
    ss = ['rs-', 'b^-', 'g^-', 'c^-', 'm^-', 'ko-', 'yo-']
    cs = [s[0] for s in ss]
    plt.figure()
    i = 0

    for line_name, data_dict_elt in sorted(data_dict.items()):
        x_li, y_li = [], []
        for x in data_dict_elt['incumbent_data']:
            x_li.append(x[1])
            y_li.append(x[0])
        plt.scatter(np.array(x_li), np.array(y_li), label=line_name, color=cs[i % len(cs)])
        plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
        i += 1

    plt.title(label)
    plt.grid(True)
    plt.legend()
    plt.axis('on')
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()

    plt.figure()
    fn = f'scatter_{label}_time.png'
    i = 0
    for line_name, data_dict_elt in sorted(data_dict.items()):
        x_li = [x[2] for x in data_dict_elt['incumbent_data']]
        y_li = [x[0] for x in data_dict_elt['incumbent_data']]
        plt.scatter(np.array(x_li), np.array(y_li), label=line_name, color=cs[i % len(cs)])
        plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
        i += 1

    plt.title(label)
    plt.grid(True)
    plt.legend()
    plt.axis('on')
    plt.savefig(join(save_dir, fn), bbox_inches='tight')


POINTS_MARKERS = ['o', '.', '.', '.', '', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
POINTS_COLORS = ["red","green","blue","blue", "blue", "yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"]
def plot_points(points_dict, label, save_dir):
    i = 0
    for pname, points in points_dict.items():
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        plt.plot(xs, ys, POINTS_MARKERS[i % len(POINTS_MARKERS)],
                 color=POINTS_COLORS[i % len(POINTS_COLORS)],
                 label=f'{pname}_{label}')
        i += 1
    plt.legend(loc='best')
    fn = f'points_{label}.png'
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()
    
def multi_plot_dimension(target_list):
    num_figure = len(target_list)
    if num_figure == 1:
        y_dim = 1
        x_dim = 1
    elif num_figure == 2:
        y_dim = 1
        x_dim = 2
    elif num_figure == 3:
        y_dim = 1
        x_dim = 3
    elif num_figure == 4:
        y_dim = 2
        x_dim = 2
    elif num_figure == 5 or num_figure == 6:
        y_dim = 2
        x_dim = 3  
    return num_figure, x_dim, y_dim 

def plot_scatter_with_subplot(points_dict_multi_target, label, save_dir, target_list, connected = True):
    i = 0
    num_figure, x_dim, y_dim = multi_plot_dimension(target_list) 
    points_dict = {}
    ss = ['r-', 'b-', 'g-', 'c-', 'm-', 'k-', 'y-', 'w-']
    cs = [s[0] for s in ss]
    fig = plt.figure()
    fig.set_figheight(18)
    fig.set_figwidth(24)
    m = {'p': 'o', 't': 'x'}
    for idx, target in enumerate(target_list):
        points_dict[f'p'] = points_dict_multi_target[target]['pred']
        points_dict[f't'] = points_dict_multi_target[target]['true']
        ax=plt.subplot(y_dim, x_dim, idx+1)
        ax.set_facecolor('xkcd:gray')
        i = 0
        for pname, points_ in points_dict.items(): # dict (true/pred) of dict (name: points)
            for gname, points in points_.items():
                x_li = [str(int(point[0])) for point in sorted(points)]
                y_li = [round(float(point[1]), 2) for point in sorted(points)]
                plt.scatter(np.array(x_li), np.array(y_li), label=f'{gname}-{pname}', color=cs[i % len(cs)], marker=m[pname])
                if connected:
                    plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
                i += 1    
        plt.legend(loc='best')
        plt.title(f'{target}')
        plt.grid(True)
        plt.axis('on')
        points_dict = {}   
    
    plt.suptitle(f'{label}')    
    fn = f'points_{label}.png'
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()
    
def plot_loss_trend(epochs, train_losses, val_losses, test_losses, save_dir, file_name='losses.png'):
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    if len(val_losses) > 0:
        plt.plot(epochs, val_losses, 'b', label='Validation loss')
    if len(test_losses) > 0:
        plt.plot(epochs, test_losses, 'r', label='Testing loss')
    plt.title('Training, Validation, and Testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(join(save_dir, file_name), bbox_inches='tight')
    plt.close()

def plot_scatter_with_subplot_trend(points_dict_multi_target, label, save_dir, target_list, connected = True):
    i = 0
    num_figure, x_dim, y_dim = multi_plot_dimension(target_list) 
    num_figure, x_dim, y_dim = 1, 1, 1
    points_dict = {}
    ss = ['r-', 'b-', 'g-', 'c-', 'm-', 'k-', 'y-', 'w-']
    cs = [s[0] for s in ss]
    fig = plt.figure()
    fig.set_figheight(18)
    fig.set_figwidth(24)
    m = {'p': 'o', 't': 'x'}
    for idx, target in enumerate(target_list):
        if 'perf' not in target:
            continue
        points_dict[f't'] = points_dict_multi_target[target]['true']
        ax=plt.subplot(y_dim, x_dim, idx+1)
        ax.set_facecolor('xkcd:gray')
        i = 0
        for pname, points_ in points_dict.items(): # dict (true/pred) of dict (name: points)
            for gname, points in points_.items():
                if len(points) <= 1:
                    continue
                fig = plt.figure()
                x_li = [str(int(point[0])) for point in sorted(points)]
                y_li = [round(float(point[1]), 2) for point in sorted(points)]
                plt.scatter(np.array(x_li), np.array(y_li), label=f'{gname}-{pname}', color=cs[i % len(cs)], marker=m[pname])
                if connected:
                    plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
                i += 1    
                plt.title(f'{target}')
                plt.grid(True)
                plt.axis('on')
        
                plt.suptitle(f'{label}')    
                fn = f'points_{gname}.png'
                plt.savefig(join(save_dir, fn), bbox_inches='tight')
                plt.close()
    
def plot_points_with_subplot(points_dict_multi_target, label, save_dir, target_list, use_sigma=False):
    i = 0
    num_figure, x_dim, y_dim = multi_plot_dimension(target_list) 
    points_dict = {}
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(22)
    for idx, target in enumerate(target_list):
        points_dict[f'pred_points'] = points_dict_multi_target[target]['pred']
        points_dict[f'true_points'] = points_dict_multi_target[target]['true']
        if use_sigma:
            points_dict[f'mu-sigma_points'] = points_dict_multi_target[target]['sigma_mu']
            points_dict[f'mu+sigma_points'] = points_dict_multi_target[target]['sigma+mu']
        plt.subplot(y_dim, x_dim, idx+1)
        i = 0
        for pname, points in points_dict.items():
            xs = [point[0] for point in sorted(points)]
            ys = [point[1] for point in sorted(points)]
            plt.plot(xs, ys, POINTS_MARKERS[i % len(POINTS_MARKERS)],
                    color=POINTS_COLORS[i % len(POINTS_COLORS)],
                    label=f'{pname}')
            i += 1    
        plt.legend(loc='best')
        plt.title(f'{target}')
        points_dict = {}
    plt.suptitle(f'{label}')    
    fn = f'points_{label}.png'
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()


def plot_points_with_subplot_sigma(points_dict_multi_target, label, save_dir, target_list, use_sigma=False):
    i = 0
    num_figure, x_dim, y_dim = multi_plot_dimension(target_list) 
    points_dict = OrderedDict()
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(22)
    for idx, target in enumerate(target_list):
        points_dict[f'error_points'] = points_dict_multi_target[target]['error']
        if use_sigma:
            points_dict[f'sigma_points'] = points_dict_multi_target[target]['sigma']
        plt.subplot(y_dim, x_dim, idx+1)
        i = 1
        for pname, points in points_dict.items():
            xs = [point[0] for point in sorted(points)]
            ys = [point[1] for point in sorted(points)]
            if i == 2:
                i += 2
            plt.plot(xs, ys, '',
                    color=POINTS_COLORS[i % len(POINTS_COLORS)],
                    label=f'{pname}')
            i += 1    
        plt.legend(loc='best')
        plt.title(f'{target}')
        points_dict = {}
    plt.suptitle(f'{label}')    
    fn = f'points_{label}_sigma_error.png'
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()
    
def plot_lr_trend(lrs,  epoch, save_dir):
    minibatches = [i+1 for i in range(len(lrs))]
    plt.plot(minibatches, lrs, 'b', label='Learning Rate')
    plt.title(f'Learning Rate at epoch {epoch}')
    plt.xlabel('minibatch number')
    plt.ylabel('learning rate')
    plt.legend()
    plt.savefig(join(save_dir, f'lr_{epoch}.png'), bbox_inches='tight')
    plt.close()

def plot_models_per_graph(save_dir, name, graph_types, plot_data, multi_target):
    for target in multi_target:
        fig = plt.figure()
        fig.set_figheight(9.6)
        fig.set_figwidth(21.6)
        for idx, graph_ in enumerate(graph_types):
            plt.subplot(3, 1, idx+1)
            cur_data = plot_data[graph_][target]
            plt.scatter(np.arange(1, len(cur_data)+1, 1), np.array(cur_data), marker='.', label=f'{graph_}')
            plt.legend(loc='best')
        plt.suptitle(f'{name}-{target}')
        plt.savefig(join(save_dir, f'{name}-{target}.png'))
        plt.close()



# 可视化
visual_save_dir = '/root/autodl-tmp/kaggle/save/harp/visual'
labels_map_dir = '/root/autodl-tmp/kaggle/save/harp/idx2text.json'
batch_size = 64

def clear_directory(directory_path):
    if exists(directory_path):
        shutil.rmtree(directory_path)
        makedirs(directory_path)
        
class SingleData():
    def __init__(self, nodes, edges, labels:map, kernel_name, design_name, attention_weights, is_pseudo=False):
        self.nodes = nodes
        self.edges = edges
        self.labels = labels
        self.kernel_name = kernel_name
        self.design_name = design_name
        self.attention_weights = attention_weights
        self.is_pseudo = is_pseudo
        
def batch_division(nodes, edges, batch, kernel_name, design_name, attention_weights, batch_size=batch_size):
    # nodes: list
    # nodes = nodes.tolist()
    edges = edges.tolist()
    batch = batch.tolist()
    batch_size = max(batch)+1 # 一个batch不一定总能凑够batchsize
    start_idx = [batch.index(i) for i in range(batch_size)]
    attention_weights_P = attention_weights['P'].view(-1).tolist()
    attention_weights_pseudo = attention_weights['pseudo'].view(-1).tolist()
    with open(labels_map_dir, 'r') as f:
        labels_map = json.load(f)

    single_data_list = []
    # 全部生成图耗时太长，只关注部分kernel
    trace = ['spmv-ellpack', 'bicg-medium', 'mvt-medium']
    for i in range(batch_size):
        if not kernel_name[i] in trace:
            continue
        if i == batch_size-1:
            nodes_ = nodes[start_idx[i]:]
            edges_start = [e for e in edges[0] if e in range(start_idx[i], len(nodes))]
            edges_end = [e for e in edges[1] if e in range(start_idx[i], len(nodes))]
            attention_weights_P_ = attention_weights_P[start_idx[i]:]
            attention_weights_pseudo_ = attention_weights_pseudo[start_idx[i]:]
        else:
            nodes_ = nodes[start_idx[i]:start_idx[i+1]]
            edges_start = [e for e in edges[0] if e in range(start_idx[i], start_idx[i+1])]
            edges_end = [e for e in edges[1] if e in range(start_idx[i], start_idx[i+1])]
            attention_weights_P_ = attention_weights_P[start_idx[i]:start_idx[i+1]]
            attention_weights_pseudo_ = attention_weights_pseudo[start_idx[i]:start_idx[i+1]]

        nodes_ = [n-start_idx[i] for n in nodes_]
        edges_start = [e-start_idx[i] for e in edges_start]
        edges_end = [e-start_idx[i] for e in edges_end]
        
        # assert len(edges_start) == len(edges_end)
        # assert len(nodes_) == len(attention_weights_P_)
        # assert len(nodes_) == len(attention_weights_pseudo_)
        edges_ = [[edges_start[i], edges_end[i]] for i in range(len(edges_start))]
        labels_map_ = {int(k): v for k, v in labels_map[kernel_name[i]].items()}

        # assert len(nodes_) == len(labels_map_.keys())
        single_data_list.append(SingleData(nodes_, edges_, labels_map_, kernel_name[i], design_name[i], attention_weights_P_, False))
        single_data_list.append(SingleData(nodes_, edges_, labels_map_, kernel_name[i], design_name[i], attention_weights_pseudo_, True))

    return single_data_list


def batch_division2(data, attention_weights):
    # nodes: list
    # nodes = nodes.tolist()
    data_list = data.to_data_list()
    # labels
    with open(labels_map_dir, 'r') as f:
        labels_map = json.load(f)
    # attention
    batch = data.batch.tolist()
    batch_size = max(batch)+1 # 一个batch不一定总能凑够batchsize
    start_idx = [batch.index(i) for i in range(batch_size)]
    attention_weights_P = attention_weights['P'].view(-1).tolist()
    attention_weights_pseudo = attention_weights['pseudo'].view(-1).tolist()
    # 处理数据
    single_data_list = []
    trace = ['spmv-ellpack', 'bicg-medium', 'mvt-medium']
    for i, data in enumerate(data_list):
        edges, kernel_name, design_name = \
            data.edge_index, data.kernel_name, data.design_name
        if not kernel_name in trace:
            continue
        nodes = [i for i in range(data.x.shape[0])]
        edges = edges.t().tolist()
        labels_map_ = {int(k): v for k, v in labels_map[kernel_name].items()}
        
        if i == batch_size-1:
            attention_weights_P_ = attention_weights_P[start_idx[i]:]
            attention_weights_pseudo_ = attention_weights_pseudo[start_idx[i]:]
        else:
            attention_weights_P_ = attention_weights_P[start_idx[i]:start_idx[i+1]]
            attention_weights_pseudo_ = attention_weights_pseudo[start_idx[i]:start_idx[i+1]]

        single_data_list.append(SingleData(nodes, edges, labels_map_, kernel_name, design_name, attention_weights_P_, False))
        single_data_list.append(SingleData(nodes, edges, labels_map_, kernel_name, design_name, attention_weights_pseudo_, True))
        
    return single_data_list

def draw_graph(data:SingleData, root_path, epoch):
    G = nx.DiGraph()
    G.add_nodes_from(data.nodes)
    G.add_edges_from(data.edges)
    node_sizes = [weight * 10000 for weight in data.attention_weights]
    # assert len(node_sizes) == len(data.nodes)
    # assert len(data.nodes) == len(data.labels)
    pos = nx.spring_layout(G, k=1, iterations=100)
    nx.draw(G, pos, node_size=node_sizes, with_labels=True, labels=data.labels, node_color='lightblue', \
        width=0.2, font_size=3, arrows=True, arrowsize=4)
    # print("\n================================")
    # print(f"kernel name:{data.kernel_name}")
    # print('edge: ')
    # print(data.edges)
    pseudo_or_P = 'pseudo' if data.is_pseudo else 'P' 
    tittle = f'{data.design_name}_epoch-{epoch}'
    path = join(root_path, f'{data.kernel_name}', f'{data.design_name}')
    if len(path) > 200:
        path = path[:200]
    makedirs(path, exist_ok=True)
    path = join(path, f'{pseudo_or_P}_epoch-{epoch}.png')
    plt.title(tittle)
    plt.savefig(path, dpi=500)
    plt.close() 


def draw_attention_graph(data, attention_weights, epoch, path):
    edges, batch, kernel_name, design_name = \
        data.edge_index, data.batch, data.kernel_name, data.design_name
    # attention_weights: {attention_weights['P'], attention_weights['pseudo']}
    # attention_weights['P'].shape = (len(nodes), 1)
    nodes = [i for i in range(data.x.shape[0])]
    single_data_list = batch_division2(data, attention_weights)
    makedirs(path, exist_ok=True)
    for i, single_data in enumerate(single_data_list):
        draw_graph(single_data, path, epoch)
        
        
# ======================= print test results =================
test_results_dict = dict()
# 收集预测结果、真实值
def collect_test_results(pred_dict, batched_data, task):
    if task == 'class':
        targets = ['perf']
    else:
        targets = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']
    global test_results_dict
    data_list = batched_data.to_data_list()
    for i, data in enumerate(data_list):
        if not data.kernel_name in test_results_dict.keys():
            test_results_dict[data.kernel_name] = dict()
        test_results_dict[data.kernel_name][data.design_name] = dict()
        if task == 'class':
            target = _get_y_with_target(data, 'perf')
            out = pred_dict['perf'][i].tolist()
            pred_valid = True if out[0] < out[1] else False
            valid = True if target else False
            correct = pred_valid == valid
            if pred_valid:
                type_ = 'tp' if valid else 'fp'
            else:
                type_ = 'fn' if valid else 'tn'
            res_str = {
                'correct': correct,
                'pred': pred_valid,
                'label': valid,
                'ratio': out,
                'type': type_
            }
            test_results_dict[data.kernel_name][data.design_name]['perf'] = res_str
        else:
            for target_name in targets:
                target = _get_y_with_target(data, target_name).tolist()
                out = pred_dict[target_name][i].tolist()
                if len(target) == 1:
                    target = target[0]
                    out = out[0]
                res_str = f'{out} (true: {target})'
                test_results_dict[data.kernel_name][data.design_name][target_name] = res_str

def print_test_results(output_path):
    with open(output_path, 'w') as f:
        json.dump(test_results_dict, f, indent=4)
        
def clear_test_results_dict():
    global test_results_dict
    test_results_dict = dict()
    