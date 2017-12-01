try:
    import cPickle as pickle
except:
    import pickle
import six, os, sys, csv, time, \
    random, os.path as osp, \
    subprocess, json, \
    numpy as np, pandas as pd, \
    glob, re, networkx as nx, \
    h5py, yaml, copy, multiprocessing as mp, \
    pandas as pd, yaml, collections, \
    logging, colorlog, yaml, cvbase as cvb, shutil, \
    easydict
import subprocess

# tensorflow as tf, keras, torch , redis
# import torch
# from torch import nn
# from torch.autograd import Variable
# import torch.nn.functional as F

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from IPython import embed
from IPython.display import display, HTML, SVG

# root_path = osp.normpath(
#     osp.join(osp.abspath(osp.dirname(__file__)))
# )

root_path = '/home/wangxinglu/prj/few-shot/'


def set_stream_logger(log_level=logging.INFO):
    sh = colorlog.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(
        colorlog.ColoredFormatter(
            '%(asctime)s %(filename)s [line:%(lineno)d] %(log_color)s%(levelname)s%(reset)s %(message)s'))
    logging.root.addHandler(sh)


def set_file_logger(work_dir=None, log_level=logging.DEBUG):
    work_dir = work_dir or os.getcwd()
    fh = logging.FileHandler(os.path.join(work_dir, 'log.txt'))
    fh.setLevel(log_level)
    fh.setFormatter(
        logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s'))
    logging.root.addHandler(fh)


# def set_logger():
logging.root.setLevel(logging.DEBUG)
set_stream_logger()
set_file_logger()


def gc_collect():
    import gc
    gc.collect()


def sel_np(A):
    dtype = str(A.dtype)
    shape = A.shape
    A = A.ravel().tolist()
    sav = {'shape': shape, 'dtype': dtype,
           'A': A
           }
    return json.dumps(sav)


def desel_np(s):
    import json
    sav = json.loads(s)
    A = sav['A']
    A = np.array(A, dtype=sav['dtype']).reshape(sav['shape'])
    return A


def append_file(line, file=None):
    file = file or 'append.txt'
    with open(file, 'a') as  f:
        f.writelines(line + '\n')


def cpu_priority(level=19):
    import psutil
    p = psutil.Process(os.getpid())
    p.nice(level)


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def init_dev(n=(0,)):
    logging.info('use gpu {}'.format(n))
    from os.path import expanduser
    home = expanduser("~")
    if isinstance(n, int):
        n = (n,)
    devs = ''
    for n_ in n:
        devs += str(n_) + ','
    os.environ["CUDA_VISIBLE_DEVICES"] = devs
    os.environ['PATH'] = home + '/cuda-8.0/bin:' + os.environ['PATH']
    # os.environ['PATH'] = home + '/anaconda2/bin:' + os.environ['PATH']
    os.environ['PATH'] = home + '/usr/local/cuda-8.0/bin:' + os.environ['PATH']

    os.environ['LD_LIBRARY_PATH'] = home + '/cuda-8.0/lib64'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-8.0/lib64'
    # os.environ['PYTHONWARNINGS'] = "ignore"


def set_env(key, value):
    value = os.path.abspath(value)
    os.environ[key] = value + ':' + os.environ[key]


def allow_growth_tf():
    import tensorflow as tf
    _sess_config = tf.ConfigProto(allow_soft_placement=True)
    _sess_config.gpu_options.allow_growth = True
    return _sess_config


def allow_growth_keras():
    import tensorflow as tf
    tf_graph = tf.get_default_graph()
    _sess_config = tf.ConfigProto(allow_soft_placement=True)
    _sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=_sess_config, graph=tf_graph)
    import keras.backend as K
    K.set_session(sess)
    return sess


def get_dev(n=1, ok=(0, 1, 2, 3, 4, 5, 6, 7), mem=(0.5, 0.9), sleep=60):
    import GPUtil, time
    logging.info('Auto select gpu')
    GPUtil.showUtilization()

    def _limit(devs, ok):
        return [int(dev) for dev in devs if dev in ok]

    devs = GPUtil.getAvailable(order='memory', maxLoad=1, maxMemory=mem[0], limit=n)  #

    devs = _limit(devs, ok)
    if len(devs) >= 1:
        logging.info('available {}'.format(devs))
        # GPUtil.showUtilization()
        return int(devs[0]) if n == 1 else devs
    while len(devs) == 0:
        devs = GPUtil.getAvailable(order='random', maxLoad=1, maxMemory=mem[1], limit=n)
        devs = _limit(devs, ok)
        if len(devs) >= 1:
            logging.info('available {}'.format(devs))
            GPUtil.showUtilization()
            return devs[0] if n == 1 else devs
        logging.info('no device avelaible')
        GPUtil.showUtilization()
        time.sleep(sleep)


# def grid_iter(tmp):
#     res = cartesian(tmp.values())
#     np.random.shuffle(res)
#     for res_ in res:
#         yield dict(zip(tmp.keys(), res_))


def shuffle_iter(iter):
    iter = list(iter)
    np.random.shuffle(iter)
    for iter_ in iter:
        yield iter_


def optional_arg_decorator(fn):
    def wrapped_decorator(*args):
        if len(args) == 1 and callable(args[0]):
            return fn(args[0])
        else:
            def real_decorator(decoratee):
                return fn(decoratee, *args)

            return real_decorator

    return wrapped_decorator


def randomword(length):
    import random, string
    return ''.join(random.choice(string.lowercase) for i in range(length))


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def cosort(tensor, y, return_y=False):
    comb = zip(tensor, y)
    comb_sorted = sorted(comb, key=lambda x: x[1])
    if not return_y:
        return np.array([comb_[0] for comb_ in comb_sorted])
    else:
        return np.array([comb_[0] for comb_ in comb_sorted]), np.array([comb_[1] for comb_ in
                                                                        comb_sorted])


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.start_time = time.time()
        # logger.info('time pass {}'.format(self.diff))
        return self.diff


timer = Timer()


@optional_arg_decorator
def timeit(fn, info=''):
    def wrapped_fn(*arg, **kwargs):
        timer = Timer()
        timer.tic()
        res = fn(*arg, **kwargs)
        diff = timer.toc()
        logging.info((info + 'takes time {}').format(diff))
        return res

    return wrapped_fn


class Database(object):
    def __init__(self, *args, **kwargs):
        self.fid = h5py.File(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fid.close()

    def __getitem__(self, keys):
        if isinstance(keys, (tuple, list)):
            return [self._get_single_item(k) for k in keys]
        return self._get_single_item(keys)

    def _get_single_item(self, key):
        return np.asarray(self.fid[key])

    def __setitem__(self, key, value):
        if key in self.fid:
            if self.fid[key].shape == value.shape and \
                            self.fid[key].dtype == value.dtype:
                print('shape type same, old is updated')
                self.fid[key][...] = value
            else:
                del self.fid[key]
                print('old shape {} new shape {} updated'.format(self.fid[key].shape, value.shape))
                self.fid.create_dataset(key, data=value)
        else:
            self.fid.create_dataset(key, data=value)

    def __delitem__(self, key):
        del self.fid[key]

    def __len__(self):
        return len(self.fid)

    def __iter__(self):
        return iter(self.fid)

    def flush(self):
        self.fid.flush()

    def close(self):
        self.fid.close()

    def keys(self):
        return self.fid.keys()


def mypickle(data, file_path):
    mkdir_p(osp.dirname(file_path), delete=False)
    print('pickle into', file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_df(df, path):
    df.to_hdf(path, 'df', mode='w')


def read_df(path):
    return pd.read_hdf(path, 'df')


def mkdir_p(path, delete=False):
    if path == '': return

    if delete:
        rm(path)
    if not osp.exists(path):
        print('mkdir -p  ' + path)
        subprocess.call(('mkdir -p ' + path).split())


def shell(cmd, block=True):
    import os
    my_env = os.environ.copy()
    home = os.path.expanduser('~')
    # logging.info('cmd is ' + cmd)
    if block:
        # subprocess.call(cmd.split())
        task = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=my_env)
        msg = task.communicate()
        if msg[0] != b'' and msg[0] != '':
            logging.info('stdout {}'.format(msg[0]))
        if msg[1] != b'' and msg[1] != '':
            logging.error('stderr {}'.format(msg[1]))
        return msg
    else:
        print('Non-block!')
        task = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=my_env)
        return task


def check_path(path):
    path = osp.dirname(path)
    if not osp.exists(path):
        mkdir_p(path)


def ln(path, to_path):
    if osp.exists(to_path):
        print('error! exist ' + to_path)
    path = osp.abspath(path)
    cmd = "ln -s " + path + " " + to_path
    print(cmd)
    proc = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    return proc


def tar(path, to_path=None):
    if not osp.exists(path):
        return
    if not osp.exists(to_path):
        mkdir_p(to_path)
    if os.path.exists(to_path) and not len(os.listdir(to_path)) == 0:
        rm(path)
        return
    if to_path is not None:
        cmd = "tar xf " + path + " -C " + to_path
        print(cmd)
    else:
        cmd = "tar xf " + path
    shell(cmd, block=True)
    if os.path.exists(path):
        rm(path)


def rmdir(path):
    cmd = "rmdir " + path
    shell(cmd)


def rm(path, block=True, hard=True):
    path = osp.abspath(path)
    if not hard:
        dst = glob.glob('{}.bak*'.format(path))
        parsr = re.compile('{}.bak(\d+)'.format(path))
        used = [0, ]
        for d in dst:
            m = re.match(parsr, d)
            used.append(int(m.groups()[0]))
        dst_path = '{}.bak{}'.format(path, max(used) + 1)

        cmd = 'mv {} {} '.format(path, dst_path)
        print(cmd)
        shell(cmd, block=block)
    else:
        if osp.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    return


def show_img(path):
    from IPython.display import Image

    fig = Image(filename=(path))
    return fig


def show_pdf(path):
    from IPython.display import IFrame
    path = osp.relpath(path)
    return IFrame(path, width=600, height=300)


def print_graph_info():
    import tensorflow as tf
    graph = tf.get_default_graph()
    graph.get_tensor_by_name("Placeholder:0")
    layers = [op.name for op in graph.get_operations() if op.type == "Placeholder"]
    print([graph.get_tensor_by_name(layer + ":0") for layer in layers])
    print([op.type for op in graph.get_operations()])
    print([n.name for n in tf.get_default_graph().as_graph_def().node])
    print([v.name for v in tf.global_variables()])
    print(graph.get_operations()[20])


def chdir_to_root(fn):
    def wrapped_fn(*args, **kwargs):
        restore_path = os.getcwd()
        os.chdir(root_path)
        res = fn(*args, **kwargs)
        os.chdir(restore_path)
        return res

    return wrapped_fn


def scp(src, dest, dry_run=False):
    cmd = ('scp -r ' + src + ' ' + dest)
    print(cmd)
    if dry_run: return
    return shell(cmd, block=False)


def read_list(file, delimi=" "):
    if osp.exists(file):
        lines = np.genfromtxt(file, dtype='str', delimiter=delimi)
        return lines
    else:
        return []


def cp(from_path, to):
    subprocess.call(('cp -r ' + from_path + ' ' + to).split())


def mv(from_path, to):
    if not osp.exists(to):
        mkdir_p(to)
    if not isinstance(from_path, list):
        subprocess.call(('mv ' + from_path + ' ' + to).split())
    else:
        for from_ in from_path:
            subprocess.call(('mv ' + from_ + ' ' + to).split())


def dict_concat(d_l):
    d1 = d_l[0].copy()
    for d in d_l[1:]:
        d1.update(d)
    return d1


def clean_name(name):
    import re
    name = re.findall('([a-zA-Z0-9/-]+)(?::\d+)?', name)[0]
    name = re.findall('([a-zA-Z0-9/-]+)(?:_\d+)?', name)[0]
    return name


class Struct(object):
    def __init__(self, entries):
        self.__dict__.update(entries)

    def __getitem__(self, item):
        return self.__dict__[item]


def dict2obj(d):
    return Struct(d)


def dict2str(others):
    name = ''
    for key, val in others.iteritems():
        name += '_' + str(key)
        if isinstance(val, dict):
            name += '_' + dict2str(val)
        elif isinstance(val, list):
            for val_ in val:
                name += '-' + str(val_)
        else:
            name += '_' + str(val)
    return name


def list2str(li, delimier=''):
    name = ''
    for name_ in li:
        name += (str(name_) + delimier)

    return name


def write_list(file, l, sort=True, delimiter=' ', fmt='%.18e'):
    l = np.array(l)
    if sort:
        l = np.sort(l, axis=0)
    np.savetxt(file, l, delimiter=delimiter, fmt=fmt)


def rsync(from_, to):
    cmd = ('rsync -avzP ' + from_ + ' ' + to)
    print(cmd)
    return shell(cmd, block=False)


def i_vis_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    import tensorflow as tf
    from IPython.display import display, HTML, SVG
    import os
    def strip_consts(graph_def, max_const_size=32):
        """Strip large constant values from graph_def."""
        import tensorflow as tf

        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add()
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>" % size)
        return strip_def

    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


if __name__ == '__main__':
    pass
