'''
For gen /home/wangxinglu/prj/few-shot/data/imglst/img1k.test.txt
'''

import lz as utils
from metadata import *
from lz import *
import tensorflow as tf

# HOME = os.environ['HOME'] or '/home/wangxinglu'
HOME = '/home/wangxinglu'
# cache_path= '/home/wangxinglu/prj/few-shot/src/nimgs.pkl'
cache_path= '/mnt/nfs1703/test/prj/cls-sample/nimgs.dbg.pkl'
num = 10000
np.random.seed(64)


def cls_sample(num, prob=None):
    if prob is None:
        prob = [1.35, 0.7, 1.35]

    # leaves = {}
    # for node in tf.gfile.ListDirectory(prefix):
    #     leaves[node] = len(tf.gfile.ListDirectory(prefix + '/' + node))
    # mypickle(leaves,cache_path)

    leaves = unpickle(cache_path)

    names, nimgs = leaves.keys(), leaves.values()
    names, nimgs = cosort(names, nimgs, True)
    names = names[nimgs >= 10]
    nimgs = nimgs[nimgs >= 10]
    comb = np.array([names, nimgs]).T
    base_p = 1. / comb.shape[0]

    p = np.concatenate(
        (np.ones(11803) * prob[0] * base_p,
         np.ones(1835) * prob[1] * base_p,
         np.ones(717) * prob[2] * base_p))
    p = p / p.sum()
    print(p.shape, comb.shape)
    # res = np.random.choice(np.arange(comb.shape[0]), size=num, replace=False, p=p)
    res = np.random.choice(np.arange(comb.shape[0]), size=num, replace=False)

    res = np.sort(res)
    res_nimgs = comb[res, :][:, 1]
    res = comb[res, :][:, 0]
    return res, res_nimgs


@chdir_to_root
def gen_imglst(names, prefix, train_file, test_file):
    os.chdir(prefix)
    imgs_train_l, imgs_test_l = [], []
    for ind, cls in enumerate(names):
        if not osp.exists(cls): continue
        imgs = tf.gfile.Glob(cls + '/*.JPEG')
        if len(imgs) == 0:
            utils.rm(cls, True)
            continue
        assert len(imgs) >= 10
        imgs = np.array(imgs)
        imgs_test = np.random.choice(imgs, max(3, imgs.shape[0] * 1 // 10), replace=False)
        imgs_train = np.setdiff1d(imgs, imgs_test)
        # imgs_train = imgs[:imgs.shape[0] * 9 // 10]
        # imgs_test = imgs[imgs.shape[0] * 9 // 10:]
        # imgs_test.shape, imgs_train.shape, imgs.shape
        imgs_train_l.append(
            np.stack((imgs_train, np.ones_like(imgs_train, dtype=int) * ind), axis=-1)
        )
        imgs_test_l.append(
            np.stack((imgs_test, np.ones_like(imgs_test, dtype=int) * ind), axis=-1)
        )

    imgs_train = np.concatenate(imgs_train_l, axis=0)
    np.random.shuffle(imgs_train)

    np.savetxt(train_file, imgs_train, delimiter=' ', fmt='%s')
    np.savetxt(test_file, np.concatenate(imgs_test_l, axis=0), delimiter=' ', fmt='%s')


if __name__ == '__main__':
    # prob = [1.5, 0., 1.5]
    # train_file = HOME + '/prj/few-shot/data/imglst/img10k.train.no1k.txt'
    # test_file = HOME + '/prj/few-shot/data/imglst/img10k.test.no1k.txt'
    # prefix = HOME + '/prj/few-shot/data/imagenet-raw'
    # utils.write_list('/home/wangxinglu/prj/few-shot/data/imagenet10k.txt.chk', names, delimiter=' ', fmt='%s')
    # gen_imglst(names, prefix, train_file, test_file)

    prob = [1.35, 0.7, 1.35]
    train_file = HOME + '/prj/few-shot/data/imglst/img10k.train.dbg'
    test_file = HOME + '/prj/few-shot/data/imglst/img10k.test.dbg'
    prefix = HOME + '/prj/few-shot/data/imagenet-raw'

    names, nimgs = cls_sample(num, prob)
    utils.write_list('/home/wangxinglu/prj/few-shot/data/imagenet10k.no1k.chk', names, delimiter=' ', fmt='%s')
    gen_imglst(names, prefix, train_file, test_file)
