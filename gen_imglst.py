'''
For gen /home/wangxinglu/prj/few-shot/data/imglst/img1k.test.txt
'''

import lz as utils
from lz import *
from chk_imglst_pool import check_individual

# HOME = os.environ['HOME'] or '/home/wangxinglu'
HOME = '/home/wangxinglu'
cache_path = '/home/wangxinglu/prj/few-shot/src/nimgs.pkl'
num = 10000
np.random.seed(64)


def cls_sample(num, prob=None):
    if prob is None:
        prob = [1.35, 0.7, 1.35]

    # leaves = {}
    # for node in tf.gfile.ListDirectory(prefix):
    #     leaves[node] = (os.listdir(prefix + '/' + node))
    # mypickle(leaves,cache_path)

    name2nimg = unpickle('nimgs.pkl')
    name2nimg = {name: len(fnames) for name, fnames in name2nimg.items()}
    name2nimg = {name: nimg for name, nimg in name2nimg.items() if nimg >= 10}

    nimgs = np.asarray(list(name2nimg.values()))
    names = np.asarray(list(name2nimg.keys()))
    names, nimgs = cosort(names, nimgs, True)

    comb = np.array([names, nimgs]).T
    base_p = 1. / comb.shape[0]
    ind1200 = bsearch(nimgs, 1200)
    ind1300 = bsearch(nimgs, 1300)
    indt = nimgs.shape[0]
    interval = np.diff([0, ind1200, ind1300, indt])
    p = np.concatenate(
        (np.ones(interval[0]) * prob[0] * base_p,
         np.ones(interval[1]) * prob[1] * base_p,
         np.ones(interval[2]) * prob[2] * base_p))
    p = p / p.sum()
    print(p.shape, comb.shape)
    res = np.random.choice(np.arange(comb.shape[0]), size=num, replace=False, p=p)
    # res = np.random.choice(np.arange(comb.shape[0]), size=num, replace=False)

    res = np.sort(res)
    res_nimgs = comb[res, :][:, 1]
    res = comb[res, :][:, 0]
    return res, res_nimgs


def bsearch(nums, query):
    nums = np.asarray(nums)
    small = nums < query
    res = np.arange(nums.shape[0])[small].max()
    return res


@chdir_to_root
def gen_imglst(names, prefix, train_file, test_file):
    os.chdir(prefix)
    imgs_train_l, imgs_test_l = [], []
    name2imgs = unpickle('nimgs.pkl')
    for ind, cls in enumerate(names):
        if not osp.exists(cls): continue
        imgs = name2imgs[cls]
        imgs = [cls + '/' + img_ for img_ in imgs]
        if len(imgs) == 0:
            utils.rm(cls)
            continue
        assert len(imgs) >= 10
        imgs = np.array(imgs)
        checks = map(check_individual, imgs)

        for check, img in zip(checks, imgs):
            if not check:
                imgs = np.setdiff1d(imgs, img)
                # utils.rm(img)
                print('corrupt ', img)

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

    prob = [1.35, 0.7, 1.35]
    train_file = HOME + '/prj/few-shot/data/imglst/img10k.train.txt'
    test_file = HOME + '/prj/few-shot/data/imglst/img10k.test.txt'
    prefix = HOME + '/prj/few-shot/data/imagenet-raw'

    names, nimgs = cls_sample(num, prob)
    utils.write_list('/home/wangxinglu/prj/few-shot/data/imagenet10k.no1k.chk', names, delimiter=' ', fmt='%s')
    gen_imglst(names, prefix, train_file, test_file)
