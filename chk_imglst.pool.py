import lz as utils
from lz import *
import cv2


def check_individual(filename, filepath):
    filepath=osp.abspath(filepath)
    if not osp.exists(filepath):
        print('no exist', filepath)
        append_file(filepath.split('/')[-2], '/home/wangxinglu/fail.txt')
        # utils.rm(osp.dirname(filepath))
        return
    try:
        im = cv2.imread(filepath, cv2.IMREAD_COLOR)
        stdout, stderr = shell('convert {} /dev/null'.format(filepath))  # -regard-warnings
        if stderr != b'':
            raise ValueError('corrupt! {} '.format(stderr))
    except Exception as inst:
        print(filepath, ' error ', inst)
        append_file(filepath, '/home/wangxinglu/corrupt.txt')
        # utils.rm(filepath)
        return
    # cv2.imwrite(filepath, im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    if im.shape[0] <= 16 or im.shape[1] <= 16:
        # utils.rm(filepath)
        append_file(filepath, '/home/wangxinglu/small.txt')
        print('rm small ', filepath)
    assert im.shape[-1] == 3
    assert im.shape[0] > 13 and im.shape[1] > 13
    assert (len(im.shape) == 3 and im.shape[-1] == 3), 'img ' + filepath + str(im.shape)
    assert im.shape[1] != 0 and im.shape[0] != 0, 'width ' + filepath + str(im.shape)


# @chdir_to_root
# def check_img(prefix='/home/wangxinglu/prj/few-shot/data/imagenet-raw'):
#     import multiprocessing as mp
#     pool = mp.Pool(1024)
#     os.chdir(prefix)
#     for dirpath, dirnames, filenames in tf.gfile.Walk('.'):
#         for filename in filenames:
#             if filename.endswith('JPEG'):
#                 filepath = dirpath + '/' + filename
#                 pool.apply_async(check_individual, args=(filename, filepath))
#                 # check_individual(filename, filepath)


@chdir_to_root
def chk_img_lst(path):
    import multiprocessing as mp
    pool = mp.Pool(320)
    prefix = '/home/wangxinglu/prj/few-shot/data/imagenet-raw/'
    cache=np.array(read_list(path))
    # mypickle(cache,'cache.pkl')
    # cache = unpickle('cache.pkl')
    # print(cache, 'ok')

    for imgpath in cache[::-1, 0]:
        pool.apply_async(check_individual, args=(imgpath.split('/')[-1], prefix + imgpath))
    pool.close()
    pool.join()


if __name__ == '__main__':
    # chk_img_lst(path='/home/wangxinglu/prj/few-shot/data/imglst/img10k.test.txt')
    chk_img_lst(path='/home/wangxinglu/prj/few-shot/data/imglst/img10k.train.txt')
    # check_img()
    # check_img(prefix='/mnt/nfs1703/kchen/imagenet-raw-trans-to-redis')
