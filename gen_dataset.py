from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os
import time
import math
import cv2
import numpy as np
import pickle
from scoop import futures

from image_utils import rotateImage
import gen_dataset_flags as FLAGS

if FLAGS.FILLED_SHAPE:
    SRC_DIR = './filled_max_areas'
    IMG_DIR = './filled_primitives'
    DST_DIR = './filled_dataset_' + str(FLAGS.PAIR_NUM)
else:
    SRC_DIR = './hollow_max_areas'
    IMG_DIR = './hollow_primitives'
    DST_DIR = './hollow_dataset_' + str(FLAGS.PAIR_NUM)


IMAGE_SIZE = FLAGS.IMAGE_SIZE
PAIR_NUM = FLAGS.PAIR_NUM


def cropImage(input_img):
    lt, rb = (input_img.shape[1], input_img.shape[0]), (0, 0)
    for r in range(input_img.shape[0]):
        for c in range(input_img.shape[1]):
            if input_img[r][c] != 0:
                lt = (min(lt[0], c), min(lt[1], r))
                rb = (max(rb[0], c), max(rb[1], r))
    return input_img[lt[1]:rb[1] + 1, lt[0]:rb[0] + 1]


def generateShapePair(params):
    i, mat, images = params
    primitive_num = mat.shape[0]
    L_idx = random.randint(0, primitive_num - 1)
    K_idx = random.randint(0, primitive_num - 1)
    L_origin = images[L_idx]
    K_origin = images[K_idx]
    L = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    K = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    L_rotate = cropImage(rotateImage(L_origin, 360 * random.random()))
    K_rotate = cropImage(rotateImage(K_origin, 360 * random.random()))
    L_translate_range = (IMAGE_SIZE - L_rotate.shape[1], IMAGE_SIZE - L_rotate.shape[0])  # (x, y)
    K_translate_range = (IMAGE_SIZE - K_rotate.shape[1], IMAGE_SIZE - K_rotate.shape[0])
    L_translate = (random.randint(0, L_translate_range[0] - 1), random.randint(0, L_translate_range[1] - 1))  # (x, y)
    K_translate = (random.randint(0, K_translate_range[0] - 1), random.randint(0, K_translate_range[1] - 1))
    L[L_translate[1]:L_translate[1] + L_rotate.shape[0], L_translate[0]:L_translate[0] + L_rotate.shape[1]] = L_rotate
    K[K_translate[1]:K_translate[1] + K_rotate.shape[0], K_translate[0]:K_translate[0] + K_rotate.shape[1]] = K_rotate
    cv2.imwrite(os.path.join(DST_DIR, str(i) + '_L.png'), L)
    cv2.imwrite(os.path.join(DST_DIR, str(i) + '_K.png'), K)
    return mat[L_idx][K_idx]


if __name__ == '__main__':
    try:
        with open(SRC_DIR) as fp:
            mat = np.array(pickle.load(fp))
    except IOError as err:
        print('File Error', err)

    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)

    if len(mat.shape) != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] <= 1:
        print('Please provide valid *_max_area file')
        exit()

    start_time = time.time()
    primitive_num = mat.shape[0]
    images = [cv2.imread(os.path.join(IMG_DIR, str(i)+'.png'), cv2.IMREAD_GRAYSCALE) for i in xrange(primitive_num)]
    params_list = [(i, mat, images) for i in xrange(FLAGS.PAIR_NUM)]

    batch_size = FLAGS.BATCH_SIZE
    batch_num = int(math.ceil(len(params_list) / float(batch_size)))

    beg_time = time.time()
    overlap_areas = []
    cnt = 0
    for i in xrange(batch_num):
        overlap_areas += list(futures.map(generateShapePair, params_list[i*batch_size:min((i+1)*batch_size, len(params_list))]))
        cnt += batch_size
        print(cnt, '- duration =', time.time() - beg_time)
        beg_time = time.time()
        with open(os.path.join(DST_DIR, 'OVERLAP_AREAS'), 'wb') as fp:
            pickle.dump(overlap_areas, fp)

    print('duration =', time.time() - start_time)

        


