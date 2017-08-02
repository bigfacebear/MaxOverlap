from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import math
import random

import cv2
from deap import creator, base, tools, algorithms
# from multiprocessing import Pool
from scoop import futures
import numpy as np
import pickle

from image_utils import overlapArea, getArea
import gen_max_areas_flags as FLAGS


FILL_SHAPE = FLAGS.FILL_SHAPE
DATA_NUM = FLAGS.DATA_NUM
total_num = int(DATA_NUM * (DATA_NUM - 1) / 2)

if FILL_SHAPE:
    DATA_DIR = './filled_primitives'
    DST_DIR = './filled_dataset100'
else:
    DATA_DIR = './hollow_primitives'
    DST_DIR = './hollow_dataset'


POP_SIZE = FLAGS.POP_SIZE
NGEN = FLAGS.NGEN
CXPB = FLAGS.CXPB
MUTPB = FLAGS.MUTPB

toolbox = base.Toolbox()

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

def newTransformation(img_size):
    x = random.randint(0, img_size[1])
    y = random.randint(0, img_size[0])
    angle = 360 * random.random()
    return creator.Individual([x, y, angle])


def checkBounds(img_size):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                child[0] = min(max(0, int(child[0])), img_size[1])
                child[1] = min(max(0, int(child[1])), img_size[0])
                child[2] %= 360.0
            return offspring

        return wrapper

    return decorator


def evaluate(individual, L, K):
    return overlapArea(L, K, individual[0:2], individual[2])


def processPair(params):
    idxPair, images = params
    beg_time = time.time()
    L = images[idxPair[0]]
    K = images[idxPair[1]]
    if L is None or K is None:
        return 0

    toolbox.register('individual', newTransformation, L.shape)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('mate', tools.cxUniform, indpb=0.5)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=5, indpb=0.5)
    toolbox.register('select', tools.selTournament, tournsize=5)
    toolbox.register('evaluate', evaluate)

    toolbox.decorate('mate', checkBounds(L.shape))
    toolbox.decorate('mutate', checkBounds(L.shape))

    pop = toolbox.population(n=POP_SIZE)

    for g in range(NGEN):
        # Select and clone the next generation individuals
        offspring = map(toolbox.clone, toolbox.select(pop, len(pop)))

        # Apply crossover and mutation on the offspring
        offspring = algorithms.varAnd(offspring, toolbox, CXPB * (1 - float(g) / NGEN), MUTPB * (1 - float(g) / NGEN))

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = []
        for ind in invalid_ind:
            fitnesses.append(evaluate(ind, L, K))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = [fit]

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        # print(str(int(float(g) / NGEN * 100)) + '%', 'max_area =', evaluate(tools.selBest(pop, k=1)[0], L, K))


    best = tools.selBest(pop, k=1)[0]
    max_area = best.fitness.values[0]

    end_time = time.time()
    # print(' - lock =', idxPair[0], ', key =', idxPair[1])
    # print(' - transformation =', best)
    # print(' - max_area =', max_area)
    print(idxPair, ' - duration =', end_time - beg_time)
    return max_area




def saveMat(mat):
    if FILL_SHAPE:
        file_name = 'filled_max_areas'
    else:
        file_name = 'hollow_max_areas'
    with open(file_name, 'wb') as fp:
        pickle.dump(mat.tolist(), fp)


if __name__ == '__main__':
    print('begin')

    if FILL_SHAPE:
        file_name = 'filled_max_areas'
    else:
        file_name = 'hollow_max_areas'
    if os.path.exists(file_name):
        with open(file_name) as fp:
            mat = np.array(pickle.load(fp))
            print('open', file_name, 'from file')
    else:
        mat = np.zeros((DATA_NUM, DATA_NUM), dtype=np.int)

    batch_size = FLAGS.BATCH_SIZE

    idxPairs = [(i, j) for i in xrange(FLAGS.DATA_NUM) for j in xrange(i) if mat[i][j] == 0]
    images = [cv2.imread(os.path.join(DATA_DIR, str(i)+'.png'), cv2.IMREAD_GRAYSCALE) for i in range(DATA_NUM)]

    params_batches = []
    for i in xrange(int(math.ceil(len(idxPairs) / batch_size))):
        idx_batch = idxPairs[i*batch_size:min((i+1)*batch_size, len(idxPairs))]
        params_batch = [(idx_batch[i], images) for i in range(len(idx_batch))]
        params_batches.append(params_batch)

    cnt = 0
    # pool = Pool()
    beg_time = time.time()
    for params_batch in params_batches:
        results = futures.map(processPair, params_batch)
        for params, res in zip(params_batch, results):
            pair = params[0]
            mat[pair[0]][pair[1]] = res
            mat[pair[1]][pair[0]] = res
        saveMat(mat)
        cnt += batch_size
        print('saved', cnt, 'batches, duration =', time.time() - beg_time)
        beg_time = time.time()

    for i in xrange(DATA_NUM):
        mat[i][i] = getArea(images[i])

    print(mat)

    saveMat(mat)
