#!/usr/bin/python

from sklearn.feature_selection import f_classif

import numpy
import pylab

import sys

def load_data(filename, n_samples, n_features):
    x = numpy.zeros((n_samples, n_features, ))
    y = numpy.zeros((n_samples, ))

    try:
        f = open(filename)
        sample_idx = 0
        for l in f:
            tokens = l.split(' ')
            y[sample_idx] = int(tokens[0])
            for i in range(2, len(tokens)):
                t = tokens[i].split(':')
                feature_idx = int(t[0])
                feature_val = float(t[1])
                x[sample_idx, feature_idx] = feature_val

            sample_idx += 1
    except Exception as e:
        print e

    return (x[0:sample_idx, :], y[0:sample_idx])


if __name__ == '__main__':
    n_samples = 10000
    n_features = 3

    x, y = load_data(sys.argv[1], n_samples, n_features)

    c, p = f_classif(x, y)
    print c
    print p
