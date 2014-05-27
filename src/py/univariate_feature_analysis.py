#!/usr/bin/python

from sklearn.feature_selection import f_classif

import numpy

import sys

def load_feature_mapping(filename):
    z = []
    f = open(filename)
    for l in f:
        z.append(l.strip())
    return z

def load_data(filename, n_samples, n_features):
    x = numpy.zeros((n_samples, n_features, ), dtype=numpy.float64)
    y = numpy.zeros(n_samples)

    try:
        f = open(filename)
        sample_idx = 0

        max_index = 0
        for l in f:
            tokens = l.split(' ')
            y[sample_idx] = int(tokens[0])
            for i in range(2, len(tokens)):
                t = tokens[i].split(':')
                feature_idx = int(t[0])
                feature_val = float(t[1])
                x[sample_idx, feature_idx] = feature_val

            sample_idx += 1
            if sample_idx >= n_samples:
                break
    except Exception as e:
        print e

    return (x[0:sample_idx, :], y[0:sample_idx])


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'USAGE: cmd data samples mapping'
    n_samples = int(sys.argv[2])
    z = load_feature_mapping(sys.argv[3])
    n_features = len(z)

    print n_features

    x, y = load_data(sys.argv[1], n_samples, n_features)

    f, p = f_classif(x, y)

    t = { z[i]:f[i] for i in range(0, n_features) }

    st = sorted(t.items(), key=lambda x:x[1], reverse = True)

    for i in st:
        print '{0}:{1}'.format(i[0], i[1])
