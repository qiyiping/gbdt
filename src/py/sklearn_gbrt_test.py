import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import argparse

def load_data(filename, n):
    f = open(filename)
    x = []
    y = []
    for l in f:
        tokens = l.split(' ')
        y.append(float(tokens[0]))
        v = np.zeros(n)
        for i in range(2, len(tokens)):
            ts = tokens[i].split(':')
            v[int(ts[0])] = float(ts[1])
        x.append(v)
    return (x, y)

import sys
import math
import time
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="gbm test")
    parser.add_argument("--feature_size", type=int)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--max_depth", type=int)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--shrinkage", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--loss", type=str)
    args = parser.parse_args()
    x, y = load_data(args.train_file, args.feature_size)
    x1, y1 = load_data(args.test_file, args.feature_size)

    start_ts = time.time()
    est = GradientBoostingRegressor(n_estimators=args.iterations,
                                    learning_rate=args.shrinkage,
                                    max_depth=args.max_depth,
                                    random_state=0,
                                    loss=args.loss).fit(x, y)
    end_ts = time.time()
    print "time to fit: ", (end_ts - start_ts)

    y1est = est.predict(x1)
    print math.sqrt(mean_squared_error(y1, y1est))
