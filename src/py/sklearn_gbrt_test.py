import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

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
if __name__ == '__main__':
    x, y = load_data(sys.argv[1], int(sys.argv[3]))
    x1, y1 = load_data(sys.argv[2], int(sys.argv[3]))

    est = GradientBoostingRegressor(n_estimators=100,
                                    learning_rate=0.1,
                                    max_depth=int(sys.argv[4]),
                                    random_state=0,
                                    loss='ls').fit(x, y)

    y1est = est.predict(x1)
    print math.sqrt(mean_squared_error(y1, y1est))

    print y1est
