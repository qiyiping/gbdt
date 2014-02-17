import random
def sample_data(input_file, output_file, line_number, negative_ratio=1, positive_weight=1, negative_weight=1):
    f = open(input_file)
    fo = open(output_file, 'w')
    cnt = 0
    for l in f:
        if cnt >= line_number:
            break
        idx = l.find(' ')
        label = float(l[:idx])
        idx2 = l.find(' ', idx+1)
        weight = float(l[idx+1:idx2])
        if negative_ratio >= 1 or label > 0 or random.random() < negative_ratio:
            s = [label]
            if label > 0:
                s.append(weight * positive_weight)
            else:
                s.append(weight * negative_weight)
            s.append(l[idx2+1:])
            fo.write(' '.join(map(str, s)))
            cnt += 1
    f.close()
    fo.close()

import sys
if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    line_number = int(sys.argv[3])
    negative_ratio = float(sys.argv[4])
    positive_weight = float(sys.argv[5])
    sample_data(input_file, output_file, line_number, negative_ratio, positive_weight)
