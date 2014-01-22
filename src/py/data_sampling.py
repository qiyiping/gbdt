import random
def sample_data(input_file, output_file, line_number, negative_ratio):
    f = open(input_file)
    fo = open(output_file, 'w')
    cnt = 0
    for l in f:
        if cnt >= line_number:
            break
        idx = l.find(' ')
        label = float(l[:idx])
        if negative_ratio >= 1 or label > 0 or random.random() < negative_ratio:
            fo.write(l)
            cnt += 1
    f.close()
    fo.close()

import sys
if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    line_number = int(sys.argv[3])
    negative_ratio = float(sys.argv[4])
    sample_data(input_file, output_file, line_number, negative_ratio)
