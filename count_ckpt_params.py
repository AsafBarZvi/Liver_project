import sys
import os
import re
import tensorflow as tf
import numpy as np
#import ipdb

if len(sys.argv) == 2:
    ckpt_fpath = sys.argv[1]
else:
    print('Usage: python count_ckpt_param.py path-to-ckpt')
    sys.exit(1)

# Open TensorFlow ckpt
reader = tf.train.NewCheckpointReader(ckpt_fpath + re.search('(\w+\.ckpt)\.meta ',' '.join(os.listdir(ckpt_fpath))).group(1))

print('\nCount the number of parameters in ckpt file(%s)' % ckpt_fpath)
#ipdb.set_trace()
param_map = reader.get_variable_to_shape_map()
total_count = 0
layers = param_map.items()
layers = sorted(layers)
for k, v in layers:
    #if 'Momentum' not in k and 'global_step' not in k:
    if 'train_step/' in k and '/Adam_1' in k:
        temp = np.prod(v)
        total_count += temp
        print('%s: %s => %d' % (k, str(v), temp))

        print('Total Param Count: %d' % total_count)

