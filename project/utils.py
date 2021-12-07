import numpy as np
import os
import os.path as osp
import argparse

Config ={}
Config['root_path'] = '/home/ubuntu/mycode/project'
Config['meta_file'] = 'train.json'
Config['checkpoint_path'] = ''


Config['use_cuda'] = True
Config['debug'] = False
# Config['debug'] = True
Config['num_epochs'] = 30
Config['batch_size'] = 64

Config['learning_rate'] = 0.001

Config['num_workers'] = 4
