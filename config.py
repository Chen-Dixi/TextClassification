import yaml
import easydict
from os.path import join

import argparse
parser = argparse.ArgumentParser(description='Code for *Universal Domain Adaptation*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')

args = parser.parse_args()

config_file = args.config

args = yaml.load(open(config_file))

args = easydict.EasyDict(args)


