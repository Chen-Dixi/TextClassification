import unittest
import yaml
import easydict
from os.path import join
import argparse

class ConfigTestCase(unittest.TestCase):
    def test_loadArgs(self):
        CONFIG_FILE = "newsgroup.yaml"
        args = yaml.load(open(CONFIG_FILE))
        args = easydict.EasyDict(args)
        print(args.data.vocab_size)

unittest.main()