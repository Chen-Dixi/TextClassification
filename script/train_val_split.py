import os
from os.path import join
import random
import shutil
random.seed(1995)

OLD_DIR = 'data/20_newsgroups'
TEST_DIR = 'data/20_newsgroups_test'
os.mkdir(TEST_DIR)
TRAIN_VAL_PROPORTION = 5

classes = [d.name for d in os.scandir(OLD_DIR) if d.is_dir()]

for target in sorted(classes):
    d = os.path.join(OLD_DIR, target)
    if not os.path.isdir(d):
        continue
    #创建新文件夹
    new_d = os.path.join(TEST_DIR,target)
    if not os.path.exists(new_d):
        os.mkdir(new_d)

    fnames = [d.name for d in os.scandir(d) if d.name.isdigit]
    tests = random.sample(fnames, len(fnames)//TRAIN_VAL_PROPORTION)

    for fname in tests:
        fpath = os.path.join(d,fname)
        new_path = os.path.join(new_d, fname)
        shutil.move(fpath,new_path)


