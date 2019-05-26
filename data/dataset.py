import torch.utils.data as data

from torchvision.datasets import ImageFolder

import torch.nn as nn
import os.path
import re
import sys


def strip_newsgroup_header(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.

    Parameters
    ----------
    text : string
        The text from which to remove the signature block.
    """
    _before, _blankline, after = text.partition(r'\n\n')
    return after


_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')


def strip_newsgroup_quoting(text):
    """
    Given text in "news" format, strip lines beginning with the quote
    characters > or |, plus lines that often introduce a quoted section
    (for example, because they contain the string 'writes:'.)

    Parameters
    ----------
    text : string
        The text from which to remove the signature block.
    """
    good_lines = [line for line in text.split(r'\n')
                  if not _QUOTE_RE.search(line)]
    return r'\n'.join(good_lines)


def strip_newsgroup_footer(text):
    """
    Given text in "news" format, attempt to remove a signature block.

    As a rough heuristic, we assume that signatures are set apart by either
    a blank line or a line made of hyphens, and that it is the last such line
    in the file (disregarding blank lines at the end).

    Parameters
    ----------
    text : string
        The text from which to remove the signature block.
    """
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break

    if line_num > 0:
        return r'\n'.join(lines[:line_num])
    else:
        return text


def load_text(text_dir, class_to_idx, remove=()):
    texts = []
    labels = []
    text_dir = os.path.expanduser(text_dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(text_dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if fname.isdigit():
                    fpath = os.path.join(d, fname)
                    #遇到了encode问题
                    with open(fpath,'rb') as f:
                        text = str(f.read())
                        #过滤
                        if 'headers' in remove:
                            text = strip_newsgroup_header(text)
                        if 'footers' in remove:
                            text = strip_newsgroup_footer(text)
                        if 'quotes' in remove:
                            text = strip_newsgroup_quoting(text)
                        texts.append(text)
                        labels.append(class_to_idx[target])

    return texts, labels


class Newsgroup(data.Dataset):

    def __init__(self,text_dir,remove=()):
        classes, class_to_idx = self._find_classes(text_dir)
        self.text_dir = text_dir
        
        #读文件，把文本保存起来，根据remove去掉文件内的无关信息
        texts, labels = load_text(text_dir, class_to_idx, remove)


        self.remove = remove
        self.texts = texts
        self.labels = labels
        #英文单词转位 数字向量表现形式, pad变为一样的长度，如果要变为数字表示形式，需要有一个单词到数字的hash，word2vec，glove



    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


