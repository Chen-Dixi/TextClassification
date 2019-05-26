import unittest
from dataset import Newsgroup

class NewsgroupTestCase(unittest.TestCase):
    def test_load(self):
        TEXT_DATA_DIR='20_newsgroups'
        news_ds = Newsgroup(TEXT_DATA_DIR,remove=('headers','footers','quotes'))
        print(news_ds.texts[0],news_ds.labels[0])



if __name__ == '__main__':
    unittest.main()