import unittest
from dataset import Newsgroup
from torch.utils.data import DataLoader

class NewsgroupTestCase(unittest.TestCase):
    # def test_load(self):
        
    #     TEXT_DATA_DIR='20_newsgroups'
    #     news_ds = Newsgroup(TEXT_DATA_DIR,remove=('headers','footers','quotes'))
    #     print(news_ds[0])

    def test_dataloader(self):
        TEXT_DATA_DIR='20_newsgroups'
        news_ds = Newsgroup(TEXT_DATA_DIR,remove=('headers','footers','quotes'))
        train_dl = DataLoader(news_ds,batch_size=64,shuffle=True,drop_last=True)

        for idx, data in enumerate(train_dl):
            inputs, labels = data
            print(inputs.size())
            break

if __name__ == '__main__':
    unittest.main()