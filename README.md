# TextClassification
本文是春季课程“自然语言处理与应用”文本分类作业的简单记录，关于文本分类方面上，知乎上很多很好的文章：[知乎“看山杯”夺冠记](https://zhuanlan.zhihu.com/p/28923961)，[用深度学习（CNN RNN Attention）解决大规模文本分类问题](https://zhuanlan.zhihu.com/p/25928551)，对写这个作业帮助也很大。
<!-- more -->

>作业源码 github地址:[<u>TextClassification</u>](https://github.com/Chen-Dixi/TextClassification)

## 一. 数据集
作业使用20_newsgroups数据集，可以在[这个网址](http://www.cs.cmu.edu/afs/cs/project/theo-11/www/naive-bayes.html)中下载，解压后全部是英文文本，这就需要自己先对文本进行预处理。

## 二. 数据预处理
- #### 1.训练集和测试集划分
按照PPT中的“一般实验设置”，我简单的把数据集按照4:1的比例划分为训练集和测试集
 
- #### 2.文本内容“过滤”
20_newsgroups的内容是新闻文本，包含很多`header`、`footer`和`quotes`内容

自己写字符串过滤比较麻烦，幸运的是我在`sciki-learn`的python包`sklearn.datasets.fetch_20newsgroups`里面找到对`20newsgroups`进行文本过滤的代码，拷贝过来直接使用。

- #### 3.每篇文本转换为长度相等一维整数向量
我的文本分类器用的是卷积神经网络CNN，所以除了把文本中的每个英文单词转换为整数数字外，还需要将每篇文章通过`pad_sequence`转换为一样的长度。上述两个繁琐的操作也可以方便地通过`keras`提供的`keras.preprocessing.text.Tokenizer`和`keras.preprocessing.sequence.pad_sequences`直接完成。
```python
def word2index(texts,vocab_size=299567,pad=True):
    #vocab_size is embedding_dim
    
    # word转为index
    tokenizer = Tokenizer(num_words=vocab_size,oov_token='something')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    # pad
    data = pad_sequences(sequences ,maxlen=5000,padding='post', truncating='post')
    #return
    return data
```

通过上面3个步骤，将得到的文本和标签数据转为`numpy`数据，保存在`.npz`文件中方便训练时加载数据集。同时获取文本数据和监督信号(标签label)的代码借鉴了pytorch中的`DatasetFolder`
```python
np.savez_compressed('data/20_newsgroups_npz/train.npz',texts=train_inputs,labels=train_labels)
np.savez_compressed('data/20_newsgroups_npz/test.npz',texts=test_inputs,labels=test_labels)
```
## 三. 训练网络
- #### 1.词嵌入Embedding
`PyTorch`中，词嵌入的结构用`nn.Embedding`，里面的权重是一个二维矩阵，每一行代表一个单词的词向量。输入必须是`LongTensor`类型。其他文章说，在数据量充足的情况下，其实不一定需要预训练的词向量，于是我就开始尝试'train from scratch'。问题是训练开始的时候非常令人揪心，仿佛要训练到时间的尽头😅，所以就先停了。

我在github中找到一个[缩小版的word2vec GoogleNews词向量](https://github.com/eyaler/word2vec-slim)，里面有299567个英文单词的300维词向量。使用预训练词向量意味着我们需要重新预处理数据集。词向量中每一行都代表不同单词，`word2vec`中除了包含\[299567x300\]的权重外，还有一个英文单词到数字的`vocabulary哈希`。因此数据预处理的第3.步骤也应该按照这个`vocabulary哈希`将单词转为对应的数字，此时数据预处理代码有一些小更改，可以在我的程序中看到处理细节。

- #### 2.分类器
embedding后面的网络结构不是这次作业的重点，我使用了北邮学长在“知乎看山杯”取得第一名自创的结构`CNNText_inception`，这是关于那次比赛的纪录文章：[知乎“看山杯” 夺冠记](https://zhuanlan.zhihu.com/p/28923961)


## 四. 总结
没有调参，没有使用什么技巧，准确率很差，50个epoch在测试集上准确率最高只有71%，就这么把作业交了吧。

