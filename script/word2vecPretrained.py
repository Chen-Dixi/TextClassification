from gensim.models import word2vec
from gensim.models import KeyedVectors
import numpy as np
slim_filename = 'data/word2vec-slim/GoogleNews-vectors-negative300-SLIM.bin.gz'


model = KeyedVectors.load_word2vec_format(slim_filename, binary=True)

keys = [w for w in model.vocab.keys()]
wordkeys=set(keys)
s='asdagdsf'
print("loaded")
#print(keys[0])# in
print(model.vocab[s if s in keys else 'else']) #Vocab(count:299567, index:0)

#print("vocab size: %d" % len(model.vocab)) #299567

#print(model.syn0.shape) #299567,300

# emb = np.array(model.syn0)
# print(emb.shape)



