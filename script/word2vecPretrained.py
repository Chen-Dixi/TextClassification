from gensim.models import word2vec
from gensim.models import KeyedVectors

slim_filename = 'data/word2vec-slim/GoogleNews-vectors-negative300-SLIM.bin.gz'


model = KeyedVectors.load_word2vec_format(slim_filename, binary=True)

print("vocab size: %d" % len(model.vocab) ) #299567

