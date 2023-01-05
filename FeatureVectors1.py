import gensim
from gensim.models import Word2Vec
import Levenshtein
import pandas as pd

df = pd.read_json('data/Reviews_Cell_Phones_Accessories_5.json' , lines = True)
# print(df.head())
# print(gensim.utils.simple_preprocess(df.reviewText[0]))

review_text = df.reviewText.apply(gensim.utils.simple_preprocess)

"""
word2Vec = Word2Vec(
    # number of words before and after the target word
    window = 10,
    # minimum number of words in a sentence to process it
    min_count = 2,
    # number of processors/multithreading
    workers = 4
)

# progress_per: After how many processed words should the progress be updated
word2Vec.build_vocab(review_text , progress_per = 1000)
print(word2Vec.epochs)

word2Vec.train(review_text,
            total_examples = word2Vec.corpus_count,
            epochs = word2Vec.epochs)
"""
# word2Vec.save('models/word2vec_amazon_cell_accessories_reviews_short.model')
word2Vec = Word2Vec.load('models/word2vec_amazon_cell_accessories_reviews_short.model')
print(word2Vec.wv.similarity(w1 = 'cheap' , w2 = 'expensive'))
print(word2Vec.wv.similarity(w1 = 'good' , w2 = 'great'))
