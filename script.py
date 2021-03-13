from normalizationR import normalize
import pandas as pd
import gensim
from gensim.models import Word2Vec
import pickle
import numpy as np

test = pd.read_parquet('data/task1_test_for_user.parquet')
test.item_name = normalize(test.item_name)

test.item_name = test.item_name.apply(lambda x: [i for i in x.split() if len(i) > 1])
model = Word2Vec.load('w2v.model', mmap='r')

logs = []
def word_averaging(wv, words):
    mean = np.zeros((wv.vector_size,))
    
    for word in words:
        if word in wv.vocab:
            mean += model.wv.get_vector(word)
        else:
            logs.append(word)

    if all(mean == 0.):
        return mean

    mean = gensim.matutils.unitvec(mean)
    return mean

def word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, review) for review in text_list])

X_wv = word_averaging_list(model.wv, test.item_name)

pipe = pickle.load(open('clf_task1', 'rb'))

pred = pipe.predict(X_wv)

res = pd.DataFrame(pred, columns=['pred'])
res['id'] = test['id']

res[['id', 'pred']].to_csv('answers.csv', index=None)
