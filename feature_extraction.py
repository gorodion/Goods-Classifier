import gensim
import numpy as np
import pandas as pd

def word_averaging(wv, words):
    mean = np.zeros((wv.vector_size,))
    
    for word in words:
        mean += wv.get_vector(word)

    mean = gensim.matutils.unitvec(mean)
    return mean

def word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, review) for review in text_list])
  
  
def vectorize(test: pd.Series):
    test = test.apply(lambda x: [i for i in x.split() if len(i) > 1])
    model = FastText.load('ft.model', mmap='r')
    X_wv = word_averaging_list(model.wv, test)
    X_wv = X_wv.astype('float16')
    return X_wv
