from preprocessing import normalize
from feature_extraction import vectorize
import pandas as pd
from gensim.models import FastText
import pickle

test = pd.read_parquet('data/task1_test_for_user.parquet')
pipe = pickle.load(open('clf_task1', 'rb'))

test.item_name = normalize(test.item_name)
X_wv = vectorize(test.item_name)
pred = pipe.predict(X_wv)

res = pd.DataFrame(pred, columns=['pred'])
res['id'] = test['id']

res[['id', 'pred']].to_csv('answers.csv', index=None)
