# Predictions have been done and saved in ``BERT/predictions`` and ``models/predictions``.

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def load_data():
    with open('../data/data.pkl', 'rb') as f:
        corpus = pickle.load(f)
    return corpus

# Import predictions from BERT models
predictions_bert_cased_large = torch.load('../BERT/predictions/pred_cased_large.bin')
val_predictions_bert_cased_large = torch.load('../BERT/predictions/pred_val_cased_large.bin')

# Import predictions from keras/pytorch models. Each file contains predictions from three models (stored sequentially)
predictions_models = np.load('../models/predictions/pred_train.npy')
val_predictions_models = np.load('../models/predictions/pred_val.npy')

corpus = load_data()
y = [i[-1] for i in corpus]
y_train, y_test = train_test_split(y, stratify=y, random_state=42, test_size=0.2)

predictions = [(predictions_bert_cased_large>0.5).astype(int)]
for i in predictions_models:
	predictions.append(np.argmax(i, axis=1))

val_predictions = [(val_predictions_bert_cased_large>0.5).astype(int)]
for i in val_predictions_models:
	val_predictions.append(np.argmax(i, axis=1))

for i in predictions:
	print(accuracy_score(y_train, i))

print('---------------------------')

for i in val_predictions:
	print(accuracy_score(y_test, i))

print('---------------------------')

print(accuracy_score(y_test, (np.mean(val_predictions, axis=0)>0.5).astype(int)))