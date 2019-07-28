import numpy as np
import pickle
import string
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_sequence
import keras
from keras import regularizers, optimizers
from keras.models import Sequential, load_model
from keras.models import *
from keras.layers import Dense, TimeDistributed, GRU, Bidirectional, Dropout
from keras.layers import merge, Multiply, concatenate
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHeadAttention
from keras_tqdm import TQDMNotebookCallback

#--------------------------PRE-PROCESSING AND TRANSLITERATE-------------------------------#
 
def load_data():
    with open('../data/data.pkl', 'rb') as f:
        corpus = pickle.load(f)
    return corpus

def remove_links():
    corpus_clean = []
    for l, i in enumerate(corpus):
        flag=[0, 0]
        for j, k in enumerate(i[:-1]):
            if k[0]=='twitter':
                try:
                    temp = k[0]+i[j+1][0]+i[j+2][0][:3]
                    if temp=="twitter.com":
                        try:
                            if i[j-2][0][-3:]=='pic' and len(i[j-2][0])!=3:
                                flag=[2, j]
                            elif i[j-2][0]=='pic':
                                flag=[1, j]

                        except:
                            pass
                except:
                    pass
        if flag[0]==0:
            corpus_clean.append(i)
        elif flag[0]==1:
            corpus_clean.append(i[:flag[1]-2]+i[flag[1]+3:])
        else:
            new=(i[flag[1]-2][0], 'Hi')
            corpus_clean.append(i[:flag[1]-2]+[new]+i[flag[1]+3:])
    corpus_clean1 = []
    for d, i in enumerate(corpus_clean):
        flag=[0, 0]
        for l, j in enumerate(i[:-1]):
            if 'twitter.com' in j[0]:
                flag=[1, l]
        if flag[0]==0:
            corpus_clean1.append(i)
        else:
            new = i[flag[1]][0].split('pic')
            if new[0]=='':
                corpus_clean1.append(i[:flag[1]]+i[flag[1]+1:])
            else:
                new=(new[0], i[flag[1]][1])             
                corpus_clean1.append(i[:flag[1]]+[new]+i[flag[1]+1:])
    corpus_clean2 = []
    for i in corpus_clean1:
        flag=[0, 0]
        for k, j in enumerate(i[:-1]):
            if 'pic' in j[0] and j[1]=='Hi':
                temp = j[0].split('pic')
                if temp[0]!='' and temp[-1]=='':
                    flag=[1, k]
                    break
        if flag[0]==0:
            corpus_clean2.append(i)
        else:
            temp = i[flag[1]][0].split('pic')
            corpus_clean2.append(i[:flag[1]] + [(temp[0], 'Hi')] + i[flag[1]+1:])
    corpus_clean3 = []
    for i in corpus_clean2:
        corpus_clean3.append(i[:-1])
        
    return corpus_clean3
def remove_empty():
    corpus1 = []
    for i in corpus_new:
        temp = []
        for j in i:
            if j[0]=='':
                pass
            else:
                temp.append(j)
        corpus1.append(temp)
    return corpus1
def remove_punc_lower():
    corpus2 = []
    x= lambda e : (e[0].lower(), e[1])
    punc = list(string.punctuation)
    punc.remove('#')
    punc.remove('@')
    corpus1 = []
    for i in corpus_new:
        temp = []
        for j in i:
            if j[1]=='Ot' and j[0][0]=='#':
                q = re.sub("([a-z])([A-Z])","\g<1> \g<2>",j[0][1:])
                for i in q.split(' '):
                    temp.append((i, 'En'))
            elif j[1]!='Ot':
                temp.append(j)
        corpus1.append(temp)   
    for i in corpus1:
        temp = []
        for j in i:
            temp.append(x(j))
        corpus2.append(temp)
    return corpus2       
def trans_literate():
    with open('/hindi_dict.txt') as f:
        hindi_trans = f.read()
    hindi_trans = hindi_trans.split('\n')
    hindi_trans = [i for i in hindi_trans if i!='']
    hindi_trans = [[i for i in j.split(' ') if i!=''] for j in hindi_trans]
    for i in hindi_trans:
        if len(i)!=2: #unusable words
            hindi_trans.remove(i)
    hindi_trans_dict = {}
    for i in hindi_trans:
        hindi_trans_dict[i[0]] = i[1]
    corpus_trans = []
    for i in corpus_new:
        temp = []
        for j in i:
            if j[1]=='Hi' and j[0] in hindi_trans_dict:
                temp.append((hindi_trans_dict[j[0]], 'Hi'))
            else:
                temp.append((j[0], 'En'))
        corpus_trans.append(temp)  
    return corpus_trans
def remove_stopwords():
    !wget 'https://raw.githubusercontent.com/Alir3z4/stop-words/master/hindi.txt'
    with open('hindi.txt') as f:
        stopwords_hi = f.read()    
    stop_words_hi = set(stopwords_hi.split('\n'))    
    stop_words_en = set(stopwords.words('english'))
    corpus2 = []
    for i in corpus_new:
        temp = []
        for j in i:
            if j[1]=='Hi' and j[0] in stop_words_hi:
                pass
            elif j[1]=='En' and j[0] in stop_words_en:
                pass
            else:
                temp.append(j)
        corpus2.append(temp)
    return corpus2

corpus = load_data()
corpus_new = remove_links()
corpus_new = remove_empty()
corpus_new = remove_punc_lower() 
corpus_new = remove_empty()
corpus_new = trans_literate()
corpus_new = remove_empty()
corpus_new = remove_stopwords()
corpus_new = remove_empty()
 

#---------------------------------------IMPORT EMBEDDINGS-----------------------------------#


def import_embeddings():
    ''' BERT and ELMO embeddings are imported (generated by ``generate_embeddings.py``).
    These embeddings are then stacked to create staked embeddings. Now all the three 
    embeddings are further appended with language tags (whether word in english or Hindi)
    using 1 for English and -1 for Hindi.
    
    Then all the obtained sentences embeddings are then:
    - truncated (post) to MAX_LEN of 20 words
    - pre-padded
    
    The obtained are stratified-split into train and test.

    The final embeddings generated are:
    - BERT with append (X1)
    - ELMO with append (X2)
    - Stacked with append (X3)
    '''

    # BERT AND ELMO EMBEDDINGS
    emb_bert = np.load('../Embeddings/bert_final.npy', allow_pickle=True)
    emb_elmo = np.load('../Embeddings/elmo_final.npy', allow_pickle=True)

    # TARGET
    y_temp =  np.array([i[-1] for i in corpus])
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(y_temp.reshape(-1, 1))
    y = enc.transform(y_temp.reshape(-1, 1)).toarray()

    # STACKED EMBEDDINGS
    emb_comb = []
    for i,j in zip(emb_elmo, emb_bert):
        temp = []
        for k, l in zip(i, j):
            temp.append(np.hstack((k, l)))
        emb_comb.append(temp)

    # APPENDING LANGUAGE TAGS FOR STACKED EMBEDDINGS
    hi_en = {'Hi':1, 'En':-1}
    emb_comb_app = []
    for i, j in zip(emb_comb, corpus_new):
        temp = [hi_en[k[1]] for k in j]
        a = np.array(temp).reshape((len(temp), 1))
        b = np.array(i)
        emb_comb_app.append(np.hstack((a, b)))

    X3_train, X3_test, y3_train, y3_test = train_test_split(emb_comb_app, y, test_size=0.2, random_state=42, stratify=y)        

    emb_list = []
    for i in X3_train:
        emb_list.append(torch.tensor(i[:20]).cuda())
    X3_train = pad_sequence(emb_list, batch_first=True)

    emb_list = []
    for i in X3_test:
        emb_list.append(torch.tensor(i[:20]).cuda())
    X3_test = pad_sequence(emb_list, batch_first=True)


    # ELMO AND BERT WITH LANGUAGE TAGS APPENDS
    hi_en = {'Hi':1, 'En':-1}

    emb_elmo_app = []
    for i, j in zip(emb_elmo, corpus_new):
        temp = [hi_en[k[1]] for k in j]
        a = np.array(temp).reshape((len(temp), 1))
        b = np.array(i)
        emb_elmo_app.append(np.hstack((a, b)))

    emb_bert_app = []
    for i, j in zip(emb_bert, corpus_new):
        temp = [hi_en[k[1]] for k in j]
        a = np.array(temp).reshape((len(temp), 1))
        b = np.array(i)
        emb_bert_app.append(np.hstack((a, b)))    

    X1_train_app, X1_test_app, X2_train_app, X2_test_app, y_train_app, y_test_app = train_test_split(emb_bert_app, emb_elmo_app, y, test_size=0.2, random_state=42, stratify=y)    

    emb_list = []
    for i in X1_train_app:
        emb_list.append(torch.tensor(i[:20]).cuda())
    X1_train_app = pad_sequence(emb_list, batch_first=True)

    emb_list = []
    for i in X1_test_app:
        emb_list.append(torch.tensor(i[:20]).cuda())
    X1_test_app = pad_sequence(emb_list, batch_first=True)

    emb_list = []
    for i in X2_train_app:
        emb_list.append(torch.tensor(i[:20]).cuda())
    X2_train_app = pad_sequence(emb_list, batch_first=True)

    emb_list = []
    for i in X2_test_app:
        emb_list.append(torch.tensor(i[:20]).cuda())
    X2_test_app = pad_sequence(emb_list, batch_first=True)
    # ELMO AND BERT NO APPENDS

    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(emb_bert, emb_elmo, y, test_size=0.2, random_state=42, stratify=y)

    emb_list = []
    for i in X1_train:
        emb_list.append(torch.tensor(i[:20]).cuda())
    X1_train = pad_sequence(emb_list, batch_first=True)
    emb_list = []
    for i in X1_test:
        emb_list.append(torch.tensor(i[:20]).cuda())
    X1_test = pad_sequence(emb_list, batch_first=True)

    emb_list = []
    for i in X2_train:
        emb_list.append(torch.tensor(i[:20]).cuda())
    X2_train = pad_sequence(emb_list, batch_first=True)
    emb_list = []
    for i in X2_test:
        emb_list.append(torch.tensor(i[:20]).cuda())
    X2_test = pad_sequence(emb_list, batch_first=True)

    return X1_train, X1_test, X2_train, X2_test, y_train, y_test, X1_train_app, X1_test_app, X2_train_app, X2_test_app, y_train_app, y_test_app, X3_train, X3_test, y3_train, y3_test

''' 1: ELMO
    2: BERT
    3: Stacked ELMO and BERT
    app: Language tags were added'''

X1_train, X1_test, X2_train, X2_test, y_train, y_test, X1_train_app, X1_test_app, X2_train_app, X2_test_app, y_train_app, y_test_app, X3_train, X3_test, y3_train, y3_test = import_embeddings()

#------------------------------FUNCTIONS TO GENERATE BEST MODELS----------------------#

def good_models(emb_len, optimizer, regulizer):
    model = Sequential()
    model.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, emb_len), merge_mode='concat'))
    if regulizer==True:
        model.add(SeqSelfAttention(
            attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
            attention_activation='sigmoid',
            use_attention_bias=True,
            kernel_regularizer=keras.regularizers.l2(1e-6),
            bias_regularizer=keras.regularizers.l1(1e-6),
            attention_regularizer_weight=1e-6,
            name='Attention'))
    else:        
        model.add(SeqSelfAttention(
            attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
            attention_activation=None,
            kernel_regularizer=keras.regularizers.l2(1e-6),
            use_attention_bias=False,
            name='Attention',
        ))
    model.add(Bidirectional(LSTM(5, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
    model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
Adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

def return_model(key):
    if key==1:
        return good_models(len(X2_train_app[0][0]), sgd, True)
    if key==2:
        return good_models(len(X2_train_app[0][0]), sgd, False)
    else:
        return good_models(len(X2_train_app[0][0]), Adadelta, False)

#--------------------------------------------ENSEMBLE MODELS--------------------------------------------#
model1 = return_model(1)
model2 = return_model(2)
model3 = return_model(3)

model1.fit(x=X2_app.cpu().numpy(), y=y, validation_split=0.2, batch_size=64, epochs=100, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)])    
model2.fit(x=X2_app.cpu().numpy(), y=y, validation_split=0.2, batch_size=64, epochs=100, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)])    
model3.fit(x=X2_app.cpu().numpy(), y=y, validation_split=0.2, batch_size=64, epochs=100, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)])    

pred1 = model1.predict(x=X2_app.cpu().numpy())
pred2 = model2.predict(x=X2_app.cpu().numpy())
pred3 = model3.predict(x=X2_app.cpu().numpy())

######### MODEL 1
predx = model2x.predict(X1_train.cpu().numpy())
predy = model2y.predict(X2_train.cpu().numpy())
predx_t = model2x.predict(X1_test.cpu().numpy())
predy_t = model2y.predict(X2_test.cpu().numpy())

input_comb_1 = Input(shape=(2, ))
input_comb_2 = Input(shape=(2, ))
combined = concatenate([input_comb_1, input_comb_2])
z = Dense(4, activation="sigmoid")(combined)
z = Dense(2, activation="sigmoid")(z)
model2_final = Model(inputs=[input_comb_1, input_comb_2], outputs=z)

model2_final.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model2_final.fit(x=[predx_t[:550], predy_t[:550]], y=y_test[:550], validation_data=([predx_t[550:], predy_t[550:]], y_test[550:]),	batch_size=64, epochs=150, shuffle=True, verbose=1)

"""Epoch 150/150 <br>
550/550 [==============================] - 0s 60us/step - loss: 0.5369 - acc: 0.7373 - val_loss: 0.5410 - val_acc: 0.7425
"""


####### MODEL 2

model1 = return_model(1)
model2 = return_model(2)
model3 = return_model(3)


inputs_layer_1 = Input(shape=(2, ))
inputs_layer_2 = Input(shape=(2, ))
inputs_layer_3 = Input(shape=(2, ))

combined = concatenate([inputs_layer_1, inputs_layer_2, inputs_layer_3])
combined = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(combined)
model = Model(inputs=[inputs_layer_1, inputs_layer_2, inputs_layer_3], outputs=combined)
sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


model.fit(x=[i for i in predictions], y=y_test, validation_split=0.2, batch_size=64, epochs=50, shuffle=True)

"""
Epoch 50/50 <br>
547/547 [==============================] - 0s 92us/step - loss: 1.1616 - acc: 0.7422 - val_loss: 1.1401 - val_acc: 0.7372
very consistent no variation
"""