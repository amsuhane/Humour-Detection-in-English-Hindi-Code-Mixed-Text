import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re

import numpy as np
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_sequence

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import GRU
from keras.layers import Bidirectional
from keras import regularizers
from keras import optimizers
from keras.layers import merge, Multiply
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.layers import concatenate
import keras
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.layers import Dropout
from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHead
from keras_multi_head import MultiHeadAttention
from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import EarlyStopping
from keras import regularizers


#--------------------------PRE-PROCESSING AND TRANSLITERATE-------------------------------#
 
def load_data():
    with open('../Data/data.pkl', 'rb') as f:
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

#------------------------------------MODELS------------------------------------------#
 

"""#### Basic BiLSTM, no appends

**MODEL 0.0**:

*   BERT
*   2 LSTM layer
*   4 dense
*   2 dense
"""

model_Bi_LSTM_1 = Sequential()
model_Bi_LSTM_1.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, 3072), merge_mode='concat'))
model_Bi_LSTM_1.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
model_Bi_LSTM_1.add(Dense(5, activation='softmax'))
model_Bi_LSTM_1.add(Dense(2, activation='softmax'))
model_Bi_LSTM_1.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_Bi_LSTM_1.fit(x=X1_train.cpu().numpy(), y=y_train, validation_data=(X1_test.cpu().numpy(), y_test),	batch_size=64, epochs=100, shuffle=True)

"""Epoch 13/100 <br>
2734/2734 [==============================] - 7s 3ms/step - loss: 0.5630 - acc: 0.7253 - val_loss: 0.5875 - val_acc: 0.7003

After this val-acc remains nearly same but val-loss increase

**MODEL 0.1**:

*   ELMO
*   2 LSTM layer
*   4 dense
*   2 dense
"""

model_Bi_LSTM_2 = Sequential()
model_Bi_LSTM_2.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, 1024), merge_mode='concat'))
model_Bi_LSTM_2.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
model_Bi_LSTM_2.add(Dense(5, activation='softmax'))
model_Bi_LSTM_2.add(Dense(2, activation='softmax'))
model_Bi_LSTM_2.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_Bi_LSTM_2.fit(x=X2_train.cpu().numpy(), y=y_train, validation_data=(X2_test.cpu().numpy(), y_test),	batch_size=64, epochs=100, shuffle=True)

"""Epoch 19/100 <br>
2734/2734 [==============================] - 7s 2ms/step - loss: 0.5190 - acc: 0.7655 - val_loss: 0.5491 - val_acc: 0.7310

After this val-acc remains nearly same but val-loss increase

#### Self written attention, no appends

**MODEL 0.3**:

*   BERT
*   1 LSTM layer
*   Attention
*   64 dense
*   16 dense
*   4 dense
"""

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    #output_attention_mul = concatenate([inputs, a_probs], mode='mul', name='attention_mul')
    output_attention_mul = Multiply()([inputs, a_probs])

    return output_attention_mul

TIME_STEPS, INPUT_DIM, lstm_units, SINGLE_ATTENTION_VECTOR = 20, 3072, 10, False

inputs_layer = Input(shape=(TIME_STEPS, INPUT_DIM, ))
lstm_out = Bidirectional(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(inputs_layer)
attention_mul = attention_3d_block(lstm_out)
attention_mul = Flatten()(attention_mul)
output = Dense(32, activation="sigmoid")(attention_mul)
output = Dropout(0.5)(output)
output = Dense(2, activation="sigmoid")(output)
model_att_1 = Model(inputs=[inputs_layer], outputs=output)

model_att_1.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model_att_1.fit(x=X1_train.cpu().numpy(), y=y_train, validation_data=(X1_test.cpu().numpy(), y_test),	batch_size=64, epochs=100, shuffle=True)

"""Epoch 25/100 <br>
2734/2734 [==============================] - 4s 1ms/step - loss: 0.4917 - acc: 0.7708 - val_loss: 0.5805 - val_acc: 0.6886

After that ....

**MODEL 0.3**:

*   ELMO
*   1 LSTM layer
*   Attention
*   64 dense
*   16 dense
*   4 dense
"""

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    #output_attention_mul = concatenate([inputs, a_probs], mode='mul', name='attention_mul')
    output_attention_mul = Multiply()([inputs, a_probs])

    return output_attention_mul

TIME_STEPS, INPUT_DIM, lstm_units, SINGLE_ATTENTION_VECTOR = 20, 1024, 10, False

inputs_layer = Input(shape=(TIME_STEPS, INPUT_DIM, ))
lstm_out = Bidirectional(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(inputs_layer)
attention_mul = attention_3d_block(lstm_out)
attention_mul = Flatten()(attention_mul)
output = Dense(64, activation="sigmoid")(attention_mul)
output = Dense(16, activation="sigmoid")(output)
output = Dense(2, activation="sigmoid")(output)
model_att_1 = Model(inputs=[inputs_layer], outputs=output)

model_att_1.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model_att_1.fit(x=X2_train.cpu().numpy(), y=y_train, validation_data=(X2_test.cpu().numpy(), y_test),	batch_size=64, epochs=100, shuffle=True)

"""Epoch 14/100 <br>
2734/2734 [==============================] - 3s 1ms/step - loss: 0.4834 - acc: 0.7751 - val_loss: 0.5299 - val_acc: 0.7515

After that ....

**MODEL 0.5**:

*   ELMO
*   2 LSTM layer
*   Attention
*   64 dense
*   16 dense
*   4 dense
"""

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    #output_attention_mul = concatenate([inputs, a_probs], mode='mul', name='attention_mul')
    output_attention_mul = Multiply()([inputs, a_probs])

    return output_attention_mul

TIME_STEPS, INPUT_DIM, lstm_units, SINGLE_ATTENTION_VECTOR = 20, 1024, 10, False

inputs_layer = Input(shape=(TIME_STEPS, INPUT_DIM, ))
lstm_out = Bidirectional(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(inputs_layer)
lstm_out = Bidirectional(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(lstm_out)
attention_mul = attention_3d_block(lstm_out)
attention_mul = Flatten()(attention_mul)
output = Dense(64, activation="sigmoid")(attention_mul)
#output = Dropout(0.2)(output)
output = Dense(16, activation="sigmoid")(output)
output = Dense(2, activation="sigmoid")(output)
model_att_1 = Model(inputs=[inputs_layer], outputs=output)

model_att_1.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model_att_1.fit(x=X2_train.cpu().numpy(), y=y_train, validation_data=(X2_test.cpu().numpy(), y_test),	batch_size=64, epochs=100, shuffle=True)

"""Epoch 16/100<br>
2734/2734 [==============================] - 7s 3ms/step - loss: 0.5042 - acc: 0.7705 - val_loss: 0.5512 - val_acc: 0.7405

After that ...
"""



"""#### Self Attention Library, no appends

**MODEL 1.1 SelfAtt**
"""

model_Bi_LSTM_att1 = Sequential()
model_Bi_LSTM_att1.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, 1024), merge_mode='concat'))
model_Bi_LSTM_att1.add(SeqSelfAttention(attention_activation='sigmoid'))
model_Bi_LSTM_att1.add(Bidirectional(LSTM(5, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
#model_Bi_LSTM_att1.add(Dense(5, activation='softmax'))
model_Bi_LSTM_att1.add(Dense(2, activation='softmax'))
model_Bi_LSTM_att1.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_Bi_LSTM_att1.fit(x=X2_train.cpu().numpy(), y=y_train, validation_data=(X2_test.cpu().numpy(), y_test),	batch_size=64, epochs=25, shuffle=True)

"""lstm: 8, 4
Dense: 2 <br>
Epoch 22/50<br>
2734/2734 [==============================] - 7s 3ms/step - loss: 0.4764 - acc: 0.7747 - val_loss: 0.5205 - val_acc: 0.7412

lstm: 16, 16
Dense: 5, 2 <br>
Epoch 15/50<br>
2734/2734 [==============================] - 7s 3ms/step - loss: 0.5087 - acc: 0.7612 - val_loss: 0.5395 - val_acc: 0.7325

lstm: 10, 5
Dense: 2 <br>
Epoch 15/25<br>
2734/2734 [==============================] - 8s 3ms/step - loss: 0.4916 - acc: 0.7615 - val_loss: 0.5077 - val_acc: 0.7500

**MODEL 1.2 SelfAtt Local**
"""

model_Bi_LSTM_att2 = Sequential()
model_Bi_LSTM_att2.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, 1024), merge_mode='concat'))
model_Bi_LSTM_att2.add(SeqSelfAttention(
    attention_width=15,
    attention_activation='sigmoid',
    name='Attention',
))
model_Bi_LSTM_att2.add(Bidirectional(LSTM(5, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
#model_Bi_LSTM_att1.add(Dense(5, activation='softmax'))
model_Bi_LSTM_att2.add(Dense(2, activation='softmax'))
model_Bi_LSTM_att2.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_Bi_LSTM_att2.fit(x=X2_train.cpu().numpy(), y=y_train, validation_data=(X2_test.cpu().numpy(), y_test),	batch_size=64, epochs=25, shuffle=True)

"""10, 5 LSTM ; 2 Dense, width=15 <br>
Epoch 18/25<br>
2734/2734 [==============================] - 7s 3ms/step - loss: 0.4813 - acc: 0.7663 - val_loss: 0.5122 - val_acc: 0.7500

**Model 1.3 Multiplicative**
"""

model_Bi_LSTM_att3 = Sequential()
model_Bi_LSTM_att3.add(Bidirectional(LSTM(8, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, 1024), merge_mode='concat'))
model_Bi_LSTM_att3.add(SeqSelfAttention(
    attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
    attention_activation=None,
    kernel_regularizer=k.regularizers.l2(1e-6),
    use_attention_bias=False,
    name='Attention',
))
model_Bi_LSTM_att3.add(Bidirectional(LSTM(4, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
#model_Bi_LSTM_att1.add(Dense(5, activation='softmax'))
model_Bi_LSTM_att3.add(Dense(4, activation='softmax'))
model_Bi_LSTM_att3.add(Dense(2, activation='softmax'))
model_Bi_LSTM_att3.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

model_Bi_LSTM_att3.fit(x=X2_train.cpu().numpy(), y=y_train, validation_data=(X2_test.cpu().numpy(), y_test),	batch_size=64, epochs=25, shuffle=True)

"""8, Att, 4 LSTM, 4, 2, Dense <br>
Epoch 11/25 <br>
2734/2734 [==============================] - 7s 3ms/step - loss: 0.5486 - acc: 0.7282 - val_loss: 0.5398 - val_acc: 0.7354

10, Att(width=None), 5 LSTM, 4, 2 Dense <Br>
Epoch 23/25 <br>
2734/2734 [==============================] - 7s 3ms/step - loss: 0.4804 - acc: 0.7783 - val_loss: 0.5256 - val_acc: 0.7485

10, att (width=15), 5 LSTM, 2, Dense <br>
Epoch 16/25 <br>
2734/2734 [==============================] - 7s 3ms/step - loss: 0.4898 - acc: 0.7593 - val_loss: 0.5189 - val_acc: 0.7500

10, att (width=None), 5 LSTM, 2, Dense <br>
Epoch 13/25<br>
2734/2734 [==============================] - 8s 3ms/step - loss: 0.4994 - acc: 0.7604 - val_loss: 0.5188 - val_acc: 0.7588

**Model 1.3 Regulizer**
"""

model_Bi_LSTM_att3 = Sequential()
model_Bi_LSTM_att3.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, 1024), merge_mode='concat'))
model_Bi_LSTM_att3.add(SeqSelfAttention(
    attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
    attention_activation='sigmoid',
    use_attention_bias=True,
    kernel_regularizer=k.regularizers.l2(1e-6),
    bias_regularizer=k.regularizers.l1(1e-6),
    attention_regularizer_weight=1e-6,
    name='Attention'))
model_Bi_LSTM_att3.add(Bidirectional(LSTM(5, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
#model_Bi_LSTM_att1.add(Dense(5, activation='softmax'))
model_Bi_LSTM_att3.add(Dense(2, activation='softmax'))
model_Bi_LSTM_att3.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_Bi_LSTM_att3.fit(x=X2_train.cpu().numpy(), y=y_train, validation_data=(X2_test.cpu().numpy(), y_test),	batch_size=64, epochs=25, shuffle=True)

"""Epoch 21/25 <br>
2734/2734 [==============================] - 8s 3ms/step - loss: 0.4582 - acc: 0.7809 - val_loss: 0.5175 - val_acc: 0.7485

#### Basic BiLSTM, with appends

**MODEL 3**
"""

model_Bi_LSTM_1 = Sequential()
model_Bi_LSTM_1.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, 3073), merge_mode='concat'))
model_Bi_LSTM_1.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
model_Bi_LSTM_1.add(Dense(5, activation='softmax'))
model_Bi_LSTM_1.add(Dense(2, activation='softmax'))
model_Bi_LSTM_1.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_Bi_LSTM_1.fit(x=X1_train_app.cpu().numpy(), y=y_train_app, validation_data=(X1_test_app.cpu().numpy(), y_test_app),	batch_size=64, epochs=30, shuffle=True)

"""Epoch 24/30 <br>
2734/2734 [==============================] - 11s 4ms/step - loss: 0.4852 - acc: 0.7740 - val_loss: 0.5843 - val_acc: 0.7149
"""

model_Bi_LSTM_1 = Sequential()
model_Bi_LSTM_1.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, 1025), merge_mode='concat'))
model_Bi_LSTM_1.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
model_Bi_LSTM_1.add(Dense(5, activation='softmax'))
model_Bi_LSTM_1.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_Bi_LSTM_1.fit(x=X2_train_app.cpu().numpy(), y=y_train_app, validation_data=(X2_test_app.cpu().numpy(), y_test_app),	batch_size=64, epochs=30, shuffle=True)

"""Epoch 18/30 <br>
2734/2734 [==============================] - 10s 4ms/step - loss: 0.5144 - acc: 0.7516 - val_loss: 0.5267 - val_acc: 0.7412

#### Self written attention, with appends

*   BERT
*   1 LSTM layer
*   Attention
*   64 dense
*   16 dense
*   4 dense
"""

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    #output_attention_mul = concatenate([inputs, a_probs], mode='mul', name='attention_mul')
    output_attention_mul = Multiply()([inputs, a_probs])

    return output_attention_mul

TIME_STEPS, INPUT_DIM, lstm_units, SINGLE_ATTENTION_VECTOR = 20, 3073, 10, False

inputs_layer = Input(shape=(TIME_STEPS, INPUT_DIM, ))
lstm_out = Bidirectional(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(inputs_layer)
attention_mul = attention_3d_block(lstm_out)
attention_mul = Flatten()(attention_mul)
output = Dense(32, activation="sigmoid")(attention_mul)
output = Dropout(0.5)(output)
output = Dense(2, activation="sigmoid")(output)
model_att_1 = Model(inputs=[inputs_layer], outputs=output)

model_att_1.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model_att_1.fit(x=X1_train_app.cpu().numpy(), y=y_train_app, validation_data=(X1_test_app.cpu().numpy(), y_test_app),	batch_size=64, epochs=35, shuffle=True)

"""Epoch 35/35 <br>
2734/2734 [==============================] - 6s 2ms/step - loss: 0.3714 - acc: 0.8403 - val_loss: 0.7313 - val_acc: 0.6944

**MODEL 0.3**:

*   ELMO
*   1 LSTM layer
*   Attention
*   64 dense
*   16 dense
*   4 dense
"""

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    #output_attention_mul = concatenate([inputs, a_probs], mode='mul', name='attention_mul')
    output_attention_mul = Multiply()([inputs, a_probs])

    return output_attention_mul

TIME_STEPS, INPUT_DIM, lstm_units, SINGLE_ATTENTION_VECTOR = 20, 1025, 10, False

inputs_layer = Input(shape=(TIME_STEPS, INPUT_DIM, ))
lstm_out = Bidirectional(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(inputs_layer)
attention_mul = attention_3d_block(lstm_out)
attention_mul = Flatten()(attention_mul)
output = Dense(64, activation="sigmoid")(attention_mul)
output = Dense(16, activation="sigmoid")(output)
output = Dense(2, activation="sigmoid")(output)
model_att_1 = Model(inputs=[inputs_layer], outputs=output)

model_att_1.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

model_att_1.fit(x=X2_train_app.cpu().numpy(), y=y_train_app, validation_data=(X2_test_app.cpu().numpy(), y_test_app),	batch_size=64, epochs=35, shuffle=True)

"""Epoch 13/35 <br>
2734/2734 [==============================] - 5s 2ms/step - loss: 0.5056 - acc: 0.7524 - val_loss: 0.5258 - val_acc: 0.7522

**MODEL 0.4**:

*   ELMO
*   2 LSTM layer
*   Attention
*   64 dense
*   16 dense
*   4 dense
"""

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    #output_attention_mul = concatenate([inputs, a_probs], mode='mul', name='attention_mul')
    output_attention_mul = Multiply()([inputs, a_probs])

    return output_attention_mul

TIME_STEPS, INPUT_DIM, lstm_units, SINGLE_ATTENTION_VECTOR = 20, 1025, 10, False

inputs_layer = Input(shape=(TIME_STEPS, INPUT_DIM, ))
lstm_out = Bidirectional(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(inputs_layer)
lstm_out = Bidirectional(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(lstm_out)
attention_mul = attention_3d_block(lstm_out)
attention_mul = Flatten()(attention_mul)
output = Dense(64, activation="sigmoid")(attention_mul)
#output = Dropout(0.2)(output)
output = Dense(16, activation="sigmoid")(output)
output = Dense(2, activation="sigmoid")(output)
model_att_1 = Model(inputs=[inputs_layer], outputs=output)

model_att_1.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model_att_1.fit(x=X2_train_app.cpu().numpy(), y=y_train_app, validation_data=(X2_test_app.cpu().numpy(), y_test_app),	batch_size=64, epochs=35, shuffle=True)

"""Epoch 17/35
2734/2734 [==============================] - 10s 4ms/step - loss: 0.4955 - acc: 0.7712 - val_loss: 0.5300 - val_acc: 0.7429

#### Self Attention Library, with appends

**MODEL 1.1 SelfAtt**
"""

model_Bi_LSTM_att1 = Sequential()
model_Bi_LSTM_att1.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, 1025), merge_mode='concat'))
model_Bi_LSTM_att1.add(SeqSelfAttention(attention_activation='sigmoid'))
model_Bi_LSTM_att1.add(Bidirectional(LSTM(5, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
#model_Bi_LSTM_att1.add(Dense(5, activation='softmax'))
model_Bi_LSTM_att1.add(Dense(2, activation='softmax'))
model_Bi_LSTM_att1.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_Bi_LSTM_att1.fit(x=X2_train_app.cpu().numpy(), y=y_train_app, validation_data=(X2_test_app.cpu().numpy(), y_test_app), batch_size=64, epochs=35, shuffle=True)

"""lstm: 10, 5
Dense: 2 <br>
Epoch 18/35
2734/2734 [==============================] - 10s 4ms/step - loss: 0.4881 - acc: 0.7670 - val_loss: 0.5125 - val_acc: 0.7427

**MODEL 1.2 SelfAtt Local**
"""

model_Bi_LSTM_att2 = Sequential()
model_Bi_LSTM_att2.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, 1025), merge_mode='concat'))
model_Bi_LSTM_att2.add(SeqSelfAttention(
    attention_width=15,
    attention_activation='sigmoid',
    name='Attention',
))
model_Bi_LSTM_att2.add(Bidirectional(LSTM(5, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
#model_Bi_LSTM_att1.add(Dense(5, activation='softmax'))
model_Bi_LSTM_att2.add(Dense(2, activation='softmax'))
model_Bi_LSTM_att2.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_Bi_LSTM_att2.fit(x=X2_train_app.cpu().numpy(), y=y_train_app, validation_data=(X2_test_app.cpu().numpy(), y_test_app), batch_size=64, epochs=35, shuffle=True)

"""10, att (width=15), 5 LSTM, 2, Dense <br>
Epoch 16/25 <br>
2734/2734 [==============================] - 7s 3ms/step - loss: 0.4898 - acc: 0.7593 - val_loss: 0.5189 - val_acc: 0.7500

10, 5 LSTM ; 2 Dense, width=None, 2, Dense  <br>
Epoch 18/25<br>
2734/2734 [==============================] - 7s 3ms/step - loss: 0.4813 - acc: 0.7663 - val_loss: 0.5122 - val_acc: 0.7500

**Model 1.3 Multiplicative**
"""

model_Bi_LSTM_att3 = Sequential()
model_Bi_LSTM_att3.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, 1025), merge_mode='concat'))
model_Bi_LSTM_att3.add(SeqSelfAttention(
    attention_width=15,
    attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
    attention_activation=None,
    kernel_regularizer=keras.regularizers.l2(1e-6),
    use_attention_bias=False,
    name='Attention',
))
model_Bi_LSTM_att3.add(Bidirectional(LSTM(5, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
#model_Bi_LSTM_att1.add(Dense(5, activation='softmax'))
#model_Bi_LSTM_att3.add(Dense(4, activation='softmax'))
model_Bi_LSTM_att3.add(Dense(2, activation='softmax'))
model_Bi_LSTM_att3.compile(loss='binary_crossentropy', optimizer='RMSProp', metrics=['accuracy'])

model_Bi_LSTM_att3.fit(x=X2_train_app.cpu().numpy(), y=y_train_app, validation_data=(X2_test_app.cpu().numpy(), y_test_app), batch_size=64, epochs=35, shuffle=True)

"""10, att (width=15), 5 LSTM, 2, Dense <br>
Epoch 13/35
2734/2734 [==============================] - 10s 4ms/step - loss: 0.5085 - acc: 0.7480 - val_loss: 0.5082 - val_acc: 0.7515

10, att (width=None), 5 LSTM, 2, Dense <br>
Epoch 14/35 <br>
2734/2734 [==============================] - 10s 4ms/step - loss: 0.5009 - acc: 0.7538 - val_loss: 0.5050 - val_acc: 0.7558

**Model 1.3 Regulizer**
"""

model_Bi_LSTM_att3 = Sequential()
model_Bi_LSTM_att3.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, 1025), merge_mode='concat'))
model_Bi_LSTM_att3.add(SeqSelfAttention(
    attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
    attention_activation='sigmoid',
    use_attention_bias=True,
    kernel_regularizer=keras.regularizers.l2(1e-6),
    bias_regularizer=keras.regularizers.l1(1e-6),
    attention_regularizer_weight=1e-6,
    name='Attention'))
model_Bi_LSTM_att3.add(Bidirectional(LSTM(5, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
#model_Bi_LSTM_att1.add(Dense(5, activation='softmax'))
model_Bi_LSTM_att3.add(Dense(2, activation='softmax'))
model_Bi_LSTM_att3.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_Bi_LSTM_att3.fit(x=X2_train_app.cpu().numpy(), y=y_train_app, validation_data=(X2_test_app.cpu().numpy(), y_test_app), batch_size=64, epochs=25, shuffle=True)

"""Epoch 21/25 <br>
2734/2734 [==============================] - 10s 4ms/step - loss: 0.4503 - acc: 0.8032 - val_loss: 0.5265 - val_acc: 0.7573

#### GRU
"""

model_GRU = Sequential()
model_GRU.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, 1025), merge_mode='concat'))
#model_GRU.add(GRU(8, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
model_GRU.add(GRU(8, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))
#model_Bi_LSTM_1.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
#model_GRU.add(Dense(8, activation='softmax'))
model_GRU.add(Dense(2, activation='softmax'))
model_GRU.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_GRU.fit(x=X2_train_app.cpu().numpy(), y=y_train_app, validation_data=(X2_test_app.cpu().numpy(), y_test_app),	batch_size=64, epochs=30, shuffle=True)

"""Epoch 16/30 <br>
2734/2734 [==============================] - 7s 3ms/step - loss: 0.5261 - acc: 0.7341 - val_loss: 0.5245 - val_acc: 0.7383
"""

model_Bi_LSTM_att3 = Sequential()
model_Bi_LSTM_att3.add(Bidirectional(LSTM(20, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, 1025), merge_mode='concat'))
model_Bi_LSTM_att3.add(SeqSelfAttention(
    attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
    attention_activation='sigmoid',
    use_attention_bias=True,
    kernel_regularizer=k.regularizers.l2(1e-6),
    bias_regularizer=k.regularizers.l1(1e-6),
    attention_regularizer_weight=1e-6,
    name='Attention'))
model_Bi_LSTM_att3.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat'))
model_Bi_LSTM_att3.add(GRU(5, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))
#model_Bi_LSTM_att1.add(Dense(5, activation='softmax'))
model_Bi_LSTM_att3.add(Dense(2, activation='softmax'))
model_Bi_LSTM_att3.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model_Bi_LSTM_att3.fit(x=X2_train_app.cpu().numpy(), y=y_train_app, validation_data=(X2_test_app.cpu().numpy(), y_test_app), batch_size=64, epochs=25, shuffle=True)

"""Epoch 23/25 <br>
2734/2734 [==============================] - 12s 4ms/step - loss: 0.4646 - acc: 0.7860 - val_loss: 0.5448 - val_acc: 0.7427

#### COMBINED EMBEDDINGS
"""

import keras

model_Bi_LSTM_att = Sequential()
model_Bi_LSTM_att.add(Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, 4097), merge_mode='concat'))
model_Bi_LSTM_att.add(SeqSelfAttention(
    attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
    attention_activation='sigmoid',
    use_attention_bias=True,
    kernel_regularizer=keras.regularizers.l2(1e-6),
    bias_regularizer=keras.regularizers.l1(1e-6),
    attention_regularizer_weight=1e-6,
    name='Attention'))
model_Bi_LSTM_att.add(Bidirectional(LSTM(5, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
#model_Bi_LSTM_att1.add(Dense(5, activation='softmax'))
#model_Bi_LSTM_att3.add(Dense(4, activation='softmax'))
model_Bi_LSTM_att.add(Dense(2, activation='softmax'))
model_Bi_LSTM_att.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

model_Bi_LSTM_att.fit(x=X3_train.cpu().numpy(), y=y3_train, validation_data=(X3_test.cpu().numpy(), y3_test), batch_size=64, epochs=35, shuffle=True)

"""#### Multi-Head"""

Adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
emb_len = len(X2_train_app[0][0])

X2_train_app.cpu().numpy().shape

model = keras.models.Sequential()
model.add(Input(shape=(20, 1025, )))
'''
model.add(MultiHead([
    Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), input_shape=(20, emb_len), merge_mode='concat'),
    Bidirectional(LSTM(5, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'),
], name='Multi-RNN'))'''
model.add(Bidirectional(LSTM(5, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
model.add(Dense(2, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
#model.build((2734, 20, 1025))

model.compile(loss='binary_crossentropy', optimizer=Adadelta, metrics=['accuracy'])

TIME_STEPS, INPUT_DIM= 20, len(X2_train_app[0][0])

inputs_layer = Input(shape=(TIME_STEPS, INPUT_DIM_1, ))

x = Bidirectional(LSTM(lstm_units_1, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(inputs_layer_1)
x = Bidirectional(LSTM(lstm_units_2, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))(x)
x = Dense(2, activation="sigmoid")(x)
x = Model(inputs=inputs_layer_1, outputs=x)

y = Bidirectional(LSTM(lstm_units_1, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(inputs_layer_2)
y = Bidirectional(LSTM(lstm_units_2, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))(y)
y = Dense(2, activation="sigmoid")(y)
y = Model(inputs=inputs_layer_2, outputs=y)


model2x = Model(inputs=x.input, outputs=x.output)
model2x.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model2y = Model(inputs=y.input, outputs=y.output)
model2y.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

model2x.fit(x=X1_train.cpu().numpy(), y=y_train, validation_data=(X1_test.cpu().numpy(), y_test), batch_size=64, epochs=25, shuffle=True)
model2y.fit(x=X2_train.cpu().numpy(), y=y_train, validation_data=(X2_test.cpu().numpy(), y_test), batch_size=64, epochs=25, shuffle=True)

model.fit(x=X2_train_app.cpu().numpy(), y=y_train_app, validation_data=(X2_test_app.cpu().numpy(), y_test_app), batch_size=64, epochs=25, shuffle=True)

model.summary()



"""### GRID SEARCH"""

emb_dict = {1:'X2_train_app', 2:'X3_train'}
emb_lengths = [len(i[0][0]) for i in [X2_train_app, X3_train]][::-1]
X_trains = [X2_train_app, X3_train][::-1]
X_tests = [X2_test_app, X3_test][::-1]
Y_trains = [y_train_app, y3_train][::-1]
Y_tests = [y_test_app, y3_test][::-1]

sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
RMSprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
Adagrad = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
Adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
optimizers = [sgd, RMSprop, Adagrad, Adadelta]

activations = ['softmax']
regulizer = [True, False]
regulizer_dict = {True:'Regulizer', False:'Multiplicative'}

def keras_self_att_multi(X, Y, X_t, Y_t, emb_len, optimizer, activation, regulizer):
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
    model.add(Dense(2, activation=activation, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    #seqModel = model.fit(x=X.cpu().numpy(), y=Y, validation_data=(X_t.cpu().numpy(), Y_t), batch_size=64, epochs=1, shuffle=True, callbacks=[TQDMNotebookCallback(leave_inner=True, leave_outter=True)], verbose=0)    
    history = model.fit(x=X.cpu().numpy(), y=Y, validation_data=(X_t.cpu().numpy(), Y_t), batch_size=64, epochs=100, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)])    
    temp = max(history.history['val_acc'])
    return temp

val_acc_list = [[], []]

for s_no, X,Y,X_t,Y_t,emb_len in zip([2, 1], X_trains, Y_trains, X_tests, Y_tests, emb_lengths):
    for optimizer in optimizers:
        for activation in activations:
            for reg_flag in regulizer:
                para = emb_dict[s_no], optimizer, activation, regulizer_dict[reg_flag]
                print(para)
                max_val = keras_self_att_multi(X, Y, X_t, Y_t, emb_len, optimizer, activation, reg_flag)
                val_acc_list[0].append(para)
                val_acc_list[1].append(max_val)
                print(max_val)
                print('\n')

'''
# ----------------------------RESULTS FOR BEST MODEL WAS-------------------------------#
('X2_train_app', <keras.optimizers.SGD object at 0x7f34ed084be0>, 'softmax', 'Regulizer') : 0.7543859649122807  76 0.7729<br>

('X2_train_app', <keras.optimizers.SGD object at 0x7f34ed084be0>, 'softmax', 'Multiplicative') : 0.7646198816466749  77 0.7644<br>

('X2_train_app', <keras.optimizers.Adadelta object at 0x7f34ed084f28>, 'softmax', 'Multiplicative'): 0.7558479542620697
30 0.7871

'''