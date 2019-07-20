#### kFold
"""

from sklearn.model_selection import StratifiedKFold

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

    #seqModel = model.fit(x=X.cpu().numpy(), y=Y, validation_data=(X_t.cpu().numpy(), Y_t), batch_size=64, epochs=1, shuffle=True, callbacks=[TQDMNotebookCallback(leave_inner=True, leave_outter=True)], verbose=0)    
    #history = model.fit(x=X.cpu().numpy(), y=Y, validation_data=(X_t.cpu().numpy(), Y_t), batch_size=64, epochs=100, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)])    
    #temp = max(history.history['val_acc'])
    #return temp
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

emb_elmo = np.load('gdrive/My Drive/Humour_Detection_Hinglish_BTP/embeddings/elmo_final.npy', allow_pickle=True)
hi_en = {'Hi':1, 'En':-1}
emb_elmo_app = []
for i, j in zip(emb_elmo, corpus_new):
    temp = [hi_en[k[1]] for k in j]
    a = np.array(temp).reshape((len(temp), 1))
    b = np.array(i)
    emb_elmo_app.append(np.hstack((a, b)))

y_temp  = np.argmax(y, axis=1)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) 
skf.get_n_splits(emb_elmo_app, y_temp)

result10 = [[],[],[]]
predictions10 = [[],[],[]]

for train_index, test_index in skf.split(emb_elmo_app, y_temp):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test, y_train1, y_test1 = [], [], [], []
    for i in train_index:
        X_train.append(emb_elmo_app[i])
        y_train1.append(y_temp[i])
    for i in test_index:
        X_test.append(emb_elmo_app[i])
        y_test1.append(y_temp[i])
    
    y_train1 = np.array(y_train1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(y_train1.reshape(-1, 1))
    y_train = enc.transform(y_train1.reshape(-1, 1)).toarray()
    y_test1 = np.array(y_test1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(y_test1.reshape(-1, 1))
    y_test = enc.transform(y_test1.reshape(-1, 1)).toarray()
     
    emb_list = []
    for i in X_train:
        emb_list.append(torch.tensor(i[:20]).cuda())
    X_train = pad_sequence(emb_list, batch_first=True)

    emb_list = []
    for i in X_test:
        emb_list.append(torch.tensor(i[:20]).cuda())
    X_test = pad_sequence(emb_list, batch_first=True)
    
    #for i in [1, 2, 3]:
    i=1
    model = return_model(i)
    model.fit(x=X_train.cpu().numpy(), y=y_train, validation_data=(X_test.cpu().numpy(), y_test), batch_size=64, epochs=100, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)])    
    result10[i-1].append(model.evaluate(x=X_test.cpu().numpy(), y=y_test, verbose=1))
    predictions10[i-1].append(model.predict(x=X_test.cpu().numpy()))

predictions

[[[0.8888459744244597, 0.7167883215159395],
  [0.8962745171541359, 0.7046783625730995],
  [0.8526838450620324, 0.7496339677018966],
  [0.9032193797573688, 0.6969253296471619],
  [0.8642711080010779, 0.7393850662348734]],
 [[0.8688302718809922, 0.7153284674143269],
  [0.8917690536432099, 0.7134502923976608],
  [0.8691211758211451, 0.7379209374788042],
  [0.9044464177303398, 0.7057101027508257],
  [0.8620628594130408, 0.7306002930875753]],
 []]

results

from scipy import stats

======MODEL 1=======
[0.7167883215159395, 0.7046783625730995, 0.7496339677018966, 0.6969253296471619, 0.7393850662348734]
DescribeResult(nobs=5, minmax=(0.6969253296471619, 0.7496339677018966), mean=0.7214822095345942, variance=0.0005051189932959484, skewness=0.2066154208425755, kurtosis=-1.5612433311940999)
0.7214822095345942 ± 0.02815175816730242
======MODEL 2=======
[0.7153284674143269, 0.7134502923976608, 0.7379209374788042, 0.7057101027508257, 0.7306002930875753]
DescribeResult(nobs=5, minmax=(0.7057101027508257, 0.7379209374788042), mean=0.7206020186258385, variance=0.00017515928282488204, skewness=0.28686256034955615, kurtosis=-1.4463898238565704)
0.7206020186258385 ± 0.01731891885296566

print('======MODEL 1=======')
results_model1 = [i[1] for i in results[0]]
print(results_model1)
print(stats.describe(results_model1))
print(np.mean(results_model1), '±', max([i-np.mean(results_model1) for i in results_model1]))
print('======MODEL 2=======')
results_model2 = [i[1] for i in results[1]]
print(results_model2)
print(stats.describe(results_model2))
print(np.mean(results_model2), '±', max([i-np.mean(results_model2) for i in results_model2]))
print('======MODEL 3=======')
results_model3 = [i[1] for i in results[2]]
print(results_model3)
print(stats.describe(results_model3))
print(np.mean(results_model3), '±', max([i-np.mean(results_model3) for i in results_model3]))

print(np.mean(results_model3), '±', max([i-np.mean(results_model3) for i in results_model3]))
print(np.mean(results_model2), '±', max([i-np.mean(results_model2) for i in results_model2]))
print(np.mean(results_model1), '±', max([i-np.mean(results_model1) for i in results_model1]))

"""[0.7167883215159395, 0.7046783625730995, 0.7496339677018966, 0.6969253296471619, 0.7393850662348734]
DescribeResult(nobs=5, minmax=(0.6969253296471619, 0.7496339677018966), mean=0.7214822095345942, variance=0.0005051189932959484, skewness=0.2066154208425755, kurtosis=-1.5612433311940999)
0.7214822095345942 ± 0.02815175816730242

### Selected Models
"""

sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
Adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

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

    #seqModel = model.fit(x=X.cpu().numpy(), y=Y, validation_data=(X_t.cpu().numpy(), Y_t), batch_size=64, epochs=1, shuffle=True, callbacks=[TQDMNotebookCallback(leave_inner=True, leave_outter=True)], verbose=0)    
    #history = model.fit(x=X.cpu().numpy(), y=Y, validation_data=(X_t.cpu().numpy(), Y_t), batch_size=64, epochs=100, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)])    
    #temp = max(history.history['val_acc'])
    #return temp
    return model

def return_model(key):
    if key==1:
        return good_models(1025, sgd, True)
    if key==2:
        return good_models(1025, sgd, False)
    else:
        return good_models(1025, Adadelta, False)

model1 = return_model(1)
model2 = return_model(2)
model3 = return_model(3)
#model1.fit(x=X2_train_app.cpu().numpy(), y=y_train, validation_data=(X2_test_app.cpu().numpy(), y_test), batch_size=64, epochs=100, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)])    
#model2.fit(x=X2_train_app.cpu().numpy(), y=y_train, validation_data=(X2_test_app.cpu().numpy(), y_test), batch_size=64, epochs=100, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)])    
#model3.fit(x=X2_train_app.cpu().numpy(), y=y_train, validation_data=(X2_test_app.cpu().numpy(), y_test), batch_size=64, epochs=100, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)])    

model1.fit(x=X2_app.cpu().numpy(), y=y, validation_split=0.2, batch_size=64, epochs=100, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)])    
model2.fit(x=X2_app.cpu().numpy(), y=y, validation_split=0.2, batch_size=64, epochs=100, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)])    
model3.fit(x=X2_app.cpu().numpy(), y=y, validation_split=0.2, batch_size=64, epochs=100, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)])    

pred1 = model1.predict(x=X2_app.cpu().numpy())
pred2 = model2.predict(x=X2_app.cpu().numpy())
pred3 = model3.predict(x=X2_app.cpu().numpy())

predictions = [pred1, pred2, pred3]

np.save('predictions.npy', predictions)

np.save('corpus_orignal.npy', corpus)

"""### ENSEMBLE MODELS

**MODEL 2**
"""

TIME_STEPS, INPUT_DIM_1, INPUT_DIM_2 = 20, len(X1_train[0][0]), len(X2_train[0][0])
lstm_units_1 = 20
lstm_units_2 = 10

inputs_layer_1 = Input(shape=(TIME_STEPS, INPUT_DIM_1, ))
inputs_layer_2 = Input(shape=(TIME_STEPS, INPUT_DIM_2, ))

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

"""Epoch 25/25 <br>
2734/2734 [==============================] - 8s 3ms/step - loss: 0.4056 - acc: 0.8224 - val_loss: 0.6188 - val_acc: 0.7098
                
Epoch 25/25 <br>
2734/2734 [==============================] - 7s 3ms/step - loss: 0.4090 - acc: 0.8109 - val_loss: 0.5526 - val_acc: 0.7259
"""

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

input_comb_1 = Input(shape=(2, ))
input_comb_2 = Input(shape=(2, ))
combined = concatenate([input_comb_1, input_comb_2])
z = Dense(4, activation="sigmoid")(combined)
z = Dense(2, activation="sigmoid")(z)
model2_final = Model(inputs=[input_comb_1, input_comb_2], outputs=z)

model2_final.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model2_final.fit(x=[predx, predy], y=y_train, validation_data=([predx_t, predy_t], y_test),	batch_size=64, epochs=30, shuffle=True, verbose=1)

"""Epoch 30/30 <br>
2734/2734 [==============================] - 0s 54us/step - loss: 0.3206 - acc: 0.8976 - val_loss: 0.5410 - val_acc: 0.7434
"""

from keras_multi_head import MultiHead

model = k.models.Sequential()
model.add(k.layers.Embedding(input_dim=100, output_dim=20, name='Embedding'))
model.add(MultiHead(k.layers.LSTM(units=32), layer_num=5, name='Multi-LSTMs'))
model.add(k.layers.Flatten(name='Flatten'))
model.add(k.layers.Dense(units=4, activation='softmax', name='Dense'))
model.build()
model.summary()

TIME_STEPS, INPUT_DIM_1, INPUT_DIM_2 = 20, len(X1_train[0][0]), len(X2_train[0][0])
lstm_units_1 = 20
lstm_units_2 = 10

inputs_layer_1 = Input(shape=(TIME_STEPS, INPUT_DIM_1, ))
inputs_layer_2 = Input(shape=(TIME_STEPS, INPUT_DIM_2, ))

x = Bidirectional(LSTM(lstm_units_1, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(inputs_layer_1)
x = Bidirectional(LSTM(lstm_units_2, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))(x)
x = Dense(10, activation="sigmoid")(x)
x = Model(inputs=inputs_layer_1, outputs=x)

y = Bidirectional(LSTM(lstm_units_1, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(inputs_layer_2)
y = Bidirectional(LSTM(lstm_units_2, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))(y)
y = Dense(10, activation="sigmoid")(y)
y = Model(inputs=inputs_layer_2, outputs=y)

combined = concatenate([x.output, y.output])

z = Dense(2, activation="tanh")(combined)

model = Model(inputs=[x.input, y.input], outputs=z)

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
SVG(model_to_dot(model).create(prog='dot', format='svg'))

model.fit(x=[X1_train.cpu().numpy(), X2_train.cpu().numpy()], y=y_train, validation_data=([X1_test.cpu().numpy(), X2_test.cpu().numpy()], y_test),	batch_size=64, epochs=100, shuffle=True)

"""Epoch 80/100 <br>
2734/2734 [==============================] - 13s 5ms/step - loss: 4.2182 - acc: 0.4943 - val_loss: 4.9409 - val_acc: 0.3969

#### USING BEST MODELS
"""

model1 = return_model(1)
model2 = return_model(2)
model3 = return_model(3)

del model1

model1 = load('gdrive/My Drive/Humour_Detection_Hinglish_BTP/model1.h5')

inputs_layer_1 = Input(shape=(2, ))
inputs_layer_2 = Input(shape=(2, ))
inputs_layer_3 = Input(shape=(2, ))

combined = concatenate([inputs_layer_1, inputs_layer_2, inputs_layer_3])
combined = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(combined)
model = Model(inputs=[inputs_layer_1, inputs_layer_2, inputs_layer_3], outputs=combined)
sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


model.fit(x=[i for i in predictions], y=y_test, validation_split=0.2, batch_size=64, epochs=50, shuffle=True)

"""Epoch 50/50 <br>
547/547 [==============================] - 0s 92us/step - loss: 1.1616 - acc: 0.7422 - val_loss: 1.1401 - val_acc: 0.7372
<br> very consistent no variation
"""

TIME_STEPS, INPUT_DIM = len(X2_train_app[0]), len(X2_train_app[0][0])


inputs_layer = Input(shape=(TIME_STEPS, INPUT_DIM, ))

x = Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(inputs_layer)
y = Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(inputs_layer)
z = Bidirectional(LSTM(10, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(inputs_layer)

x = SeqSelfAttention(
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_activation='sigmoid',
        use_attention_bias=True,
        kernel_regularizer=keras.regularizers.l2(1e-6),
        bias_regularizer=keras.regularizers.l1(1e-6),
        attention_regularizer_weight=1e-6,
        name='Attentionx')(x)

y = SeqSelfAttention(
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_activation=None,
        kernel_regularizer=keras.regularizers.l2(1e-6),
        use_attention_bias=False,
        name='Attentiony',
    )(y)

z = SeqSelfAttention(
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_activation=None,
        kernel_regularizer=keras.regularizers.l2(1e-6),
        use_attention_bias=False,
        name='Attentionz',
    )(z)

x = Bidirectional(LSTM(5, dropout=0.5, recurrent_dropout=0.5, return_sequences=False), merge_mode='concat')(x)
y = Bidirectional(LSTM(5, dropout=0.5, recurrent_dropout=0.5, return_sequences=False), merge_mode='concat')(y)
z = Bidirectional(LSTM(5, dropout=0.5, recurrent_dropout=0.5, return_sequences=False), merge_mode='concat')(z)

x = Model(inputs=inputs_layer, outputs=x)
y = Model(inputs=inputs_layer, outputs=y)
z = Model(inputs=inputs_layer, outputs=z)

combined = concatenate([x.output, y.output, z.output])
combined = Dense(8, activation='softmax', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(combined)
combined = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(combined)

model = Model(inputs=inputs_layer, outputs=combined)

sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

#x = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x)
#y = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(y)
#z = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(z)

#model.add(Bidirectional(LSTM(5, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat'))
#model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
#model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.fit(x=[X2_train_app.cpu().numpy(), X2_train_app.cpu().numpy(), X2_train_app.cpu().numpy()], y=y_train, validation_data=([X2_test_app.cpu().numpy(), X2_test_app.cpu().numpy(), X2_test_app.cpu().numpy()], y_test),	batch_size=64, epochs=100, shuffle=True)

"""### Flair"""

from tqdm import tqdm

corpus_new = [(' ').join([i[0] for i in j]) for j in tqdm(corpus_new)]
 

labels = [i[-1] for i in corpus]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(corpus_new, labels, random_state=42, stratify=labels)

with open('train.txt', 'w') as f:
    for i, j in zip(y_train, X_train):
        f.write('__label__'+str(i)+ ' '+ j+'\n')
with open('test.txt', 'w') as f:
    for i, j in zip(y_test, X_test):
        f.write('__label__'+str(i)+ ' '+ j+'\n')

!pip install flair
 



from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, BertEmbeddings, DocumentPoolEmbeddings, DocumentLSTMEmbeddings, CharacterEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from typing import List

# load corpus containing training, test and dev data
corpus: Corpus = ClassificationCorpus(data_folder='./', test_file='test.txt',
                                      train_file='train.txt')

label_dict = corpus.make_label_dictionary()


# 3. make a list of word embeddings

# initialize the word embeddings

#bert_embedding = BertEmbeddings('bert-base-multilingual-cased')
#glove_embedding = WordEmbeddings('hi')
#flair_forward = FlairEmbeddings('news-forward')
#flair_backward = FlairEmbeddings('news-backward')
char_emb = CharacterEmbeddings()

document_embeddings = DocumentLSTMEmbeddings([char_emb])



'''
word_embeddings = [WordEmbeddings('glove'),

                   # comment in flair embeddings for state-of-the-art results
                    FlairEmbeddings('news-forward'),
                    FlairEmbeddings('news-backward'),
                   ]

from typing import List
from flair.embeddings import TokenEmbeddings
embedding_types: List[TokenEmbeddings] = [

    WordEmbeddings('glove'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
'''
# 5. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

# 6. initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)

# 7. start the training
trainer.train('./',
              #learning_rate=0.1,
              mini_batch_size=32,
#              anneal_factor=0.5,
              patience=5,
              max_epochs=10)

from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('./loss.tsv')
plotter.plot_weights('./weights.txt')

from flair.data import Sentence

from tqdm import tqdm

from sklearn.metrics import accuracy_score

y_h = []
for i in tqdm(X_test):
    sentence = Sentence(i)
    classifier.predict(sentence)
    y_h.append(sentence.labels)



y_h_1 = [int(str(i[0]).split(' ')[0]) for i in y_h]

accuracy_score(y_test, y_h_1)

"""### NEW MODEL"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Bidirectional, concatenate, Embedding, Input, CuDNNLSTM, Dropout, SpatialDropout1D,GlobalMaxPooling1D,GlobalAveragePooling1D, add
from keras.models import Sequential,Model
from keras import regularizers, optimizers
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l1_l2
from keras import backend as K
from keras.layers.recurrent import LSTM
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True
        #super(Attention, self).build(input_shape) 

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
    

def build_model_1(embedding_matrix):
    words = Input(shape=(MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    
    y = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    y = add([y, Dense(128*4, activation='relu')(y)])
    y = add([y, Dense(128*4, activation='relu')(y)])

    z = Attention(MAX_LEN)(x)
    z = add([z, Dense(128*2, activation='relu')(y)])
    z = add([z, Dense(128*2, activation='relu')(y)])    
    
    result = concatenate([y, z])
    result = Dense(1, activation='sigmoid')(y)
    
    model = Model(inputs=words, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model



def build_model(embedding_matrix):
    words = Input(shape=(MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    
    model = Model(inputs=words, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

TIME_STEPS, INPUT_DIM, LSTM_UNITS, DENSE_HIDDEN_UNITS = 20, 1025, 10, 10*4

def build_model():
    
    inputs_layer = Input(shape=(TIME_STEPS, INPUT_DIM, ))
    
    x = SpatialDropout1D(0.2)(inputs_layer)
    x = Bidirectional(LSTM(LSTM_UNITS, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(x)
    x = Bidirectional(LSTM(LSTM_UNITS, dropout=0.5, recurrent_dropout=0.5, return_sequences=True), merge_mode='concat')(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(hidden)])
    #hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    
    model = Model(inputs=inputs_layer, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = build_model()

model.fit(X2_train_app.cpu().numpy(), np.argmax(y_train_app, axis=1), validation_data=[X2_test_app.cpu().numpy(), np.argmax(y_test_app, axis=1)], epochs=30, verbose=1, shuffle=True)

