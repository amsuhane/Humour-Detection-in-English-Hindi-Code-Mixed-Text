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
<br> very consistent no variation
"""