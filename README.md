
**Instructions to run**

Run ``embeddings/generate_embeddings.py`` to create embeddings and store in ``embeddings/``. ``models/all_model_trials.py`` contains all the tried architecture and models variations. ``models/best_models.py`` contains the best models, whose hyperparameters have been tuned using grid search.
The best models have been saved to be further used in ensemble model.

Along with these **BERT transformer** was also used. Apex, from nvidia, and sequence-bucketing was used to speed up the training. Also layer-wise decreasing learning rate was used to tune the BERT transformer on the corpus. Run  
``run_bert_classifier.py`` to train a BERT transformer. The model is saved as bert_model.bin.

**Future work**
- Use XLnet
- Ensemble modeling

**Results**

A stratified train-test split of 0.8 was used for all the models:
Results are written in the form: (test accuracy, train, accuracy)

| Model        | Train Accuracy           | Test accuracy  |
| ------------- |:-------------:| -----:|
| BERT-cased-large      | 0.754 | 0.817 |
| BERT-cased-base      | 0.731      |   0.826 |
| BERT-uncased-base | 0.748      |    0.874 |
| Keras Model | 0.737 | 0.762
| Ensemble | WIP|WIP

**Project details**

Papers used for reference:
- Humor Detection in English-Hindi Code-Mixed Social Media Content : Corpus and Baseline System <sup>[link](https://arxiv.org/abs/1806.05513v1)</sup>
- Detecting Offensive Tweets in Hindi-English Code-Switched Language <sup>[link](https://www.aclweb.org/anthology/W18-3504)</sup>
- Learning Joint Multilingual Sentence Representations with Neural Machine Translation <sup>[link](https://www.aclweb.org/anthology/W17-2619)</sup>
- Humor Detection in English-Hindi Code-Mixed Social Media Content : Corpus and Baseline System <sup>[link](https://aclweb.org/anthology/L18-1193) </sup>
- How to Fine-Tune BERT for Text Classification? <sup>[link](https://arxiv.org/pdf/1905.05583.pdf)</sup>


Embeddings used:
- Elmo
- Bert
- Fasttext
- Stacked embeddings

Model architechures used are:
- BiLSTM
- Attention (Self written, keras-tqdm, keras-self-attention, keras-multi-head)
- Char-RNN
- BERT (pre-trained, with addiional hyper-tuning on dataset)

Language tags were also given for each word of the sentence. ex: INSERT EXAMPLE. To include this additional information in the word embeddings, 
an extra digit (1 for english, -1 for hindi) was appended to generated word embeddings. This gave an extra edge of nearly 0.6 points.

Method to find best model (the code is in ``All_model_trials``:
- First of all tried many combinations of model architecture and embeddings were tried
- The best embedding was clear: BERT embeddings appended language tag 
- Then the top models were selected, they consisted of
  BiLSTM layers with attention (Regulizer and Multiplicative) (from keras-self-attention)
- Then grid search was used for hyperparameter tuning

The final three models are shown in ``Selected_models.py``. The function to build the best models was:

```python

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
	return model

def return_model(key):
    if key==1:
        return good_models(len(X2_train_app[0][0]), sgd, True)
    if key==2:
        return good_models(len(X2_train_app[0][0]), sgd, False)
    else:
        return good_models(len(X2_train_app[0][0]), Adadelta, False)
```