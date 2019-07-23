import numpy as np
import pandas as pd
import os
import datetime
import pkg_resources
import time
import scipy.stats as stats
import gc
import re
import operator
import sys
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.utils import data
import pickle
from apex import amp
import shutil
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import (
    BertTokenizer,
    BertForSequenceClassification,
    BertAdam,
)


device = torch.device("cuda")
MAX_SEQUENCE_LENGTH = 220
SEED = 1234
EPOCHS = 5
Data_dir = "../data/data.pkl"
Input_dir = "../data/"
WORK_DIR = "../BERT/"
BERT_MODEL_PATH = 'bert-base-uncased'

def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm_notebook(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)

def prepare_input():
	tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)
	with open(Data_dir, 'rb') as f:
	    corpus = pickle.load(f)
	y_corpus = [i[-1] for i in corpus]
	train_df = pd.DataFrame([[(' ').join([j[0] for j in i[:-1]]), l] for i, l in zip(corpus, y_corpus)], columns = ['comment_text', 'target'])
	print('loaded %d records' % len(train_df))

	# Make sure all comment_text values are strings
	train_df['comment_text'] = train_df['comment_text'].astype(str) 
	sequences = convert_lines(train_df["comment_text"].fillna("DUMMY_VALUE"),MAX_SEQUENCE_LENGTH,tokenizer)
	train_df=train_df.fillna(0)

	# List all identities
	y_columns=['target']

	train_df = train_df.drop(['comment_text'],axis=1)
	# convert target to 0,1
	train_df['target']=(train_df['target']>=0.5).astype(float)

	num_to_train = len(sequences)-1  #no validation
	valid_size = len(sequences)-num_to_train
	X = sequences[:num_to_train]                
	y = train_df[y_columns].values[:num_to_train]
	X_val = sequences[num_to_train:]                
	y_val = train_df[y_columns].values[num_to_train:]
	test_df=train_df.tail(valid_size).copy()
	train_df=train_df.head(num_to_train)

	return X, y, X_val, y_val, test_df, train_df


class LenMatchBatchSampler(data.BatchSampler):
    def __iter__(self):

        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            count_zeros = torch.sum(self.sampler.data_source[idx][0] == 0)
            count_zeros = int(count_zeros / 64) 
            if len(buckets[count_zeros]) == 0:  buckets[count_zeros] = []

            buckets[count_zeros].append(idx)

            if len(buckets[count_zeros]) == self.batch_size:
                batch = list(buckets[count_zeros])
                yield batch
                yielded += 1
                buckets[count_zeros] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert len(self) == yielded, "produced an incorrect number of batches. expected %i, but yielded %i" %(len(self), yielded)

def trim_tensors(tsrs):
    max_len = torch.max(torch.sum( (tsrs[0] != 0  ), 1))
    if max_len > 2: 
        tsrs = [tsr[:, :max_len] for tsr in tsrs]
    return tsrs 


X, y, X_val, y_val, test_df, train_df  - prepare_input()
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.long),
                                               torch.tensor(y,dtype=torch.float))


output_model_file = "bert_pytorch.bin"

lr=2e-5
batch_size = 32
accumulation_steps=2
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def configure_model():
	model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH,cache_dir=None,num_labels=1)
	model.zero_grad()
	model = model.to(device)

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	lr = 3e-5
	epsilon=1
	lr_d = {}
	weight_d = {}
	for n, p in param_optimizer:
	    if any(nd in n for nd in no_decay):
	        weight_d[n] = 0.0
	    else:
	        weight_d[n] = 0.01
	for n, p in param_optimizer[:5]:
	    lr_d[n] = lr*(epsilon**(11))
	for n, p in param_optimizer:
	    if 'bert.encoder.layer.' in n:
	        for i in range(0, 12):
	            if 'bert.encoder.layer.'+str(i)+'.'  in n:
	                lr_d[n] = lr*(epsilon**(11-i))
	                break
	for n, p in param_optimizer[-4:]:
	    lr_d[n] = lr
	comb_dict = {}
	for n, p in param_optimizer:
	    para = (weight_d[n], lr_d[n])
	    if para in comb_dict:
	        comb_dict[para].append(p)
	    else:
	        comb_dict[para] = [p]
	optimizer_grouped_parameters = []
	for i, j in comb_dict.items():
	    optimizer_grouped_parameters.append({'params':j, 'weight_decay' : i[0], 'lr' : i[1]})

	train = train_dataset

	num_train_optimization_steps = int(EPOCHS*len(train)/batch_size/accumulation_steps)

	optimizer = BertAdam(optimizer_grouped_parameters,
	                     lr=lr,
	                     warmup=0.05,
	                     t_total=num_train_optimization_steps)

	model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
	model=model.train()

	return model, optimizer, train

def run_model()
	tq = tqdm_notebook(range(EPOCHS))
	for epoch in tq:
	    ran_sampler = data.RandomSampler(train)
	    len_sampler = LenMatchBatchSampler(ran_sampler, batch_size = 32, drop_last = False)
	    #train_loader = torch.utils.data.DataLoader(train, batch_sampler = len_sampler)
	    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
	    avg_loss = 0.
	    avg_accuracy = 0.
	    lossf=None
	    #tk0 = tqdm_notebook(enumerate(train_loader),total=len(train_loader),leave=False)
	    tk0 = tqdm_notebook(enumerate(train_loader),total=len(train_loader),leave=False)
	    optimizer.zero_grad()
	    for i, batch in tk0:
	        tsrs = trim_tensors(batch)
	        (x_batch, y_batch) = tuple(t.to(device) for t in tsrs)
	        y_pred = model(x_batch, attention_mask=(x_batch>0), labels=None)
	        loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch)
	        #loss = nn.BCEWithLogitsLoss(weight=weight_batch)(y_pred,y_batch)
	        with amp.scale_loss(loss, optimizer) as scaled_loss:
	            scaled_loss.backward()
	        if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
	            optimizer.step()                            # Now we can do an optimizer step
	            optimizer.zero_grad()
	        if lossf:
	            lossf = 0.98*lossf+0.02*loss.item()
	        else:
	            lossf = loss.item()
	        tk0.set_postfix(loss = lossf)
	        avg_loss += loss.item() / len(train_loader)
	        avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5)).to(torch.float) ).item()/len(train_loader)
	    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)

	torch.save(model.state_dict(), output_model_file)

model, optimizer, train = configure_model()
run_model()