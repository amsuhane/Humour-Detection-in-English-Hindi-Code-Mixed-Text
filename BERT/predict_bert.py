import sys
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from pytorch_pretrained_bert import BertConfig
import torch
import torch.utils.data
import os

device = torch.device('cuda')

def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)


MAX_SEQUENCE_LENGTH = 220
SEED = 42 
BATCH_SIZE = 32
INFER_BATCH_SIZE = 64
BERT_MODEL_PATH = 'bert-base-uncased'
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)
with open(Data_dir, 'rb') as f:
    corpus = pickle.load(f)
y_corpus = [i[-1] for i in corpus]
test_df = pd.DataFrame([(' ').join([j[0] for j in i[:-1]]) for i  in corpus], columns = ['comment_text'])
print('loaded %d records' % len(test_df))

test_df['comment_text'] = test_df['comment_text'].astype(str)
X_test = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), MAX_SEQUENCE_LENGTH, tokenizer)
test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))

model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH,cache_dir=None,num_labels=1)
model.load_state_dict(torch.load('bert_pytorch.bin'))
model.to(device)

for param in model.parameters():
    param.requires_grad = False
model.eval()

test_preds = np.zeros((len(X_test)))
test_loader = torch.utils.data.DataLoader(test, batch_size=INFER_BATCH_SIZE, shuffle=False)
tk0 = tqdm(test_loader)
for i, (x_batch,) in enumerate(tk0):
    pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
    test_preds[i * INFER_BATCH_SIZE:(i + 1) * INFER_BATCH_SIZE] = pred[:, 0].detach().cpu().squeeze().numpy()

predictions_bert = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()

np.save('bert_predictions.npy', predictions_bert) 
