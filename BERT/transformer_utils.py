import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    """
    note: decoding inputs have "[BOS]" only, because it predict 1, 0 sentimental label
    self.vocab: sentencepiece object
    self.labels: sentiment lable
        - eg. [1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    self.sentence: review sentence
        - eg. [ 'GDNTOPCLASSINTHECLUB', '뭐야 이 평점들은.... 나쁘진 않지만 10점 짜리는 더더욱 아니잖아',]
    """
    def __init__(self, vocab, data_fn):
        self.vocab = vocab
        self.labels = []
        self.sentence = []
        
        with open(data_fn, 'r') as f:
            for i, json_data in enumerate(f):
                if i == 50:
                    break
                json_data = json.loads(json_data)
                self.labels.append(json_data['label'])
                self.sentence.append([vocab.piece_to_id(tok) for tok in json_data['doc']])

    def __len__(self):
        assert len(self.labels) == len(self.sentence)
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.labels[idx]), 
               torch.tensor(self.sentence[idx]),
               torch.tensor([self.vocab.piece_to_id("[BOS]")])) # input only [BOS] into Decoder 
    
def custom_collate_fn(batch_data):
    """
    batch_data: n data(n=batch size) from Dataset 
    return
    - label
    - enc_inputs(token_ids): [bs, seq_len]
    - dec_inputs(BOS_token): [bs, seq_len]
    """
    label, token_ids, BOS_token = list(zip(*batch_data))
    
    token_ids = nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=0)
    BOS_token = nn.utils.rnn.pad_sequence(BOS_token, batch_first=True, padding_value=0)
    
    return [torch.stack(label, dim=0), token_ids, BOS_token] # [label, encoder inputs, decoder inputs]

def eval_model(config, model, data_loader):
    matchs = []
    model.eval()
    
    n_word_total = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            labels, enc_inputs, dec_inputs = map(lambda x: x.to(config.device), batch)
            
            outputs = model(enc_inputs, dec_inputs)
            logits = outputs[0] # logist = feedforard(dec_outputs): [bs, n_out]
            val, idx = logits.max(axis=1) # eg. binary clf: [3, 5] -> idx: [1]; idx shape: [bs]
            
            match = torch.eq(idx, labels).detach().cpu()
            matchs.extend(match.detach().cpu())
            
            running_accuracy = np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0
            print('running acc avg', running_accuracy)
            
    return running_accuracy

def train_model(config, n_epoch, model, criterion, optimizer, data_loader):
    model.train()
    running_loss = 0
    for i, batch in enumerate(data_loader):
        labels, enc_inputs, dec_inputs = map(lambda x: x.to(config.device), batch)
        optimizer.zero_grad()
        outputs = model(enc_inputs, dec_inputs)
        logits = outputs[0]
        
        loss = criterion(logits, labels)
        loss_val = loss.item()
        running_loss += loss_val
        
        loss.backward()
        optimizer.step()
        
        print('batch loss average: {}'.format(running_loss/(i+1)))
    
    return running_loss / len(data_loader)