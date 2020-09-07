#!/usr/bin/env python
# coding: utf-8
# reference: https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import time

import en_core_web_sm
import nl_core_news_sm

SEED = 99
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = nl_core_news_sm.load()
spacy_en = en_core_web_sm.load()


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize = tokenize_de, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

TRG = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
                                                    fields = (SRC, TRG))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)


BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)


class Encoder(nn.Module):
    
    def __init__(self, input_dim, emb_dim, enc_hidn_dim, dec_hidn_dim, dropout):
        """
        Parameters
        ----------
        input_dim: equal to number of (voc + special token)
        emb_dim: embedded dimension
        enc_hidn_dim: encoder hidden dimension, i.e. dimension of encoder annotations, "h" in the paper. 
        dec_hidn_dim: decoder hidden dimension, i.e. dimension of decoder hidden state, "s" in the paper.
        dropout: boolean
        
        Layers
        ----------
        self.emb: embedding layer
        self.enc_recurrent: return decoder state
        self.fc: return hidden state, s_0, i.e. initial hidden state for decoder
        - trick: using final hidden state of bidirection LSTM as initial hidden state for decoder
        """
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.enc_recurrent = nn.LSTM(emb_dim, enc_hidn_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hidn_dim*2, dec_hidn_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        Variables
        -----------
        src
        - source. encoded by vocab index.
        - shape: [bs, src_len]
        
        embedded
        - embedded source sequence
        - shape: [src_len, bs, emb_dim]
        
        annotations
        - shape: [src_len, bs, hidn_dim*num_directions]
        
        final_hidn
        - shape: [stacked_layer*num_directions, bs, hidn_dim]
        
        decoder_init_state
        - shape: [bs, dec_hidn_dim]
        """
        embedded = self.dropout(self.embedding(src))
        annotations, (final_hidn, _) = self.enc_recurrent(embedded)
        final_concat = torch.cat((final_hidn[-2,:,:], final_hidn[-1,:,:]), dim=1) 
        decoder_init_state = torch.tanh(self.fc(final_concat))
        
        return annotations, decoder_init_state


class Attention(nn.Module):
    def __init__(self, enc_hidn_dim, dec_hidn_dim):
        """
        Layers
        ----------
        self.attn: calc attenion using 
                oncat(cocatenated encoder annotations sequence, t-1 step decoder hidden state)
        self.linear_cmb: get weight(associated energy) for each encoder sequence annotation  
        """
        super().__init__()
        self.attn = nn.Linear(enc_hidn_dim * 2 + dec_hidn_dim, dec_hidn_dim)
        self.linear_comb = nn.Linear(dec_hidn_dim, 1, bias=False)
    
    def forward(self, decoder_state, encoder_annotation_seq):
        """
        Variables
        -----------
        encoder_annotation_seq
        - sequence of encoder hidden state. concat(h_LtoR, h_RtoL).
        - shape: [bs, src_len, enc_hidn dim]
        
        decoder_state
        - decoder hidden state, i.e. s.
        - shape: [bs, src_len, dec_hidn_dim]
        
        enc_annot_dec_state_concat
        - concat(encoder annotations sequence, decoder_state)
        - shape: [enc_hidn_dim*2+dec_hidn_dim, dec_hiddne_dim]
        
        associated_energy
        - correlation b/w encoder annotations and decoder hidden state
        - shape: [bs, dec_hidn_dim]
        
        attention
        - linear combination of 
        - shape: [bs, src_len]
        """        
        bs = encoder_annotation_seq.shape[1]
        src_len = encoder_annotation_seq.shape[0]
        
        encoder_annotation_seq = encoder_annotation_seq.permute(1, 0, 2)
        decoder_state = decoder_state.unsqueeze(1).repeat(1, src_len, 1)
        
        enc_annot_dec_state_concat = torch.cat((encoder_annotation_seq, decoder_state), dim=1)
        associated_energy = torch.tanh(self.attn(enc_annot_dec_state_concat))
        
        attention = self.linear_comb(associated_energy).squeeze(2)
        attention = F.softmax(attention, dim=1)
        
        return attention


class Decoder(nn.Module):
    
    def __init__(self, dec_output_dim, emb_dim, enc_hidn_dim, dec_hidn_dim, dropout, attention):    
        """
        - Why should hidn_state_module be recurrent net, even thuoght it has 1 sequence? 
        - To reflect previous state
        - There are two kinds of input here:
            1. (due to this one)previous decoder hidden state and  
            2. normal input(previous y and context vector)
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dec_output_dim = dec_output_dim
        self.attention_module = attention
        self.embedding = nn.Embedding(dec_output_dim, emb_dim)
        self.hidn_state_module = nn.LSTM((enc_hidn_dim * 2) + emb_dim, dec_hidn_dim, bidirectional=False)
        self.final_out = nn.Linear((enc_hidn_dim * 2) + dec_hidn_dim, dec_output_dim)
    
    def forward(self, y_previous, hidn_state_previous, encoder_annotation_seq):
        """
        Variables
        -----------
        y_previous
        - y_(t-1)
        - shape: [1, bs] -> [1, bs, emb_dim]
        
        attention_weight
        - attentuini weight for H(annotation sequences)
        - shape: [bs, src_len] -> [bs, 1, src_len]
        
        encoder_annotation_seq
        - concatenated annotation sequences
        - shape: [bs, src_len, enc_hidn_dim*2]
        
        context_vector
        - encoder_annotation_seq to which applied attention
        - shape: [bs, 1, enc_hidn_dim*2] -> [1, bs, enc_hidn_dim*2]
        
        hidn_state
        - decoder hidden state i.e. s_i = f(s_(i-1), y_(i-1), c_i)
        
        y_current
        - generated y_t
        """
        y_previous = self.dropout(self.embedding(y_previous.unsqueeze(0)))
        attention_weight = self.attention_module(hidn_state_previous, encoder_annotation_seq).unsqueeze(1)
        encoder_annotation_seq = encoder_annotation_seq.permute(1, 0, 2)
        context_vector = torch.bmm(attention_weight, encoder_annotation_seq)
        hidn_state, (last_hidn_state, _) = self.hidn_state_module(torch.cat((y_previous,                                                                             context_vector), dim = 2),                                                                  hidn_state_previous.unsqueeze(0))
        
        assert (hidn_state==last_hidn_state).all()
        y_current = self.final_out(torch.cat(y_previous.squeeze(0),                                             hidn_state.squeeze(0),                                             context_vector.squeeze(0)))
        
        return y_current, last_hidn_state.squeeze(0)

    
class Seq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.init_weights()
        
    def init_weights(self):
        
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if 'bias' in name:
                nn.init.constant_(param.data, 0)
                
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        process
        - encoder return concatenated annotations sequence & decoder initial hidden state
        - decoder return 
            - c_i = z(s_(i-1), annotations); 
            - s_i = f(y_(i-1), s_(i-1), c_i); 
            - y_i = g(y(i-1), s_i, c_i)
        """
        bs = src.shape[1]
        trg_len = trg.shape[0]
        trg_voc_size = self.decoder.dec_output_dim
        
        y_generated = torch.zeros(trg_len, bs, trg_voc_size).to(self.device)
        annotations, decoder_state_previous = self.encoder(src)
        
        y_previous = trg[0, :] # <SOS> token
        
        for t in range(1, trg_len):
            y_current, last_hidn_state = self.decoder(y_previous, decoder_state_previous, annotations)
            y_generated[t] = y_current
            
            teacher_force = torch.rand(size=(1,)) < teacher_forcing_ratio
            idx = y_current.argmax(1)
            # trg[t]: true token; idx: predicted token;
            y_previous = trg[t] if teacher_force else idx
        
        return outputs



INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

sample_batch = next(iter(train_iterator))
sample_src = sample_batch.src
sample_trg = sample_batch.trg

output = model(sample_src, sample_trg)