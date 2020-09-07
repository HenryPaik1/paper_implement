import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_sinusoid_encoding_table(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table

def get_attn_pad_mask(seq_q, seq_k, i_pad):
    """
    key_vector의 pad열은 모두 0으로 padding
    - row: query token
    - col: key token
    
    params
    - seq_q, seq_k: [bs, len_seq]
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)  # key vector에 masking한 것을, len_q만큼 늘려줌(row wise)
    return pad_attn_mask

def get_attn_decoder_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of a matrix
    return subsequent_mask

class PoswiseFeedForwardNet(nn.Module):
    """
    variables
    - inputs: output of attn layer(attn_outputs)
        - shape:[bs, len_query_seq, d_hidn]
    return
    - output
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels=self.config.d_hidn, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_hidn, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        # (bs, d_ff, n_seq)
        output = self.active(self.conv1(inputs.transpose(1, 2)))
        # (bs, n_seq, d_hidn)
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        # (bs, n_seq, d_hidn)
        return output
    
class ScaledDotProductAttention(nn.Module):
    """
    variables
    - V, Q: [bs, len_seq, d_hidn]
    - attn_prob: [bs, n_head, len_query_seq, len_key_seq]
    - context: [bs, n_head, len_query_seq, d_hidn]
    calculate
    - scores: Q * K.T = [bs, len_seq, d_hidn] * [bs, d_hidn, len_seq]
        - shape: [bs, len_seq, len_seq]
    - attn_prob: dropout(softmax(score))
    - context: torch.matmul(attn_prob, Value vector)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (self.config.d_head ** 0.5)
    
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9) # softmax에서 sum!=0 위해 -1e9
        
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        
        context = torch.matmul(attn_prob, V)
        # (bs, n_head, n_q_seq, d_v), (bs, n_head, n_q_seq, n_v_seq)
        return context, attn_prob
    
class MultiHeadAttention(nn.Module):
    """
    weight mat
    - W_Q, W_K, W_K: [d_hidn, n_head*d_head]
    
    vectors
    - q_s, k_s, v_s: [bs, n_head, seq_len, d_head]
    
    calculate
    - eg. W_Q(Q): Q * W_Q = [bs, seq_len, d_hidn] * [d_hidn, n_head*d_head] 
                            = [bs, seq_len, n_head*d_head] 
                            <- ([seq_len * d_head] mat = embedded Q가 n_head만큼 있음) * bs
    - eg. W_Q(Q).view(bs, -1, n_head, d_head).transpose(1, 2) = (bs, n_head, -1=seq_len, d_head) 
    
    return
    - output: [bs, len_query_seq, d_hidn]
    - attn_prob: [bs, n_head, len_query_seq, len_key_seq]
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # n_head의 K, Q, V 한번에 생성
        self.W_Q = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_K = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_V = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.n_head * self.config.d_head, self.config.d_hidn)
        self.dropout = nn.Dropout(config.dropout)
        
    # n_head의 K, Q, V 한번에 생성
    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)

        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.n_head * self.config.d_head) # [bs, seq_len, n_head * d_head]
        output = self.linear(context) # 각 head의 output은 position wise sum이므로 해당 calculate 가능
        output = self.dropout(output)
        
        return output, attn_prob
    
class EncoderLayer(nn.Module):
    """
    variables
    - attn_outputs: [bs, lne_query_seq, d_hidn]
    - ffn_outputs: [bs, lne_query_seq, d_hidn]
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
    
    def forward(self, inputs, attn_mask):
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        att_outputs = self.layer_norm1(inputs + att_outputs)
        
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        
        return ffn_outputs, attn_prob
    
class Encoder(nn.Module):
    """
    variable
    - inputs: [bs, seq_len] <- inputs before embedding elems of which refer to voc_idx
    - outputs: embeded output; [bs, seq_len, d_hidn];
    - final outputs: result context vector of attn layers; [bs, seq_len, d_hidn];
    - attn_mask: [bs, seq_q_len, seq_k_len]
    - attn_prob: [bs, seq_q_len, d_hidn] <- 최종 결과물
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_hidn)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_enc_seq + 1, self.config.d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])
    
    def forward(self, inputs):
        position = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(self.config.i_pad)
        position.masked_fill_(pos_mask, 0)

        outputs = self.enc_emb(inputs) + self.pos_emb(position)
        
        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_pad)

        attn_probs = []
        for layer in self.layers:
            outputs, attn_prob = layer(outputs, attn_mask) # outputs = ffn_outputs
            attn_probs.append(attn_prob)
        return outputs, attn_probs
    
class DecoderLayer(nn.Module):
    """
    1st attn layer: self attention layer
    - Q, K, V = decoder inputs
    2nd attn layer: decoder encoder attention layer
    - Q, K, V = 1st attn layer outputs, encoder outputs, encoder outputs
    
    MultiHeadAttention return
    - attn_outputs: [bs, seq_q_len, d_hidn]
    - attn_prob: [bs, n_head, seq_q_len, seq_k_len]
    - dec_enc_attn_prob: [bs, n_head, n_dec_seq, n_enc_seq]
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        
        self.dec_enc_attn = MultiHeadAttention(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        
        self.pos_feedforward = PoswiseFeedForwardNet(self.config)
        self.layer_norm3 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        
    def forward(self, dec_inputs, enc_outputs, self_attn_mask, dec_enc_attn_mask):
        self_attn_outputs, self_attn_prob = self.self_attn(dec_inputs, dec_inputs, dec_inputs, self_attn_mask) 
        self_attn_outputs = self.layer_norm1(dec_inputs + self_attn_outputs) # LaterNorm(residual)
        
        dec_enc_attn_outputs, dec_enc_attn_prob = self.dec_enc_attn(self_attn_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_enc_attn_outputs = self.layer_norm2(self_attn_outputs + dec_enc_attn_outputs) # LaterNorm(residual)
        
        ffn_outputs = self.pos_feedforward(dec_enc_attn_outputs)
        ffn_outputs = self.layer_norm3(dec_enc_attn_outputs + ffn_outputs)
        
        return ffn_outputs, self_attn_prob, dec_enc_attn_prob
    
class Decoder(nn.Module):
    """
    variables
    - dec_inputs: [bs, seq_len]
    - dec_outputs: token & postion embedded inputs
        - shape: [bs, seq_len, d_hidn]
    - dec_attn_pad_mask: mask padded key seq
    - dec_attn_decoder_mask: mask triangular part
    - dec_self_attn_mask: dec_attn_pad_mask + dec_attn_decoder_mask
        - or(+) operation both "dec_attn_pad_mask" and "dec_attn_decoder_mask"
    - dec_enc_attn_mask: masking key vector(=enc_inputs)
    
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_hidn)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_dec_seq + 1, self.config.d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])
        
    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        position = torch.arange(dec_inputs.size(1), device=dec_inputs.device, dtype=dec_inputs.dtype)\
                        .expand(dec_inputs.size(0), dec_inputs.size(1)).contiguous() + 1
        pos_mask = dec_inputs.eq(self.config.i_pad)
        position.masked_fill_(pos_mask, 0)
        
        dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(position)
        
        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_pad) # get_attn_pad_mask(seq_q, seq_k, i_pad)
        dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs) # 대각선 기준 위 mask
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0)
        
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.config.i_pad)
        
        self_attn_probs, dec_enc_attn_probs = [], []
        for layer in self.layers:
            # (bs, n_dec_seq, d_hidn), (bs, n_dec_seq, n_dec_seq), (bs, d_dec_seq, n_enc_seq)
            dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            self_attn_probs.append(self_attn_prob)
            dec_enc_attn_probs.append(dec_enc_attn_prob)
        return dec_outputs, self_attn_probs, dec_enc_attn_probs
    
class Transformer(nn.Module):
    """
    variables
    - enc_outputs: [bs, len_enc_seq, d_hidn]
    - enc_self_attn_probs: [bs, n_head, len_enc_seq, len_enc_seq]
    - dec_outputs: [bs, len_seq, d_hidn]
    - dec_self_attn_probs: [bs, n_head, len_dec_seq, len_dec_seq]
    - dec_enc_dec_porbs: [bs, n_head, len_dec_seq, len_enc_seq]
    """
    def __init__(self, config):
        super().__init__()
        self.config = config 
        
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attn_probs = self.encoder(enc_inputs)
        dec_outputs, dec_self_attn_probs, dec_enc_attn_probs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs
    
class ReviewSentClf(nn.Module):
    """
    Transformer classification
    - average or max pooling over hidden_dim -> linear(in_feature=-1, out_feature=n_class)
    
    torch.max() return values and indices
    - dec_outputs(values): [bs, d_hidn] <- max pooling
    
    logits: [bs, n_output]
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = Transformer(self.config)
        self.feedforward = nn.Linear(self.config.d_hidn, self.config.n_output, bias=False)
        
    def forward(self, enc_inputs, dec_inputs):
        dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.transformer(enc_inputs, dec_inputs)
        dec_outputs, _ = torch.max(dec_outputs, dim=1)
        logits = self.feedforward(dec_outputs)
        
        return logits, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs