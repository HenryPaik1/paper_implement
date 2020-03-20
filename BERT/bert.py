import os, inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from transformer.transformer import *

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        
    def forward(self, inputs, attn_mask):
        attn_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        attn_outputs = self.layer_norm1(inputs + attn_outputs)
        
        ffn_outputs = self.pos_ffn(attn_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + attn_outputs)
        
        return ffn_outputs, attn_prob

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_hidn)
        self.pos_emb = nn.Embedding(self.config.n_enc_seq+1, self.config.d_hidn)
        self.seg_emb = nn.Embedding(self.config.n_seg_type, self.config.d_hidn)
        
        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])
        
    def forward(self, inputs, segments):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype)\
            .expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask, 0)
        
        outputs = self.enc_emb(inputs) + self.pos_emb(positions) + self.seg_emb(segments)
        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_pad)
        
        attn_probs = []
        for layer in self.layers:
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)
        return outputs, attn_probs
    
class BERT(nn.Module):
    """
    outputs: [bs, len_seq, d_hidn] <- 잘 임베딩된 input seq
    ointput_cls = outputs[:, 0].contiguous(): [bs, d_hidn]
    - classification은 [cls] token의 임베딩만 사용

    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(self.config)
        self.linear = nn.Linear(config.d_hidn, config.d_hidn)
        self.activation = torch.tanh
    
    def forward(self, inputs, segments):
        outputs, self_attn_probs = self.encoder(inputs, segments)
        outputs_cls = outputs[:, 0].contiguous()
        outputs_cls = self.linear(outputs_cls)
        outputs_cls = self.activation(outputs_cls)
        return outputs, outputs_cls, self_attn_probs
    
    def save(self, epoch, loss, path):
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'state_dict': self.state_dict()
                   }, path)
        
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save['state_dict'])
        return save['epoch'], save['loss']
    
class BERTpretrain(nn.Module):
    """
    self.feedforward_lm.weight
    - transformer encoder의 pretrained embedding layer weight 사용
    logits_cls: [bs, 2]
    - binary classification
    logits_lm: [bs, len_enc_seq, n_enc_vocab]
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert  = BERT(self.config)
        # cls
        self.feedforward_cls = nn.Linear(self.config.d_hidn, 2, bias=False)
        # lm
        self.feedforward_lm = nn.Linear(self.config.d_hidn, self.config.n_enc_vocab, bias=False)
        self.feedforward_lm.weight = self.bert.encoder.enc_emb.weight
    
    def forward(self, inputs, segments):
        outputs, outputs_cls, attn_probs = self.bert(inputs, segments)
        logits_cls = self.feedforward_cls(outputs_cls)
        logits_lm = self.feedforward_lm(outputs)
        return logits_cls, logits_lm, attn_probs