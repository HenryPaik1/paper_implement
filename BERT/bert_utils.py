import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json

def create_pretrain_mask(tokens, mask_cnt, vocab_list):
    """
    masking subwords(15% of entire subwords)
    - mask_cnt: len(subwords) * 0.15
    - [MASK]: 80% of masking candidate token
    - original token: 10% of masking candidate token
    - another token: 10% of masking candidate token
    """
    candidate_idx = []

    ## subwords in the same list augment a sementic word 
    ## eg. [[0], [1], [2], [4, 5]] -> token_idx 4 + 5 is semantic word
    # A list represent a sementic word
    for i, token in enumerate(tokens):
        if token == '[CLS]' or token == '[SEP]':
            continue
        if 0 < len(candidate_idx) and token.find(u'\u2581') < 0: #  LOWER ONE EIGHTH BLOCK
#        if 0 < len(candidate_idx) and token.find('_') < 0: #  test code
            candidate_idx[-1].append(i)
        else:
            candidate_idx.append([i])
    np.random.shuffle(candidate_idx)

    mask_lms = []
    for idx_set in candidate_idx:
        # check if len(mask_lms) exceeds threshold
        if len(mask_lms) >= mask_cnt:
            break
        if len(mask_lms) + len(idx_set) > mask_cnt:
            continue

        ## masking subwords with 15% probability
        ## mask_cnt is len(subwords) * 0.15 
        # iter subwords idx
        for sub_idx in idx_set:
            masked_token = None

            ### assign value to masked token: [MASK], original token, random token
            # 80% of masking candidate are replaced with '[MASK]' token
            if np.random.uniform() < 0.8:
                masked_token = '[MASK]'
            # remainng 20% of masking candidate
            else:
                # 10% of remaining preserve original token
                if np.random.uniform() < 0.5:
                    masked_token = tokens[sub_idx]
                # 10% of ones are replaced with rnadom token    
                else:
                    masked_token = np.random.choice(vocab_list)

                ### replace subword with masked_token value    
                mask_lms.append({'idx': sub_idx, 'label':tokens[sub_idx]})
                tokens[sub_idx] = masked_token
                
    mask_lms = sorted(mask_lms, key=lambda x: x['idx'])
    mask_idx = [mask_dict['idx'] for mask_dict in mask_lms]
    mask_label = [mask_dict['label'] for mask_dict in mask_lms]
#     print(candidate_idx)
#     print(mask_lms)
    print(mask_idx, mask_label)
    return tokens, mask_idx, k_label

def truncate_token(tokenA, tokenB, max_seq):
    """
    truncate long sequence
    """
    while True:
        total_len = len(tokenA) + len(tokenB)
        print('max token {}\ntotal_len {} = {} + {}'.format(max_seq, total_len, len(tokenA), len(tokenB)))
        if total_len <= max_seq:
            break
        if len(tokenA) > len(tokenB):
            tokenA.pop()
        else:
            tokenB.pop()
            
def create_pretrain_instances(paragraph_ls, paragraph_idx, paragraph, n_seq, mask_prob, vocab_list):
    """
    create NSP train set
    """
    # 3 special token: [CLS], [SEP] for sent A, [SEP] for sent B
    max_seq_len = n_seq - 2 - 1
    target_seq_len = max_seq_len # [CLS], segmentA, segmentA, ..., [SEP], segmentB, segmentB, ...

    instances = []
    temp_sentence = []
    temp_sent_seq_length = 0 # num of tokens

    max_num_tokens = 256
    target_seq_len = np.random.randint(2, max_num_tokens) # min len of tokens
    for i, sent in enumerate(paragraph):
        ## A. not the last sentence of the paragraph
        temp_sentence.append(sent)
        temp_sent_seq_length += len(sent)

        ## B. check if it is the last sentence of the paragraph
        ## or temp_sent_seq_length is longer than or equal to target_seq_len
        if i == len(paragraph) - 1 or temp_sent_seq_length >= target_seq_len:
            if temp_sentence:
                ## A. sentence A segment: from 0 to a_end
                a_end = 1
                if len(temp_sentence) != 1:
                    a_end = np.random.randint(1, len(temp_sentence))
                # append the sentences to tokenA 
                # from the front to the back
                tokenA = []
                for _, s in enumerate(temp_sentence[:a_end]):
                    tokenA.extend(s)

                ## B. sentence B segment
                tokenB = []
                # A. Actual next
                # is_next will be the label for NSP pretrain
                if len(temp_sentence) > 1 and np.random.uniform() >= 0.5:
                    is_next = True
                    for j in range(a_end, len(temp_sentence)):
                        tokenB.extend(temp_sentence[j])
                # B. random next
                else:
                    is_next = False
                    tokenB_len = target_seq_len - len(tokenA)
                    random_para_idx = para_idx
                    while para_idx == random_para_idx:
                        random_para_idx = np.random.randint(0, len(paragraph_ls))
                    random_para = paragraph[random_para_idx]

                    random_start = np.random.randint(0, len(random_para))
                    for j in range(random_start, len(random_para)):
                        tokenB.extend(random_para[j])

                truncate_token(tokenA, tokenB, max_seq)
                assert 0 < len(tokenA)
                assert 0 < len(tokenB)

                tokens = ["[CLS]"] + tokenA + ["[SEP]"] + tokenB + ["[SEP]"]
                segment = [0]*(len(tokenA)  + 2) + [1]*(len(tokenB) + 1)
                
                tokens, mask_idx, mask_label = \
                    create_pretrain_mask(tokens, int((len(tokens)-3)*mask_prob), vocab_list)
                instance = {
                    'tokens': tokens,
                    'segment': segment,
                    'is_next': is_next,
                    'mask_idx': mask_idx,
                    'mask_label': mask_label
                }

                instances.append(instance)

            # reset segment candidate
            temp_sentence = []
            temp_sent_seq_length = 0
    
    return instances

def make_pretrain_data(vocab, in_file, out_file, count, n_seq, mask_prob):
    """
    read text and return train data set format
    """
    vocab_list = []
    for id_ in range(vocab.get_piece_size()):
        if not vocab.is_unknown(id_):
            vocab_list.append(vocab.id_to_piece(id_))
    
    paragraph_ls = []
    with open(in_file, 'r') as in_f:
        paragraph = []
        for i, sent in enumerate(in_f):
            sent = sent.strip()
            
            ## blank means end of the paragraph
            if sent == '':
                # if not the beggining of the paragraph
                # it is the end of the paragraph
                if 0 < len(paragraph):
                    paragraph_ls.append(paragraph)
                    paragraph = [] # generate new paragraph list
                    # check if exceeding 100 thaousand paragraphs
                    if 1e+5 < len(paragraph_ls): 
                        break 
                        
            ## subwords in list is part of semantic token
            # eg. ['▁지','미','▁카','터']
            else:
                pieces = vocab.encode_as_pieces(sent)
                if 0 < len(pieces):
                    paragraph.append(pieces)
        if paragraph:
            paragraph_ls.append(paragraph)
    # masking def: create_pretrain_mask
    for index in range(count):
        output = out_file.format(index)
#         if os.path.isfile(output):
#             continue
        with open(output, 'w') as out_f:
            for i, paragraph in enumerate(paragraph_ls):
                masking_info = create_pretrain_instances(paragraph_ls, i, paragraph, n_seq, mask_prob, vocab_list)
                for elem in masking_info:
                    out_f.write(json.dumps(elem))
                    out_f.write('\n')    
                    
class PretrainDataset(Dataset):
    """
    eg. instance
    {tokens:
        ['[CLS]', '▁지', ', '대학교', '를', '▁졸업', '하였다', '.', '▁그', '▁후', ...],
    segment:
        [0, 0, 0, 0, 0, 0, ..., 1, 1, 1],
    is_next: True,
    mask_idx: 
        [16, 21, ..., 41],
    mask_label:
        ['▁192', '▁1', '일', '▁~', '는', ..., '▁조지', '법을']}
    """
    def __init__(self, vocab, infile):
        self.vocab = vocab
        self.labels_cls = []
        self.label_lm_ls = []
        self.sentence_ls = []
        self.segments = []
        
        with open(infile, 'r') as f:
            for i, line in enumerate(f):
                instance = json.loads(line)
                self.labels_cls.append(instance['is_next'])
                sentence = [vocab.piece_to_id(p) for p in instance['tokens']]
                
                self.sentence_ls.append(sentence)
                self.segments.append(instance['segment'])
                
                mask_idx = np.array(instance['mask_idx'], dtype=np.int)
                mask_label = np.array([vocab.piece_to_id(p) for p in instance['mask_label']], dtype=np.int)
                label_lm = np.full(len(sentence), dtype=np.int, fill_value=-1)
                label_lm[mask_idx] = mask_label
                self.label_lm_ls.append(label_lm)
    
    def __len__(self):
        assert len(self.labels_cls) == len(self.label_lm_ls)
        assert len(self.labels_cls) == len(self.sentence_ls)
        assert len(self.labels_cls) == len(self.segments)
        return len(self.labels_cls)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.labels_cls[idx]),
                torch.tensor(self.label_lm_ls[idx]),
                torch.tensor(self.sentence_ls[idx]),
                torch.tensor(self.segments[idx]),)
    
def pretrain_collate_fn(inputs):
    """
    padding batch
    """
    labels_cls, labels_lm, inputs, segments = list(zip(*inputs))
    labels_lm = torch.nn.utils.rnn.pad_sequence(labels_lm, batch_first=True, padding_value=-1)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    segments = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True, padding_value=0)
    
    batch = [
        torch.stack(labels_cls, dim=0),
        labels_lm,
        inputs,
        segments,
    ]
    return batch

def train_epoch(config, epoch, model, criterion_lm, criterion_cls, optimizer, train_loader):
    loss_ls = []
    model.train()
    print('model train')
    for i, value in enumerate(train_loader):
        labels_cls, labels_lm, inputs, segments = map(lambda x: x.to(config.device), value)
        
        optimizer.zero_grad()
        outputs = model(inputs, segments)
        logits_cls, logits_lm = outputs[0], outputs[1]
        
        loss_cls = criterion_cls(logits_cls, labels_cls)
        loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))
        loss = loss_cls + loss_lm
        
        loss_val = loss_lm.item()
        loss_ls.append(loss_val)
        
        loss.backward()
        optimizer.step()
    
    return np.mean(loss_ls)