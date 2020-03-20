from bert import *
from bert_utils import *
import sentencepiece as spm
from types import SimpleNamespace


PATH = '/home/henry/Documents/wrapper/source/'
VOCAB_PATH = "/home/henry/Documents/wrapper/source"
vocab_file = f"{VOCAB_PATH}/kowiki.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

config = dict({
    "n_enc_vocab": len(vocab),
    "n_enc_seq": 256,
    "n_seg_type": 2,
    "n_layer": 6,
    "d_hidn": 256,
    "i_pad": 0,
    "d_ff": 1024,
    "n_head": 4,
    "d_head": 64,
    "dropout": 0.1,
    "layer_norm_epsilon": 1e-12
})
config = SimpleNamespace(**config)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config.device = device

# Create pretrain dataset
# in_file = PATH + "kowiki.txt"
# out_file = PATH + "kowiki_bert" + "_{}.json"
# count = 1
# n_seq = 256
# mask_prob = 0.15

# make_pretrain_data(vocab, in_file, out_file, count, n_seq, mask_prob)

batch_size = 128
dataset = PretrainDataset(vocab, PATH+'kowiki_bert_0.json')
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size\
                                           , shuffle=True, collate_fn=pretrain_collate_fn)
learning_rate = 5e-5
n_epoch = 10

# define & load model
model = BERTpretrain(config)
save_pretrain = PATH + 'bert_pretrain_weights.pkl'
best_epoch, best_loss = 0, 0
if os.path.isfile(save_pretrain):
    best_epoch, best_loss = model.bert.load(save_pretrain)
    best_epoch += 1

# train
model.to(config.device)
criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
criterion_cls = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_ls = []
offset = best_epoch
for step in range(n_epoch):
    epoch = step + offset
#     if 0 < step:
#         del train_loader
#         dataset = PretrainDataset(vocab, PATH + 'kowiki_bert_0.json')
#         train_loader = DataLoader(dataset, batch_size=batch_size, \
#                                   suffle=True, collate_fn=pretrain_collate_fn)
    loss = train_epoch(config, epoch, model, criterion_lm, criterion_cls,\
                      optimizer, train_loader)
    loss_ls.append(loss)
    model.bert.save(epoch, loss, save_pretrain)