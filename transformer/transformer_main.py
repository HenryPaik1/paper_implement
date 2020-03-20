from transformer import *
from transformer_utils import *
import sentencepiece as spm
from types import SimpleNamespace
import pandas as pd

def main():
    # load vocab
    VOCAB_PATH = "/home/henry/Documents/wrapper/source"
    vocab_file = f"{VOCAB_PATH}/kowiki.model"
    vocab = spm.SentencePieceProcessor()
    vocab.load(vocab_file)

    # load config
    config = dict({
        "n_enc_vocab": len(vocab),
        "n_dec_vocab": len(vocab),
        "n_enc_seq": 256,
        "n_dec_seq": 256,
        "n_layer": 6,
        "d_hidn": 256,
        "i_pad": 0,
        "d_ff": 1024,
        "n_head": 4,
        "d_head": 64,
        "dropout": 0.1,
        "layer_norm_epsilon": 1e-12,
        "n_output": 2,
        "device": None
    })
    config = SimpleNamespace(**config)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset and dataloader
    DATA_PATH = "/home/henry/Documents/wrapper/source/"
    bs = 128
    train_dataset = CustomDataset(vocab, data_fn=DATA_PATH + 'ratings_train.json')
    train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=custom_collate_fn)
    test_dataset = CustomDataset(vocab, data_fn=DATA_PATH + 'ratings_test.json')
    test_loader = DataLoader(test_dataset, batch_size=bs, collate_fn=custom_collate_fn)

    config.n_output = 2

    # train and eval
    learning_rate = 5e-5
    n_epoch = 10

    model = ReviewSentClf(config)
    model.to(config.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_ls, score_ls = [], []
    best_epoch, best_loss, best_score = 0, 0, 0
    for epoch in range(n_epoch):
        loss = train_model(config, epoch, model, criterion, optimizer, train_loader)
        score = eval_model(config, model, test_loader)
        loss_ls.append(loss)
        score_ls.append(score)

        if best_score < score:
            best_epoch, best_loss, best_score = epoch, loss, score
            
            fn = 'model.pkl'
            torch.save(model.state_dict(), fn)

    data = {
         'loss': loss_ls,
         'score': score_ls
    }

    pd.DataFrame(data).to_pickle('result.pkl', index=False)

if __name__ == "__main__":
    main()