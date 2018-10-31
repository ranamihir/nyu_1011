import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import pickle as pkl
import math
import random
import numpy as np
random.seed(1111)

import pickle as pkl

def tokenize_dataset(dataset): 
    token_dataset = []
    # we are keeping track of all tokens in dataset 
    # in order to create vocabulary later
    all_tokens = []
    
    with open(dataset, 'r') as dataset_file:
        for sample in dataset_file:
            tokens = sample.strip().split() + ['</s>']
            #token_dataset.append(tokens)
            all_tokens += tokens

    return all_tokens

val_data = 'data/ptb.valid.bpe.txt'
test_data = 'data/ptb.test.bpe.txt'
train_data = 'data/ptb.train.bpe.txt'

print ("Tokenizing val data")
val_data_tokens = tokenize_dataset(val_data)
pkl.dump(val_data_tokens, open("val_bpe_tokens.p", "wb"))

# test set tokens
print ("Tokenizing test data")
test_data_tokens = tokenize_dataset(test_data)
pkl.dump(test_data_tokens, open("test_bpe_tokens.p", "wb"))

# train set tokens
print ("Tokenizing train data")
train_data_tokens = tokenize_dataset(train_data)
pkl.dump(train_data_tokens, open("train_bpe_tokens.p", "wb"))

# Then, load preprocessed train, val and test datasets
train_data_tokens = pkl.load(open("train_bpe_tokens.p", "rb"))

val_data_tokens = pkl.load(open("val_bpe_tokens.p", "rb"))
test_data_tokens = pkl.load(open("test_bpe_tokens.p", "rb"))

# double checking
print ("Train dataset size is {}".format(len(train_data_tokens)))
print ("Val dataset size is {}".format(len(val_data_tokens)))
print ("Test dataset size is {}".format(len(test_data_tokens)))

from collections import Counter

max_vocab_size = 20000
def build_vocab(all_tokens):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(0,len(vocab)))) 
    return token2id, id2token

token2id, id2token = build_vocab(train_data_tokens)

# convert token to id in the dataset
def token2index_dataset(tokens_data):
    indices_data = []
    for token in tokens_data:
        token_id = token2id[token] if token in token2id else token2id['<unk>'] 
        indices_data.append(token_id)
    return indices_data

train_data_indices = torch.LongTensor(token2index_dataset(train_data_tokens))
val_data_indices = torch.LongTensor(token2index_dataset(val_data_tokens))
test_data_indices = torch.LongTensor(token2index_dataset(test_data_tokens))

# double checking
print ("Train dataset size is {}".format(len(train_data_indices)))
print ("Val dataset size is {}".format(len(val_data_indices)))
print ("Test dataset size is {}".format(len(test_data_indices)))

def batchify(data, bsz, random_start_idx=False):
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    if random_start_idx:
        start_idx = random.randint(0, data.size(0) % bsz - 1)
    else:
        start_idx = 0
    data = data.narrow(0, start_idx, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

eval_batch_size = 32
val_data = batchify(val_data_indices, eval_batch_size)
test_data = batchify(test_data_indices, eval_batch_size)

import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        self.init_weights()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.hidden_size),
                    weight.new_zeros(self.num_layers, bsz, self.hidden_size))


embed_size = 200
hidden_size = 400
num_layers = 2
num_epochs = 10
batch_size = 16
lr = 0.01
dropout = 0.3
max_seq_len = 35
vocab_size = len(token2id)
model = RNNModel(vocab_size, embed_size, hidden_size, num_layers, dropout)
with open('model.pt', 'rb') as f:
    print("loading pretrain model")
    model = torch.load(f)

def get_batch(source, i, max_seq_len):
    seq_len = min(max_seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(data_source, max_seq_len):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, max_seq_len):
            data, targets = get_batch(data_source, i, max_seq_len)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, vocab_size)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


clip = 0.3
log_interval = 200
def train():
    model.train()
    total_loss = 0.
    ntokens = vocab_size
    hidden = model.init_hidden(batch_size)
    train_data = batchify(train_data_indices, batch_size, random_start_idx=True)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, max_seq_len)):
        data, targets = get_batch(train_data, i, max_seq_len)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch %log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // max_seq_len, lr,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0

best_val_loss = np.inf
criterion = nn.CrossEntropyLoss()

for epoch in range(1, num_epochs+1):
    train()
    val_loss = evaluate(val_data, max_seq_len)
    print('-' * 89)
    print('| end of epoch {:3d} | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, 
                                           val_loss, math.exp(val_loss)))
    print('-' * 89)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        with open('model.pt', 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
    else:
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        lr /= 4.0

