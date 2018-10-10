import numpy as np
import pandas as pd
import os
import pickle
import argparse
from collections import Counter

import re
from unidecode import unidecode
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--project-dir', metavar='PROJECT_DIR', dest='project_dir', help='path to project directory', required=False)
parser.add_argument('--batch-size', metavar='BATCH_SIZE', dest='batch_size', help='batch size', required=False, type=int, default=64)
parser.add_argument('--epochs', metavar='EPOCHS', dest='epochs', help='number of epochs', required=False, type=int, default=20)
parser.add_argument('--device', metavar='DEVICE', dest='device', help='device', required=False)
parser.add_argument('--device-id', metavar='DEVICE_ID', dest='device_id', help='device id of gpu', required=False, type=int)
parser.add_argument('--target', metavar='TARGET', dest='target', help='target column = sentiment | rating', required=False, type=str, default='sentiment')
parser.add_argument('--force', action='store_true', help='overwrites all existing data', required=False)
parser.add_argument('--scheduler', action='store_true', help='uses LR step scheduler', required=False)
parser.add_argument('--raw', action='store_true', help='use raw data', required=False)
args = parser.parse_args()


# Globals
PROJECT_DIR = args.project_dir if args.project_dir else '/home/mihir/Desktop/GitHub/nyu/nyu_1011/homeworks/hw1/'
DATA_DIR, PLOTS_DIR = os.path.join(PROJECT_DIR, 'data'), os.path.join(PROJECT_DIR, 'plots')
PLOTS_DIR = os.path.join(PROJECT_DIR, 'plots')
NUM_VAL = 5000

MAX_N = 3           # max n for n-grams
LR = 1e-3           # learning rate
VOCAB_SIZE = 10000  # max vocab size
MAX_SENTENCE_LENGTH = 200
EMB_DIM = 256       # size of embedding
BATCH_SIZE = args.batch_size    # input batch size for training
N_EPOCHS = args.epochs          # number of epochs to train
DEVICE = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cuda:2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.device_id and DEVICE == 'cuda':
    torch.cuda.set_device(args.device_id)
CLEAN = not args.raw
FORCE = args.force

# Set target column and number of outputs
TARGET = args.target
NUM_OUTPUTS = 2 if TARGET == 'sentiment' else 10

# LR Scheduler
SCHEDULER_STEP_SIZE = 1
SCHEDULER_GAMMA = 0.9

# Save index 0 for unk and 1 for pad
PAD_IDX = 0
UNK_IDX = 1


def load_dataset(dataset='train', clean=True):
    data_path = os.path.join(DATA_DIR, dataset)
    data = []
    for sentiment in ['pos', 'neg']:
        target = 1 if sentiment == 'pos' else 0
        data_target_path = os.path.join(data_path, sentiment)
        for file in os.listdir(data_target_path):
            file_path = os.path.join(data_target_path, file)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file_text:
                    text = file_text.readlines()[0]
                    if clean:
                        text = re.sub('<[^<]+?>', '', text).replace(')', ' ').replace('(', ' ')
                    rating = file.replace('.txt', '').split('_')[-1]
                    data.append([text, target, rating])
    data = pd.DataFrame(data, columns=['text', 'sentiment', 'rating'])
    data['text'] = data['text'].astype(str)
    data['sentiment'] = data['sentiment'].astype(int)
    data['rating'] = data['rating'].astype(int) - 1 # 0 <= label < 10 for PyTorch
    return data

def split_train_val(train_data):
    train_data.sample(frac=1, random_state=1337)
    val_data = train_data[:NUM_VAL]
    train_data = train_data[NUM_VAL:]
    return train_data, val_data

def load_train_val_datasets(clean=True, force=False):
    train_data_path = os.path.join(DATA_DIR, 'train.pkl')
    val_data_path = os.path.join(DATA_DIR, 'val.pkl')
    if not force and os.path.exists(train_data_path) and os.path.exists(val_data_path):
        train_data = pickle.load(open(train_data_path, 'rb'))
        val_data = pickle.load(open(val_data_path, 'rb'))
    else:
        train_data = load_dataset('train', clean)
        train_data, val_data = split_train_val(train_data)
        pickle.dump(train_data, open(train_data_path, 'wb'))
        pickle.dump(val_data, open(val_data_path, 'wb'))
    return train_data.reset_index(drop=True), val_data.reset_index(drop=True)

def load_test_dataset(clean=True, force=False):
    test_data_path = os.path.join(DATA_DIR, 'test.pkl')
    if not force and os.path.exists(test_data_path):
        test_data = pickle.load(open(test_data_path, 'rb'))
    else:
        test_data = load_dataset('test', clean)
        pickle.dump(test_data, open(test_data_path, 'wb'))
    return test_data

def prepare_stopwords():
    NEGATE = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere","no",
     "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

    stopwords = STOP_WORDS.copy()
    for word in STOP_WORDS:
        if word in NEGATE:
            stopwords.remove(word)

    return stopwords

def clean_and_tokenize(tokens, max_n, stopwords, punctuations, clean=True):
    if clean:
        tokens = [tok.lemma_.lower().strip() for tok in tokens]
        tokens = [unidecode(tok) for tok in tokens if (tok not in stopwords and tok not in punctuations)]
    else:
        tokens = [tok.text for tok in tokens]
    for n in range(1, max_n+1):
        tokens.extend([' '.join(list(tokens[i:i+n])) for i in range(len(tokens)-n+1)])
    return tokens

def tokenize_dataset(data, tokenizer, max_n, stopwords, punctuations, dataset='train', clean=True, force=False):
    tokens_data_path = os.path.join(DATA_DIR, '{}_tokenized_upto_{}_{}.pkl'.format(dataset, max_n, 'clean' if clean else 'raw'))
    all_train_ngrams_path = os.path.join(DATA_DIR, 'all_train_ngrams_upto_{}_{}.pkl'.format(max_n, 'clean' if clean else 'raw'))
    if not force and os.path.exists(tokens_data_path):
        tokens_data = pickle.load(open(tokens_data_path, 'rb'))
        if dataset == 'train':
            all_train_ngrams = pickle.load(open(all_train_ngrams_path, 'rb'))
            return tokens_data, all_train_ngrams
        return tokens_data
    else:
        tokens_data, all_train_ngrams = [], []
        for sample in tokenizer.pipe(data['text'], batch_size=512, n_threads=-1):
            tokens = clean_and_tokenize(sample, max_n, stopwords, punctuations, clean)
            tokens_data.append(tokens)
            all_train_ngrams.extend(tokens if dataset == 'train' else [])
        tokens_data = pd.Series(tokens_data)
        pickle.dump(tokens_data, open(tokens_data_path, 'wb'))
        if dataset == 'train':
            all_train_ngrams = Counter(all_train_ngrams)
            pickle.dump(all_train_ngrams, open(all_train_ngrams_path, 'wb'))
            return tokens_data, all_train_ngrams
    return tokens_data

def build_vocabulary(all_ngrams, vocab_size):
    '''
    Returns:
    id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    token2id: dictionary where keys represent tokens and corresponding values represent indices
    '''

    vocabulary, count = zip(*all_ngrams.most_common(vocab_size))
    id2token = list(vocabulary)
    token2id = dict(zip(vocabulary, range(2, 2+len(vocabulary))))
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token

# Convert token to id in the dataset
def token2index_dataset(tokens_data, token2id):
    indices_data = tokens_data.apply(lambda tokens: [token2id[token] if token in token2id else UNK_IDX for token in tokens])
    return indices_data

class IMDBReviewsDataset(Dataset):
    def __init__(self, data_list, target_list):
        """
        @param data_list: list of review tokens
        @param target_list: list of review targets
        """
        self.data_list = data_list
        self.target_list = target_list
        assert (len(self.data_list) == len(self.target_list))

    def __getitem__(self, key):
        """
        Triggered when dataset[i] is called
        """
        token_idx = self.data_list[key][:MAX_SENTENCE_LENGTH]
        label = self.target_list[key]
        return [token_idx, len(token_idx), label]

    def __len__(self):
        return len(self.data_list)

def imdbreviews_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []

    for datum in batch:
        length_list.append(datum[1])
        label_list.append(datum[2])

    # Padding
    for datum in batch:
        padded_vec = np.pad(np.array(datum[0]),
                            pad_width=((0, MAX_SENTENCE_LENGTH-datum[1])),
                            mode="constant", constant_values=0)
        data_list.append(padded_vec)
    return [torch.from_numpy(np.array(data_list)), torch.LongTensor(length_list), torch.LongTensor(label_list)]

class BagOfWords(nn.Module):
    """
    BagOfWords classification model
    """
    def __init__(self, vocab_size, emb_dim, num_outputs):
        """
        @param vocab_size: size of the vocabulary
        @param emb_dim: size of the word embedding
        """
        super(BagOfWords, self).__init__()

        # Pay attention to padding_idx
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim, num_outputs)

    def forward(self, data, length):
        """
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        out /= length.view(length.size()[0], 1).expand_as(out).float()

        # Return logits
        out = self.linear(out.float())
        return out

# Function for testing the model
def test_model(dataloader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0.
    total = 0.
    model.eval()
    with torch.no_grad():
        for data_batch, lengths_batch, labels_batch in dataloader:
            data_batch, lengths_batch, labels_batch = data_batch.to(DEVICE), lengths_batch.to(DEVICE), labels_batch.to(DEVICE)
            outputs = nn.functional.softmax(model(data_batch, lengths_batch), dim=1)
            predicted = outputs.max(1, keepdim=True)[1]

            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch.view_as(predicted)).sum().item()
    return (100 * correct / total)

def run_training(model, train_loader, val_loader, criterion, optimizer, n_epochs, scheduler=None):
    train_loss_history, val_accuracies = [], []
    for epoch in range(1, n_epochs+1):
        if scheduler:
            scheduler.step()
        for batch_idx, (data_batch, lengths_batch, labels_batch) in enumerate(train_loader):
            data_batch, lengths_batch, labels_batch = data_batch.to(DEVICE), lengths_batch.to(DEVICE), labels_batch.to(DEVICE)
            model.train()
            optimizer.zero_grad()
            outputs = model(data_batch, lengths_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss_history.append(loss.item())

            if batch_idx % (len(train_loader.dataset)//(10*labels_batch.shape[0])) == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * labels_batch.shape[0], len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        val_accuracy = test_model(val_loader, model)
        print('Val Accuracy: {:.2f}'.format(val_accuracy))
        val_accuracies.append(val_accuracy)

    return train_loss_history, val_accuracies

def hyperparameter_tuning(train_data, train_data_tokens, val_data, val_data_tokens, all_train_ngrams, params_dict, use_scheduler=False):
    try:
        cv_results = pd.DataFrame(columns=['batch_size', 'lr', 'emb_dim', 'vocab_size', \
                                           'max_sent_length', 'optimizer', 'ngrams', \
                                           'train_loss_hist', 'val_accuracies', 'max_accuracy'])

        params = pd.DataFrame([1]*len(params_dict['EMB_DIMs']), columns=['key'])
        for df in (params_dict['BATCH_SIZEs'], params_dict['LRs'], params_dict['EMB_DIMs'], params_dict['VOCAB_SIZEs'], \
                params_dict['MAX_SENTENCE_LENGTHs'], params_dict['OPTIMIZERS'], params_dict['NGRAMs']):
            df['key'] = 1
            params = pd.merge(params, df, on='key')
        params = params.drop('key', axis=1).drop_duplicates()

        print('Total number of configurations: {}'.format(len(params)))

        for row in params.iterrows():
            print('\n', params.iloc[row[0]:row[0]+1])

            batch_size, lr, emb_dim, vocab_size, max_sent_length, max_n = \
                int(row[1]['batch_size']), row[1]['lr'], int(row[1]['emb_dim']), \
                int(row[1]['vocab_size']), int(row[1]['max_sent_length']), row[1]['ngrams']

            print('Tokenizing train data... ', end='', flush=True)
            train_data_tokens, all_train_ngrams = tokenize_dataset(train_data, tokenizer, max_n, stopwords, punctuations, 'train', CLEAN, FORCE)
            print('Done.\nTokenizing val data... ', end='', flush=True)
            val_data_tokens = tokenize_dataset(val_data, tokenizer, max_n, stopwords, punctuations, 'val', CLEAN, FORCE)
            print('Done.')

            print("Total number of tokens in train dataset = {}".format(len(all_train_ngrams)))

            token2id, id2token = build_vocabulary(all_train_ngrams, vocab_size)

            train_data_indices = token2index_dataset(train_data_tokens, token2id)
            val_data_indices = token2index_dataset(val_data_tokens, token2id)

            train_dataset = IMDBReviewsDataset(train_data_indices, train_data[TARGET])
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=batch_size,
                                                       collate_fn=imdbreviews_collate_func,
                                                       shuffle=True)

            val_dataset = IMDBReviewsDataset(val_data_indices, val_data[TARGET])
            val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=batch_size,
                                                     collate_fn=imdbreviews_collate_func,
                                                     shuffle=True)

            model = BagOfWords(len(id2token), emb_dim, NUM_OUTPUTS).to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = {
                'adam': torch.optim.Adam(model.parameters(), lr=lr), \
                'sgd': torch.optim.SGD(model.parameters(), lr=lr)} \
                [row[1]['optimizer']]
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA) \
                        if use_scheduler else None

            train_loss_history, val_accuracies = run_training(model, train_loader, val_loader, criterion, \
                                                              optimizer, N_EPOCHS, scheduler)
            max_accuracy = np.max(val_accuracies)

            result = pd.DataFrame([[batch_size, lr, emb_dim, vocab_size, max_sent_length, \
                                    row[1]['optimizer'], max_n, train_loss_history, \
                                    val_accuracies, max_accuracy]], columns=cv_results.columns)
            cv_results = cv_results.append(result)

    except KeyboardInterrupt:
        return cv_results

    return cv_results


def main():
    train_data, val_data = load_train_val_datasets(CLEAN, FORCE)
    test_data = load_test_dataset(CLEAN, FORCE)

    print("Train dataset size is {}".format(len(train_data)))
    print("Val dataset size is {}".format(len(val_data)))
    print("Test dataset size is {}".format(len(test_data)))

    # Random sample from train dataset
    print(train_data.iloc[np.random.randint(0, len(train_data)-1)])

    # Load English tokenizer+tagger+parser+NER+word vectors, and punctuations and stopwords
    tokenizer = spacy.load('en_core_web_lg', disable=['parser', 'tagger', 'ner'])
    punctuations = string.punctuation
    stopwords = prepare_stopwords()

    print('Tokenizing train data... ', end='', flush=True)
    train_data_tokens, all_train_ngrams = tokenize_dataset(train_data, tokenizer, MAX_N, stopwords, punctuations, 'train', CLEAN, FORCE)
    print('Done.\nTokenizing val data... ', end='', flush=True)
    val_data_tokens = tokenize_dataset(val_data, tokenizer, MAX_N, stopwords, punctuations, 'val', CLEAN, FORCE)
    print('Done.\nTokenizing test data... ', end='', flush=True)
    test_data_tokens = tokenize_dataset(test_data, tokenizer, MAX_N, stopwords, punctuations, 'test', CLEAN, FORCE)
    print('Done.')

    print("Total number of tokens in train dataset = {}".format(len(all_train_ngrams)))

    token2id, id2token = build_vocabulary(all_train_ngrams, VOCAB_SIZE)

    # Check the dictionary by loading random token from it
    random_token_id = np.random.randint(0, len(id2token)-1)
    random_token = id2token[random_token_id]
    print("Token id: {}; Token: {}".format(random_token_id, id2token[random_token_id]))
    print("Token: {}; Token id: {}".format(random_token, token2id[random_token]))

    train_data_indices = token2index_dataset(train_data_tokens, token2id)
    val_data_indices = token2index_dataset(val_data_tokens, token2id)
    test_data_indices = token2index_dataset(test_data_tokens, token2id)

    train_dataset = IMDBReviewsDataset(train_data_indices, train_data[TARGET])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               collate_fn=imdbreviews_collate_func,
                                               shuffle=True)

    val_dataset = IMDBReviewsDataset(val_data_indices, val_data[TARGET])
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=BATCH_SIZE,
                                             collate_fn=imdbreviews_collate_func,
                                             shuffle=True)

    test_dataset = IMDBReviewsDataset(test_data_indices, test_data[TARGET])
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE,
                                              collate_fn=imdbreviews_collate_func,
                                              shuffle=False)

    # Model, Criterion, and Optimizer
    model = BagOfWords(len(id2token), EMB_DIM, NUM_OUTPUTS).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA) if args.scheduler else None

    train_loss_history, val_accuracies = run_training(model, train_loader, val_loader, criterion, optimizer, N_EPOCHS, scheduler)

    train_loss_history = pd.DataFrame({
        'train': train_loss_history
    })
    plt.figure(figsize=(10,8))
    train_loss_history.plot(alpha=0.5)
    plt.title('Train Loss Curve')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'train_loss_hist_{}.jpg'.format(TARGET)))

    val_accuracies = pd.DataFrame({
        'val': val_accuracies
    })
    plt.figure(figsize=(10,8))
    val_accuracies.plot(alpha=0.5)
    plt.title('Validation Accuracy Curve')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'val_accuracies_hist_{}.jpg'.format(TARGET)))

    print("After training for {} epochs:".format(N_EPOCHS))
    print("Val Accuracy: {}".format(test_model(val_loader, model)))

    params_dict = {
        'BATCH_SIZEs': pd.DataFrame([64], columns=['batch_size']),
        'LRs': pd.DataFrame([1e-3], columns=['lr']),
        'EMB_DIMs': pd.DataFrame([100, 512, 1024], columns=['emb_dim']),
        'VOCAB_SIZEs': pd.DataFrame([10000, 20000, 50000], columns=['vocab_size']),
        'OPTIMIZERS': pd.DataFrame(['adam', 'sgd'], columns=['optimizer']),
        'MAX_SENTENCE_LENGTHs': pd.DataFrame([200, 500], columns=['max_sent_length']),
        'NGRAMs': pd.DataFrame(list(range(1,5)), columns=['ngrams'])
    }

    cv_results = hyperparameter_tuning(train_data, train_data_tokens, val_data, val_data_tokens, all_train_ngrams, params_dict, args.scheduler)
    pickle.dump(cv_results, open(os.path.join(DATA_DIR, 'cv_results_{}.pkl'.format(TARGET)), 'wb'))

    best_conf = cv_results[cv_results['max_accuracy'] == np.max(cv_results['max_accuracy'])].iloc[0]
    print('Best hyperparameter configuration:\n', best_conf)

    plt.figure(figsize=(10,8))
    plt.plot(best_conf['train_loss_hist'])
    plt.title('Train Loss Curve')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'train_loss_best_{}.jpg'.format(TARGET)))

    plt.figure(figsize=(10,8))
    plt.plot(best_conf['val_accuracies'])
    plt.title('Validation Accuracy Curve')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'val_accuracies_best_{}.jpg'.format(TARGET)))

    print("Test Accuracy: {}".format(test_model(test_loader, model)))

if __name__ == '__main__':
    main()
