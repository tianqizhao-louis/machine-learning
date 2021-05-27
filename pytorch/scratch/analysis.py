import time
from collections import Counter
from functools import partial

import csv
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.vocab import Vocab
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
from pathlib import Path


def process_to_pytorch(file_path, max_lines):
    """ process the file into a list of tuple (index, sentence)

    instead of using pandas, just use csv reader
    """

    # read the tsv file
    tsv_file = open(file_path)
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    list_of_tokens = []

    i = 0
    for row in read_tsv:
        if i < max_lines:
            # index
            index = 0
            if row[0] == "2":
                index = 1

            # tuple of (index, sentence)
            list_of_tokens.append((index, row[1].split(" ")))
            i += 1

    tsv_file.close()
    return list_of_tokens


def read_params(json_file_path):
    """ Read parameters, like epochs

    return as a dict of all parameters
    """
    with open(json_file_path, 'r') as f:
        json_params = json.load(f)

    return json_params


def vocab_counter(train_dataset):
    """ Calculate the number of unique vocabs in the training dataset

    Return the Vocab counter and the size of the Vocab
    """
    vocab = Vocab(Counter(tok for _, tokens in train_dataset for tok in tokens))
    return vocab, len(vocab)


def unique_label(train_dataset):
    """ Calculate the number of unique labels in the training dataset

    Return the set of unique labels and the size
    """
    label = set(unique for unique, _ in train_dataset)
    return label, len(label)


def collate_batch(batch, vocab, device):
    """ Process the dataset into applicable tensors"""
    labels_list = []
    tokens_list = []
    offsets_list = []
    last_offset = 0

    for label, tokens in batch:
        # Labels are already 0/1
        labels_list.append(label)

        offsets_list.append(last_offset)
        # Increment for next sequence
        last_offset += len(tokens)

        token_indices = torch.tensor([vocab[tok] for tok in tokens], dtype=torch.long)
        tokens_list.append(token_indices)

    labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=device)
    tokens_tensor = torch.cat(tokens_list).to(device)
    offsets_tensor = torch.tensor(offsets_list, dtype=torch.long, device=device)

    return labels_tensor, tokens_tensor, offsets_tensor


class TextClassificationModel(nn.Module):

    def __init__(self, embed_dim, num_class, pretrained_embedding, vocab_size, use_pretrained):
        super(TextClassificationModel, self).__init__()
        if use_pretrained:
            self.embedding = nn.EmbeddingBag.from_pretrained(pretrained_embedding, freeze=False, sparse=True)
            self.linear = nn.Linear(embed_dim, num_class)
        else:
            self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
            self.linear = nn.Linear(embed_dim, num_class)
            self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.linear.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.linear(embedded)


def count_accuracy(torch, dataloader, model):
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for iter_, (labels, sequences, offsets) in enumerate(dataloader):
            output = model(sequences, offsets)
            total_acc += (output.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
        accu_val = total_acc / total_count
    return accu_val


def main():
    # read parameters from json
    params = read_params("param.json")
    epochs = params["epochs"]
    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    emb_size = params["emb_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the dataset
    train_dataset = process_to_pytorch(params["train_file_path"], 50000)
    test_dataset = process_to_pytorch(params["test_file_path"], 10000)

    # split the dataset
    num_train = int(len(train_dataset) * 0.9)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])


    # get the vocab
    vocab, vocab_size = vocab_counter(split_train_)

    # get unique label
    unique_label_set, size_unique_label = unique_label(split_train_)
    print(f"{size_unique_label} unique labels: {unique_label_set}")

    collator = partial(collate_batch, vocab=vocab, device=device)

    # make dataset into Dataloader object for processing
    train_dataloader = DataLoader(split_train_, batch_size=batch_size, collate_fn=collator,
                                  shuffle=True)
    valid_dataloader = DataLoader(split_valid_, batch_size=batch_size, collate_fn=collator,
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator,
                                 shuffle=True)

    # use glove to vectorize words
    glove_output_path = Path(params["glove_format_path"])
    if not glove_output_path.exists():
        glove2word2vec(glove_input_file=params["glove_file_path"], word2vec_output_file=params["glove_output_file_path"])
        vec_model = gensim.models.KeyedVectors.load_word2vec_format(params["glove_format_path"])
        pretrained_embedding = torch.FloatTensor(vec_model.vectors)
    else:
        vec_model = gensim.models.KeyedVectors.load_word2vec_format(params["glove_format_path"])
        pretrained_embedding = torch.FloatTensor(vec_model.vectors)

    # init the model
    use_pretrained = params["use_pretrained"]
    model = TextClassificationModel(emb_size, size_unique_label, pretrained_embedding, vocab_size, use_pretrained).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
    total_accu = None

    # run_epoch(epochs, model, optimizer, criterion, scheduler, total_accu, train_dataloader, valid_dataloader, test_dataloader)
    train_start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 250

        for iter, (labels, sequences, offsets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(sequences, offsets)
            loss = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (output.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
            if iter % log_interval == 0 and iter > 0:
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f}'.format(epoch, iter, len(train_dataloader),
                                                  total_acc / total_count))
                total_acc, total_count = 0, 0

        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for iter, (labels, sequences, offsets) in enumerate(valid_dataloader):
                output = model(sequences, offsets)
                total_acc += (output.argmax(1) == labels).sum().item()
                total_count += labels.size(0)
            accu_val = total_acc / total_count

        epoch_lr = scheduler.get_last_lr()[0]
        if total_accu is not None and total_accu > accu_val:
          scheduler.step()
        else:
           total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | lr: {} | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               epoch_lr,
                                               accu_val))
        print('-' * 59)

    training_time = time.time() - train_start_time
    print(f"training time {training_time:5.2f}s")

    print('Checking the results of test dataset.')
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for iter, (labels, sequences, offsets) in enumerate(test_dataloader):
            output = model(sequences, offsets)
            total_acc += (output.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
        accu_test = total_acc / total_count
    print('test accuracy {:8.3f}'.format(accu_test))


if __name__ == '__main__':
    main()
