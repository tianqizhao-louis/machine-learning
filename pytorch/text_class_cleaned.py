import time
from collections import Counter
from functools import partial

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.vocab import Vocab


def process_to_pytorch(file_path, max_lines):
    tsv_file = pd.read_csv(filepath_or_buffer=file_path, delimiter='\t', quoting=3, nrows=max_lines, header=None)

    # using lowercase
    tsv_file.iloc[:, 1] = tsv_file.iloc[:, 1].str.lower()

    tsv_file.iloc[:, 0] = tsv_file.iloc[:, 0].astype(float)
    for row_number in range(len(tsv_file.iloc[:, 0])):
        if tsv_file.iloc[row_number, 0] == 2:
            # in pytorch, labels are int: 1 or 0 instead of 1.0 or 0.0
            tsv_file.iloc[row_number, 0] = 1
        else:
            tsv_file.iloc[row_number, 0] = 0

    return_list = []
    for row_number in tsv_file.index:
        # split with space
        return_list.append((tsv_file.iloc[row_number][0], tsv_file.iloc[row_number][1].split(" ")))

    return return_list


def collate_batch(batch, vocab, device):
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

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


def main():
    # This isn't the best way to define these as constants shouldn't be inside a function. But
    # in a real implementation, these shouldn't be constants at all.
    EPOCHS = 10
    LR = 5.0
    BATCH_SIZE = 32  # ???
    EMB_SIZE = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_dataset = process_to_pytorch('yelp_sentiment_tokenized/train_tokenized.tsv', 50000)
    test_dataset = process_to_pytorch('yelp_sentiment_tokenized/test_tokenized.tsv', 10000)
    num_train = int(len(train_dataset) * 0.9)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    # Get vocab from training data
    token_counter = Counter(tok for _, tokens in split_train_ for tok in tokens)
    vocab = Vocab(token_counter)
    vocab_size = len(vocab)

    # Set up batch collator
    collator = partial(collate_batch, vocab=vocab, device=device)  # ???

    # Create Dataloaders
    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, collate_fn=collator,
                                  shuffle=True)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, collate_fn=collator,
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collator,
                                 shuffle=True)

    # Get labels from the training data
    unique_labels = set(label for label, _ in split_train_)
    num_class = len(unique_labels)
    print(f"{num_class} unique labels: {unique_labels}")

    model = TextClassificationModel(vocab_size, EMB_SIZE, num_class).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
    total_accu = None

    train_start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
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
