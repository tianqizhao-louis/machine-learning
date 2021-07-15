from flair.data import Corpus
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter
from flair.datasets import ColumnDataset


# load the model you trained
model = SequenceTagger.load('resources/ner/slicing_1/best-model.pt')
# won't work in local computer

# create example sentence
columns = {0: 'text', 3: 'ner'}
text = ColumnDataset(path_to_column_file='conll_03/slice_2',
                     column_name_map=columns)
# splitter = SegtokSentenceSplitter()
# sentences = splitter.split(text)

# predict tags and print
# model.predict(sentences)
#
# for sentence in sentences:
#     print(sentence.to_tagged_string())

model.predict(text)

with open('tagged_sentence.txt', 'w', encoding='utf-8') as f:
    for line in text.sentences:
        f.write(str(line.to_tagged_string()))
        f.write('\n')

with open('original_sentence.txt', 'w', encoding='utf-8') as f:
    for line in text.sentences:
        f.write(str(line))
        f.write('\n')
