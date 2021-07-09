from flair.data import Corpus
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter
from flair.datasets import ColumnDataset


# load the model you trained
model = SequenceTagger.load('resources/ner/slicing_1/final-model.pt')
# won't work in local computer

# create example sentence
columns = {0: 'text', 1: 'ner'}
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

print(text.sentences)