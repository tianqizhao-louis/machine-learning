from flair.data import Corpus
from flair.datasets.sequence_labeling import CONLL_03
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.embeddings import FlairEmbeddings
from flair.trainers import ModelTrainer
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter


# load the model you trained
model = SequenceTagger.load('flair/resources/ner/ner-training/final-model.pt')

# create example sentence
text = 'Today we will see how we can use huggingface’s transformers library to summarize any given text. T5 is an abstractive summarization algorithm. It means that it will rewrite sentences when necessary than just picking up sentences directly from the original text.'
splitter = SegtokSentenceSplitter()
sentences = splitter.split(text)

# predict tags and print
model.predict(sentences)

for sentence in sentences:
    print(sentence.to_tagged_string())