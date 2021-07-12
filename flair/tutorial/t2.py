# Tagging with Pre-Trained Sequence Tagging Models

from flair.models import SequenceTagger
from flair.data import Sentence

tagger = SequenceTagger.load('ner')

# .predict()
sentence = Sentence('George Washington went to Washington.')
tagger.predict(sentence)
print(sentence.to_tagged_string(), '\n')

# Getting Annotated Spans, like George Washington
for entity in sentence.get_spans('ner'):
    print(entity)
# Each such Span has a text, its position in the sentence and
#   Label with a value and a score (confidence in the prediction).

# additional information: position offsets of each entity in the sentence
print(sentence.to_dict(tag_type='ner'))

# Multi-Tagging
from flair.models import MultiTagger
tagger = MultiTagger.load(['pos', 'ner'])
sentence = Sentence("George Washington went to Washington.")
tagger.predict(sentence)
print('\n', sentence)

# Tagging a German sentence
tagger = SequenceTagger.load('de-ner')
sentence = Sentence('George Washington ging nach Washington.')
tagger.predict(sentence)
print('\n', sentence.to_tagged_string())

# Semantic Frame Detection
print()
tagger = SequenceTagger.load('frame')
sentence_1 = Sentence('George returned to Berlin to return his hat.')
sentence_2 = Sentence('He had a look at different hats.')

tagger.predict(sentence_1)
tagger.predict(sentence_2)
print(sentence_1.to_tagged_string())
print(sentence_2.to_tagged_string())

# Tagging a List of Sentences
# like if you want to tag an entire text corpus
from flair.tokenization import SegtokSentenceSplitter
text = "This is a sentence. This is another sentence. I love Berlin."

print()
splitter = SegtokSentenceSplitter()
sentences = splitter.split(text)

tagger = SequenceTagger.load('ner')
tagger.predict(sentences)

for sentence in sentences:
    print(sentence.to_tagged_string())
# Using the mini_batch_size parameter of the .predict() method,
#   you can set the size of mini batches passed to the tagger.


# Tagging with Pre-Trained Text Classification Models
from flair.models import TextClassifier
classifier = TextClassifier.load('sentiment')
sentence = Sentence("enormously entertaining for moviegoers of any age.")
classifier.predict(sentence)
print('\n', sentence)
