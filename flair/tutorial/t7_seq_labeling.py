# Training a Sequence Labeling Model
from flair.data import Corpus
from flair.datasets import UD_ENGLISH
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.embeddings import FlairEmbeddings
from flair.trainers import ModelTrainer
from flair.data import Sentence

# downsample the data to 10%

# 1. get the corpus
corpus: Corpus = UD_ENGLISH().downsample(0.1)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'pos'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

# 4. initialize embeddings
embedding_types = [

    WordEmbeddings('glove'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    # FlairEmbeddings('news-forward'),
    # FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# 6. initialize trainer
trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-pos',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)


# load the model you trained
model = SequenceTagger.load('resources/taggers/example-pos/final-model.pt')

# create example sentence
sentence = Sentence('I love Berlin')

# predict tags and print
model.predict(sentence)

print(sentence.to_tagged_string())