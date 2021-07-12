from typing import List
from flair.data import MultiCorpus
from flair.datasets import UD_ENGLISH, UD_GERMAN
from flair.embeddings import FlairEmbeddings, TokenEmbeddings, StackedEmbeddings
from flair.training_utils import EvaluationMetric


# 1. get the corpora - English and German UD
corpus: MultiCorpus = MultiCorpus([UD_ENGLISH(), UD_GERMAN()]).downsample(0.1)

# 2. what tag do we want to predict?
tag_type = 'upos'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    # we use multilingual Flair embeddings in this task
    FlairEmbeddings('multi-forward'),
    FlairEmbeddings('multi-backward'),
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
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-universal-pos',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              write_weights=True  # a helper method to plot training curves and weights in the neural network
              )


# visualize
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('loss.tsv')
plotter.plot_weights('weights.txt')