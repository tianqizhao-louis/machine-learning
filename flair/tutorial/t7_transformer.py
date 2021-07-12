from torch.optim.adam import Adam

from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


# 1. get the corpus
corpus: Corpus = TREC_6()

# 2. create the label dictionary
label_dict = corpus.make_label_dictionary()

# 3. initialize transformer document embeddings (many models are available)
document_embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased', fine_tune=True)

# 4. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

# 5. initialize the text classifier trainer with Adam optimizer
trainer = ModelTrainer(classifier, corpus, optimizer=Adam)

# 6. start the training
trainer.train('resources/taggers/trec',
              learning_rate=3e-5, # use very small learning rate
              mini_batch_size=16,
              mini_batch_chunk_size=4, # optionally set this if transformer is too much for your machine
              max_epochs=5, # terminate after 5 epochs
              )
