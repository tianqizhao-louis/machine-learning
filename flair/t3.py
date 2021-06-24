from flair.embeddings import WordEmbeddings
from flair.data import Sentence

# classic word embeddings
glove_embedding = WordEmbeddings('glove')
sentence = Sentence('The grass is green .')

glove_embedding.embed(sentence)
for token in sentence:
    print(token)
    print(token.embedding)

# Flair Embeddings
from flair.embeddings import FlairEmbeddings
flair_embedding_forward = FlairEmbeddings('news-forward')
sentence = Sentence('The grass is green .')
flair_embedding_forward.embed(sentence)
print()
for token in sentence:
    print(token)
    print(token.embedding)

# init forward embedding for German
flair_embedding_forward = FlairEmbeddings('de-forward')
flair_embedding_backward = FlairEmbeddings('de-backward')


# Stacked Embeddings
from flair.embeddings import StackedEmbeddings

glove_embedding = WordEmbeddings('glove')
flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')
# combine
stacked_embeddings = StackedEmbeddings([
                                        glove_embedding,
                                        flair_embedding_forward,
                                        flair_embedding_backward,
                                       ])
sentence = Sentence('The grass is green .')
stacked_embeddings.embed(sentence)
print()
for token in sentence:
    print(token)
    print(token.embedding)