## Document Embeddings

# Document Pool Embeddings
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, DocumentRNNEmbeddings
from flair.data import Sentence

glove_embedding = WordEmbeddings('glove')
document_embeddings = DocumentPoolEmbeddings([glove_embedding])
sentence = Sentence('The grass is green . And the sky is blue .')
document_embeddings.embed(sentence)
print(sentence.embedding, '\n')

# Document RNN Embeddings
document_embeddings = DocumentRNNEmbeddings([glove_embedding])
sentence = Sentence('The grass is green . And the sky is blue .')
document_embeddings.embed(sentence)
print(sentence.get_embedding(), '\n')

# TransformerDocumentEmbeddings
from flair.embeddings import TransformerDocumentEmbeddings
embedding = TransformerDocumentEmbeddings('bert-base-uncased')
sentence = Sentence('The grass is green .')
embedding.embed(sentence)
print(sentence.get_embedding())

# SentenceTransformerDocumentEmbeddings
from flair.embeddings import SentenceTransformerDocumentEmbeddings
embedding = SentenceTransformerDocumentEmbeddings('bert-base-nli-mean-tokens')
sentence = Sentence('The grass is green .')
embedding.embed(sentence)
print(sentence.get_embedding())

# ATTENTION! The library "sentence-transformers" is not installed!
# To use Sentence Transformers, please first install with "pip install sentence-transformers"
