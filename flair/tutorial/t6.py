# The Corpus Object

import flair.datasets
corpus = flair.datasets.UD_ENGLISH()
# print the number of Sentences in the train split
print(len(corpus.train))

# print the number of Sentences in the test split
print(len(corpus.test))

# print the number of Sentences in the dev split
print(len(corpus.dev))

# access the first sentence in the training split
print(corpus.test[0])
print(corpus.test[0].to_tagged_string('pos'))  # pos tag


# downsample
corpus = flair.datasets.UD_ENGLISH()
downsampled_corpus = flair.datasets.UD_ENGLISH().downsample(0.1)
print("--- 1 Original ---")
print(corpus)

print("--- 2 Downsampled ---")
print(downsampled_corpus)


# make_label_dictionary
corpus = flair.datasets.UD_ENGLISH()
print(corpus.make_label_dictionary('upos'))

corpus = flair.datasets.CONLL_03_DUTCH()
print(corpus.make_label_dictionary('ner'))

corpus = flair.datasets.TREC_6()
print(corpus.make_label_dictionary(), '\n\n')


# obtain_statistics
corpus = flair.datasets.TREC_6()
stats = corpus.obtain_statistics()
print(stats)

# The MultiCorpus Object
english_corpus = flair.datasets.UD_ENGLISH()
german_corpus = flair.datasets.UD_GERMAN()
dutch_corpus = flair.datasets.UD_DUTCH()

from flair.data import MultiCorpus
multi_corpus = MultiCorpus([english_corpus, german_corpus, dutch_corpus])

