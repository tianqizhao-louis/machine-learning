# create a sentence
from flair.data import Sentence

sentence = Sentence('The grass is green.')

print(sentence)

# access the token via token id or index
print(sentence.get_token(4))
print(sentence[3])

print()
# iterate
for token in sentence:
    print(token)

# if you don't want tokenization
untokenized_sentence = Sentence('The grass is green.', use_tokenizer=False)
print('\n', untokenized_sentence)

# different tokenizer
from flair.tokenization import JapaneseTokenizer
tokenizer = JapaneseTokenizer("janome")
japanese_sentence = Sentence("私はベルリンが好き", use_tokenizer=tokenizer)
print(japanese_sentence)

# Using pretokenized sequences
from flair.data import Sentence
sentence = Sentence(['The', 'grass', 'is', 'green', '.'])
print('\n', sentence)

# Adding labels
# adding an NER tag of type 'color' to the word 'green'.
sentence[3].add_tag('ner', 'color')
print('\n', sentence.to_tagged_string())

token = sentence[3]
tag = token.get_tag('ner')
print(f'"{token}" is tagged as "{tag.value}" with confidence score "{tag.score}"')

# add label to sentence
print()
sentence = Sentence('France is the current world cup winner.')
sentence.add_label('topic', 'sports')
print(sentence)

sentence = Sentence('France is the current world cup winner.').add_label('topic', 'sports')  # same
print(sentence)

# assign label multiple times
sentence = Sentence('France is the current world cup winner.')
sentence.add_label('topic', 'sports')
sentence.add_label('topic', 'soccer')
sentence.add_label('language', 'English')
print('\n', sentence)

# Accessing a sentence's labels
for label in sentence.labels:
    print(label)
print(sentence.to_plain_string())
for label in sentence.labels:
    print(f' - classified as "{label.value}" with score {label.score}')

#  interested only in the labels of one layer of annotation
for label in sentence.get_labels('topic'):
    print(label)