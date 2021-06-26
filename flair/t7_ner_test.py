from flair.data import Corpus
from flair.datasets.sequence_labeling import CONLL_03
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.embeddings import FlairEmbeddings
from flair.trainers import ModelTrainer
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter


# load the model you trained
model = SequenceTagger.load('resources/ner/ner-training/final-model.pt')
# won't work in local computer

# create example sentence
text = 'Under Minnesota law, Chauvin will have to serve two-thirds of his sentence, or 15 years -- and he will be eligible for supervised release for the remaining seven and a half years. The sentence exceeds the Minnesota sentencing guideline range of 10 years and eight months to 15 years for the crime. Floyd\'s death sparked massive protests across the nation over police brutality. Floyd\'s final moments, captured on searing cell phone footage by a 17-year-old, illustrated in clear visuals what Black Americans have long said about how the criminal justice system treats Black people. Floyd\'s death set off mass protests across the globe as well as incidents of looting and unrest.'

splitter = SegtokSentenceSplitter()
sentences = splitter.split(text)

# predict tags and print
model.predict(sentences)

for sentence in sentences:
    print(sentence.to_tagged_string())
