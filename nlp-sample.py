# install nltk - nlp toolkit
# sudo pip3 install nltk
# go to python console
# $ python3
# run the below commands to download all the libraries
# >> import nltk
# >> nltk.download()
# choose 'all'

from nltk.corpus import brown
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
# test
print(brown.words())

tokenizer = RegexpTokenizer(r'\w+')
# remove all type of punctuations from given text
# tokenization of words
filterdText=tokenizer.tokenize('Hello there, Python is a very powerful & amazing library & I love working with it.')
print(filterdText)

# tokenization of sentences
text = "God is Great! I am still having fun."
print(sent_tokenize(text))


