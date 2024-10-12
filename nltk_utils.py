import nltk
#nltk.download('punkt')      #   you can comment this after your first run
#nltk.download('wordnet')
#nltk.download('punkt_tab')  
#nltk.download('omw-1.4')
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag

#a = "how long does it take?"
#a = tokenize(a)
#print(a)

#words = ["Organize","organizes","organizing"]
#stemmed_words = [stem(w) for w in words]
#print(stemmed_words)

#sentence = ["hello", "how", "are", "you"]
#words = ["hi", "hello", "I", "you", "bye", "thank","cool"]
#bag = bag_of_words(sentence, words)
#print(bag)