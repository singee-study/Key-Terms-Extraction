# find vocabulary by "voc"

import numpy as np
from nltk import word_tokenize

text = str(input())

def bag_of_words(voc, text):
    array = np.zeros((len(voc))) # len(voc) stands for the size of the vocabulary

    # code here below


    return array 
    # return a non-binary array
    # it may look like this: [0., 2., 4., 0.]

print(bag_of_words(voc, text))