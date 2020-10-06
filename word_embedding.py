#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:37:37 2020

@author: siddharthsmac
"""

from tensorflow.keras.preprocessing.text import one_hot

sent = ['the glass of milk',
        'the glass of juice',
        'the cup of tea',
        'I am a good boy',
        'I am a good datascientist',
        'understand the meaning of words',
        'you are very good']

voc_size = 10000

onehot_repr = [one_hot(words, voc_size) for words in sent]

from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

sent_length = 8

embedded_docs = pad_sequences(onehot_repr, padding = 'pre', maxlen = sent_length)

model = Sequential()

model.add(Embedding(voc_size, 10, input_length = sent_length))

model.compile('adam', 'mse')

vect_matrix = model.predict(embedded_docs)

print(vect_matrix)

print(vect_matrix[0])
