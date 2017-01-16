from sklearn.cluster import KMeans
import time
import nltk.data
import logging
from gensim.models import Word2Vec
import pandas as pd
import os
from keras.layers import Embedding

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, MaxPooling1D, Input
from keras.utils import np_utils


MAX_SEQUENCE_LENGTH=200
MAX_NB_WORDS=88500
nb_validation_samples=100
EMBEDDING_DIM=300



texts = []  # list of text samples
labels = []  # list of label ids


train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )

# limit the number of reviews to 10 for the purpose of testing. For full training, remove the break on i
#i=0
for review in train["review"]:
    texts.append(review)
    #i=i+1
    #if i==10:
    #    break
#i=0
for sentiment in train["sentiment"]:
    labels.append(sentiment)
    #i = i + 1
    #if i == 10:
    #    break


tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = np_utils.to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]


x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


# laoding the embedded vectors obtained from running word2Vec_av.py
pretrained_model = Word2Vec.load("300features_40minwords_10context")
#word_vectors = pretrained_model.syn0
#index=pretrained_model.index2word


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in pretrained_model.vocab:
        embedding_vector=pretrained_model[word]
        embedding_matrix[i] = embedding_vector


# Put the vectors into Keras embedding layer
embedding_layer = Embedding(len(word_index) + 1 ,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(2)(x)  # global max pooling
#x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=2, batch_size=128)

model.save_weights('TEST_CONV_my_trained_weights.h5')





