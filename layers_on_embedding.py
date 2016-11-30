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


MAX_SEQUENCE_LENGTH=10
MAX_NB_WORDS=1000
nb_validation_samples=3



texts = []  # list of text samples
labels = []  # list of label ids


train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )

# limit the number of reviews to 10 for the purpose of testing. For full training, remove the break on i
i=0
for review in train["review"]:
    texts.append(review)
    i=i+1
    if i==10:
        break
i=0
for sentiment in train["sentiment"]:
    labels.append(sentiment)
    i = i + 1
    if i == 10:
        break


tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
train_generator = tokenizer.texts_to_sequences_generator(texts[:-nb_validation_samples])
val_generator = tokenizer.texts_to_sequences_generator(texts[-nb_validation_samples:])

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
word_vectors = pretrained_model.syn0
index=pretrained_model.index2word
num_of_words=len(index)


# Put the vectors into Keras embedding layer
embedding_layer = Embedding(num_of_words ,
                            300,
                            weights=[word_vectors],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# Following is an attempt to save the output of embedding layer on input of train and validation data to speed up
# the training of the top layer for fine tuning
base_model = Sequential()
base_model.add(embedding_layer)



embedding_train_output=base_model.predict(x_train)
embedding_val_output=base_model.predict(x_val)

np.save(open('embedding_train_output.npy','w'),embedding_train_output )
np.save(open('embedding_val_output.npy','w'),embedding_val_output)

train_data = np.load(open('embedding_train_output.npy'))
val_data = np.load(open('embedding_val_output.npy'))


# Feed the saved output of embedding layer into top layers and train the top layers
# This top layers are the same as the one provided on HW3
# The commented out part at the bottom is Conv1d layers and need to be fixed.

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax',name='my_output'))

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

print (model.summary())

print 'Trainable weights'
print model.trainable_weights

model.fit(train_data,y_train,nb_epoch=10,batch_size=32,verbose=2,validation_data=(val_data,y_val))
model.save_weights('TEST_my_trained_weights.h5')


model.evaluate()


'''
sequence_input = Flatten(input_shape=train_data.shape[1:])
#embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(sequence_input)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
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

model.evaluate()
'''


