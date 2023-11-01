import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

import os
import re
import pickle
import random
import sys

# Отключаем вывод ошибок
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Количество эпох
EPOCHS = 2

raw_text = open('wonderland.txt', 'r', encoding='utf-8').read()

raw_text = re.sub('[^\nA-Za-z0-9 ,.:;?!-]+', '', raw_text)

raw_text = raw_text.lower()

number_chars = len(raw_text)
print('Длина текста:', number_chars)

chars = sorted(list(set(raw_text)))
n_vocab = len(chars)
print('Количество найденных уникальных символов:', n_vocab)

char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

seq_length = 100

inputs = []
outputs = []

for i in range(0, number_chars - seq_length, 1):
    inputs.append(raw_text[i:i + seq_length])
    outputs.append(raw_text[i + seq_length])

n_sequences = len(inputs)
print('Всего последовательностей:', n_sequences)

indeces = list(range(len(inputs)))
random.shuffle(indeces)

inputs = [inputs[x] for x in indeces]
outputs = [outputs[x] for x in indeces]

X = np.zeros((n_sequences, seq_length, n_vocab), dtype=bool)
y = np.zeros((n_sequences, n_vocab), dtype=bool)

for i, example in enumerate(inputs):
    for t, char in enumerate(example):
        X[i, t, char_to_int[char]] = 1
    y[i, char_to_int[outputs[i]]] = 1

model = Sequential()
model.add(LSTM(128, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.50))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate(sentence, prediction_length=50, diversity=0.35):
    print('Семя генератора:', "\n", sentence, "\n")

    generated = sentence
    sys.stdout.write(generated)

    for i in range(prediction_length):

        x = np.zeros((1, X.shape[1], X.shape[2]))
        for t, char in enumerate(sentence):
            x[0, t, char_to_int[char]] = 1.

        preds = model.predict(x, verbose=0)[0]

        next_index = sample(preds, diversity)
        next_char = int_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()


filepath = '-basic_LSTM.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

prediction_length = 50

for iteration in range(EPOCHS):
    print('Эпоха:', iteration + 1, '/', EPOCHS)
    model.fit(X, y, validation_split=0.2, batch_size=256, epochs=1, callbacks=callbacks_list)

    start_index = random.randint(0, len(raw_text) - seq_length - 1)
    seed = raw_text[start_index: start_index + seq_length]

    print('Полученное значение генератора:', "\n", seed, "\n")
    generate(seed, prediction_length)

pickle_file = '-basic_data.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'X': X,
        'y': y,
        'int_to_char': int_to_char,
        'char_to_int': char_to_int,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as exception:
    print('Unable to save data to', pickle_file, ':', exception)
    raise

stat_info = os.stat(pickle_file)
print('\n Данные сохранены в файле:', pickle_file)
