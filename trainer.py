from keras.callbacks import LambdaCallback
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random, sys, io, re, os, json
import json_take as j

jsontext = j.jsonfile()

print('original Source length : ' + str(len(jsontext)))

jsontext = re.sub(r'<.*>', '', jsontext)
jsontext = re.sub(r'\n', ' ', jsontext)
jsontext = re.sub(r' +', ' ', jsontext)

print('corpus length : ', len(jsontext))


chars = sorted(list(set(jsontext)))
print('total chars : ', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 10
step = 1 
sentences = []
next_chars = []

for i in range(0, len(jsontext) - maxlen, step):
    sentences.append(jsontext[i: i + maxlen]) 
    next_chars.append(jsontext[i + maxlen])
print('sequences : ', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)


for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print('Building model')
model = Sequential()
model.add(LSTM(1024, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001))

def sample(preds, temperature=1.0): 
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    print('\nEpoch : ' + str(epoch))

    start_index = random.randint(0, len(jsontext) - maxlen - 1)

    generated = ''
    sentence = jsontext[start_index: start_index + maxlen]
    #sentence = ''
    generated += sentence
    print('Presented word : "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(200):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, 0.5)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
    

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y, batch_size=128, epochs=200, callbacks=[print_callback])

model_json = model.to_json()
with open("lstmwritter.json", "w") as json_file : 
    json_file.write(model_json)

model.save('lstm_writter.h5')
