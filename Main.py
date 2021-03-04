from keras.callbacks import LambdaCallback
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop
from gensim.summarization.summarizer import summarize
import random, sys, io, re, os, json
import numpy as np
import json_take as j
import crawler as c

crowl_link_result = []
crowl_text_result = []

file = open('jsontext.txt', 'r', -1, 'utf-8')
jsontext = file.read()
file.close()

print('original Source length : ' + str(len(jsontext)))
jsontext = re.sub(r'<.*>', '', jsontext)
jsontext = re.sub(r'\n', ' ', jsontext)
jsontext = re.sub(r' +', ' ', jsontext)

print('corpus length : ', len(jsontext))

file = open('chars.txt', 'r', -1, 'utf-8')
chars = list(file.read())
file.close()

print('total chars : ', len(chars))

user_word = input('Type your Search Word : ')
serching_time = input('How many will you take posts? : ')
serching_type = input('similarity?(Type sim) or latest?(Type date) : ')

crowl_link_result = c.crowl_link(user_word, serching_time, serching_type)
crowl_text_result = c.crowl_text(crowl_link_result)


char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 10
step = 1
sentences = []
next_chars = []

for i in range(0, len(jsontext) - maxlen, step):
    sentences.append(jsontext[i: i + maxlen])#input
    next_chars.append(jsontext[i + maxlen])#output 
print('sequences : ', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)


for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print('loading model')
model = load_model('lstm_writter.h5')

def sample(preds, temperature=1.0): 
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    print('\nEpoch : ' + str(epoch))

    start_index = random.randint(0, len(crowl_text_result) - maxlen - 1)
    generated = ''
    crowl_gene_text = crowl_text_result[start_index: start_index + 10]
    crowl_gene_text = ''.join(crowl_gene_text)
    sentence = crowl_gene_text
    generated += sentence
    print('Presented word : "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(1000):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, 0.5)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)#print
        sys.stdout.flush()
    print()
    

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])
