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
#     for diversity in [0.2, 0.5, 1.0, 1.2]:
#         print('----- diversity:', diversity)

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


'''

Epoch : 0
Presented word : "봄 기운을 담은 봄"
봄 기운을 담은 봄곳 러나 지만 있다. 부근 남을c:\Users\elppa\Desktop\고등\학교활동\3학년\뚝메\LSTM_writter\asdfsadf.py:68: RuntimeWarning: divide by zero encountered in log
  preds = np.log(preds) / temperature
 김하사 역시 앞에서 엄물난 호 나처나 우내는 북한군이 남한선의 강장으로 작정오?” “대구가 여인있은 대원 두 명이 장바닥는 발 앞소리는 분이에 내려오시 기가의 군을 받는 것은 고향 남포 사람이면 누구나 다 아는 일이다. 그는 문득 방금 전에 만
난 또 한 명의 고향 사람 최가를 머릿속에 떠올린다. 최가도 서울까지 올라왔을 정도라면 그보다 더 열성 당원인 동렬의 형이 서울에 올라오지 말라는 법은 없다. 아니 어쩌면 최가와 동렬의 형 두 사람이 어떤 조직의 조직원으로 함께 서울로 올라왔는
지도 모를 일이다. 차가 피난민들의 행렬을 거슬러 계속 느릿느릿 시 중심가로 전진한다. 시가지 왼쪽으로는 화재라도 났는지 진회색의 짙은 연기가 파란 하늘로 무럭무럭 치솟고 있다. 난민들은 이제 사병들을 향해 막무가내로 몸을 부딪쳐 저지선을 
돌파하려 하고 있다. “가자우요!” 귀익은 음성이다. 꿈에도 잊지 못할 최가의 그 탁하고 높은 음성이다. 최가에게 서울을 인간적 떨어들 중익은 우동무도 이 끝내 김하사가 있을 때 이도다. “거의 동무가 말씀이시기요?” “아마 나실는 오래 전시에 지금
까지 분지 못다 전디로 한 온 모양이군?” “킬머는 아마 내일쯤 배편으로 도착할 거요. 한데 참 서울에 있는 당신의 신문사도 내려왔소?” “내려오긴 했지만 일은 아직 못 하고 있소.” 오랫동안 잡고 있던 손들을 두 사람이 동시에 놓는다. 로이가 경민의
 등뒤에 서 있는 강윤정 쪽으로 다가간다. 윤정은 화사한 분홍색 블라우스에 치마는 몸에 꽉 끼는 연청색의 타이트 스커트를 입고 있다. 여름철 뜨거운 한낮의 햇볕 속에 서 있어서 그녀는 눈이 부신 듯 이마를 잔뜩 찌푸리고 있다. 경민은 그러나 로이
의 뜻을 무시하고 잠시 사이를 두었다가 가장 궁금한 것을 묻는다. “당신은 참 어디에 머물고 있소?” “군에서 임시로 징발해준 어떤 커다란 서양식 저택이오. 하지만 그곳은 군사 기밀상 당신에게 소재지를 알려줄 수가 없소.” “줄곧 그 비밀 장소에 머
물 작정이오?” “아니오, 본사와 연락

'''