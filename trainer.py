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
#괄호에 있는 거랑 줄바꿈, 다중 공백을 그냥 공백으로 인식하게 함

jsontext = re.sub(r'<.*>', '', jsontext)
jsontext = re.sub(r'\n', ' ', jsontext)
jsontext = re.sub(r' +', ' ', jsontext)

print('corpus length : ', len(jsontext))


#일단 기본 개념 자체는 chars의 len만큼 zeros를 만든 뒤에 특정한 글자의 인덱스 값에 1을 넣어서 벡터화 시켜서 각각의 글자의 고유성을 부여해서 그 뒤에 무슨 글자가 나오게 할 지 학습시키는거임

chars = sorted(list(set(jsontext)))
print('total chars : ', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 10
step = 1 
sentences = []
next_chars = []

for i in range(0, len(jsontext) - maxlen, step):
    sentences.append(jsontext[i: i + maxlen])#input 40글자씩 
    next_chars.append(jsontext[i + maxlen])#output 1글자씩
print('sequences : ', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)#false로 가득찬 배열들 x는 3차원 y는 2차원
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)


#인덱싱일 확률이 가장 높고 x[int, int, dict의 int]를 불러오는 거니까 x가 false로 배열되어 있는 와중에 저 위치에 True를 꽂아 넣는거지 =1 이 왜 있는지 생각해야해 

#여기가 원핫인코딩인거고 그 글자에 해당하는 인덱스에 1을 넣는 부분임 5:53
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print('Building model')
model = Sequential()
model.add(LSTM(1024, input_shape=(maxlen, len(chars))))#10, 1453의 행 그러니까 문장이 10개의 길이이고 글자가 1453개잖아 원핫인코딩이니까 글자마다 한 개 인거임
model.add(Dense(len(chars), activation='softmax'))#1453개의 노드
#softmax는 활성화함수인 것 같은데 모든 합이 1이 되도록 만들어주는 활성화 함수 / 1608개의 Output
#ㄴ 분류하고자 하는 클래스가 k개일 때, k차원의 벡터를 입력받아서 모든 벡터 원소의 값을 0과 1사이의 값으로 값을 변경하여 다시 k차원의 벡터를 리턴
print(model.summary())


#보통 멀티 클래스에서 하나의 클래스를 구분할 때 softmax랑 ce 조합을 많이 사용한다고 함
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001))
#클래스가 2개일 때 시그모이드와 소맥은 같은 식이 됨
#https://ebbnflow.tistory.com/135



#커스텀 콜백임
#루프에 빠질 수도 있기 때문에 확률 조작
def sample(preds, temperature=1.0): 
    # helper function to sample an index from a probability array
    #그냥 소프트맥스 구현한 부분인듯
    preds = np.asarray(preds).astype('float64')#입력데이터를 ndarray로 변환하는데 이미 ndarray면 새로 생성은 안함 그냥 nd변환기
    preds = np.log(preds) / temperature#preds를 log에 넣고 0.5로 나눔
    #여기서부터
    exp_preds = np.exp(preds)#e^x로 바꿔줌
    preds = exp_preds / np.sum(exp_preds)#시그마
    #여기는 소프트맥스 식이랑 일치함
    probas = np.random.multinomial(1, preds, 1)#
    
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    print('\nEpoch : ' + str(epoch))

    start_index = random.randint(0, len(jsontext) - maxlen - 1)#랜덤으로 40글자를 뽑고
#     for diversity in [0.2, 0.5, 1.0, 1.2]:
#         print('----- diversity:', diversity)

    generated = ''
    sentence = jsontext[start_index: start_index + maxlen]#모델에게 40글자 힌트를 주고 그 뒷부분의 이야기를 지어내게 함
    #sentence = ''
    generated += sentence
    print('Presented word : "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(200):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]#데이터를 인풋으로 받아 예측함
        next_index = sample(preds, 0.5)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char#문장 생성

        sys.stdout.write(next_char)#print
        sys.stdout.flush()
    print()
    

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)#람다콜백에 등록하고 변수에 넣음

model.fit(x, y, batch_size=128, epochs=200, callbacks=[print_callback])#60번 반복할거고
'''
model_json = model.to_json()
with open("lstmwritter.json", "w") as json_file : 
    json_file.write(model_json)

model.save('lstm_writter.h5')
'''