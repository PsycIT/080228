#!/usr/bin/env python

# -*- coding: utf-8 -*-
# 코드 내부에 한글을 사용가능 하게 해주는 부분입니다.

# 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 필요한 라이브러리를 불러옵니다.
import numpy as np
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다.
np.random.seed(3)
tf.random.set_seed(3)

# 준비된 수술 환자 데이터를 불러들입니다.
Data_set = np.loadtxt("../../dataset/ThoraricSurgery.csv", delimiter=",")

# 환자의 기록과 수술 결과를 X와 Y로 구분하여 저장합니다.
X = Data_set[:,0:17]
Y = Data_set[:,17]

# 딥러닝 구조를 결정합니다(모델을 설정하고 실행하는 부분입니다).
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝을 실행합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X, Y, epochs=100, batch_size=10)


x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

mx = np.mean(x)
my = np.mean(y)

divisor = sum([(i - mx)**2 for i in x])


def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)

    return d


dividend = top(x, mx, y, my)

print("분모:", divisor)
print("분자:", dividend)


a = dividend / divisor
b = my - (mx*a)

print("기울기 a =", a)
print("y 절편 b =", b)


fake_a_b = [3, 76]

data = [[2, 81], [4, 93], [6, 91], [8, 97]]

x = [i[0] for i in data]
y = [i[1] for i in data]


def predict(x):
    return fake_a_b[0]*x + fake_a_b[1]

def mse(y, y_hat):
    return ((y-y_hat)**2).mean()

def mse_val(y, predict_result):
    return mse(np. array(y), np.array(predict_result))


predict_result = []

for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부시간:%.f, 실제 점수:%.f, 예측 점수:%.f" % (x[i], y[i], predict(x[i])))


print("mse 최종값: " + str(mse_val(y, predict_result)))