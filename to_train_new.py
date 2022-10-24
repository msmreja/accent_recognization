import tensorflow as tf
import os
import numpy as np
import librosa
paths=[]
i=0
langs=os.listdir('train/')
for l in langs:
    tmp=os.listdir('train/'+str(l)+'/')
    for fn in tmp:
        paths.append(['train/'+str(l)+'/'+(fn),i])
    i+=1
def features_extractor(path):
    raw,sam_rate=librosa.load(path)
    mfccs=librosa.feature.mfcc(y=raw,sr=sam_rate,n_mfcc=40)
    extractedfeatures=np.mean(mfccs.T,axis=0)
    return extractedfeatures
#segregating the X_train and y_train
X_train=[]
y_train=[]
for p in paths:
    expath=p[0]
    X_train.append(features_extractor(str(expath)))
    y_train.append(p[1])
X_train=np.array(X_train)
y_train=np.array(y_train)
# print(len(langs))
print(X_train.shape,y_train.shape)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(40,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(len(langs),activation=tf.nn.softmax))
model.compile(optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy']
			)
model.fit(X_train,y_train,epochs=5)
model.save('accent_rec')
# prediciting the native language
# path='arabic5.mp3'
# x_predict=features_extractor(path)
# x_predict=x_predict.reshape(1,40)
# y_predict=model.predict(x_predict)
# print(y_predict)
# print(y_predict.max())