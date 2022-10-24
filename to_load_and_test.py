from tensorflow.keras.models import load_model 
import os
import librosa
import numpy as np
paths=[]
langs=os.listdir('test/')
actlang=os.listdir('train/')
print(langs)
def features_extractor(path):
    raw,sam_rate=librosa.load(path)
    mfccs=librosa.feature.mfcc(y=raw,sr=sam_rate,n_mfcc=40)
    extractedfeatures=np.mean(mfccs.T,axis=0)
    return extractedfeatures
model=load_model('accent_rec')
pred={}
for p in langs:
    x_predict=features_extractor('test/'+str(p))
    x_predict=x_predict.reshape(1,40)
    y_predict=model.predict(x_predict)
    pred[str(p.replace('.mp3',''))]=(str(actlang[(np.where(y_predict==y_predict.max())[0][0])]))
fp=open('ground_truth.json','w')
fp.write(str(pred))
fp.close()