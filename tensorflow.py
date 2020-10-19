# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 13:19:43 2020

@author: joeda
"""

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import regex as re
from nltk.stem import WordNetLemmatizer 

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

lemmatizer = WordNetLemmatizer() 
mot_interdit=np.loadtxt('1-1000.txt',dtype=str)[0:600]
#clf = svm.SVC(kernel='poly')
svc=svm.SVC()
clf = GridSearchCV(svc, param_grid)


class LinearModel:
    def __init__(self,train_inputs,train_labels):
        interdiction=[]
        for interdit in mot_interdit :
            interdiction.append(lemmatizer.lemmatize(interdit))
        self.temp_word=[]
        for abstract in train_inputs[:,1] :
                 abstract=re.sub(r"[^a-zA-Z0-9]+", ' ', abstract)
                 abstract=abstract.replace("0123456789", ' ')
                 temp=[]
                 for word in abstract.lower().split() :
                     word=lemmatizer.lemmatize(word)
                     if word not in interdiction :
                         if word not in ['0','1','2','3','4','5','6','7','8','9'] :
                             if word not in ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'] :
                                    temp+=[word]
               
                    
                 self.temp_word+=temp
        self.dict=np.unique(self.temp_word)
        self.n_dict=len(self.dict)
        self.n_classes = len(np.unique(train_labels))
        self.classes=np.unique(train_labels)
        self.train_inputs=train_inputs
        self.train_dict=[]
        
        nn=np.zeros((self.n_classes))
        for (ex,c) in enumerate(self.classes) :
            nn[ex]+=len(np.where(train_inputs[:,2]==c)[0])
        self.n_byclasses=nn
        
        self.w=np.random.random((15,15))
        self.b=np.random.random(size=(15,15))
        for i in range(0,len(self.classes)) :
            self.train_dict.append({})
            
        self.A=np.random.random(size=(self.n_dict,int(self.n_dict/20)))
    def word2vec(self,abstract) :
        abstract=re.sub(r"[^a-zA-Z0-9]+", ' ', abstract)
        abstract=abstract.replace("0123456789", ' ')
        words=abstract.lower().split()
       
      
        vecteur=np.zeros(self.n_dict)
        for word in words :
            word=lemmatizer.lemmatize(word)
            if word in self.dict :
                vecteur[np.where(self.dict==word)[0]]+=1/len(words)
        
        #vecteur=np.matmul(np.transpose(self.A),vecteur[:,np.newaxis])
        #vecteur=np.array(np.fft.ifft(vecteur),dtype=np.float32)
        maximum=np.max(vecteur) # normalisation
        vecteur/=maximum
        #vecteur=np.matmul(np.transpose(self.A),vecteur)
        #vecteur=np.array(np.fft.ifft(vecteur),dtype=np.float32)
        return vecteur
    
    def answer2vec(self,y) :
        a=np.where(self.classes==y)[0]
        
        return a
train=pd.read_csv('data/train.csv')


df1=train.values[0:7000,:]
df2=train.values[7000:7500,:]
f=LinearModel(df1,df1[:,2])

X=[]
y=[]
for (ex,abstract) in enumerate(df1[:,1]):
    print(ex)
    X.append(f.word2vec(abstract))
    y.append(f.answer2vec(df1[ex,2]))
X=np.array(X)
y=np.array(y)

X2=[]
for (ex,abstract) in enumerate(df2[:,1]):
    print(ex)
    X2.append(f.word2vec(abstract))
  
X2=np.array(X2)

"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(1178,1)),
  tf.keras.layers.Dense(128, activation='softmax'),

  tf.keras.layers.Dense(32, activation='softmax'),
  tf.keras.layers.Dense(32, activation='softmax'),
  tf.keras.layers.Dense(15, activation='softmax')
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
"""
import pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
# load dataset

# encode class values as integers
y2=df2[:,2]
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
encoder = LabelEncoder()
encoder.fit(y2)
encoded_Y2 = encoder.transform(y2)
dummy_y2 = tf.keras.utils.to_categorical(encoded_Y2)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = tf.keras.utils.to_categorical(encoded_Y)


#%%

# define baseline model

def baseline_model(taille1):
	# create model
	model = Sequential()
	model.add(Dense(taille1, input_dim=24792, activation='sigmoid'))
	#model.add(Dense(15, activation='sigmoid'))
	model.add(Dense(15, activation='sigmoid'))
	#
	#model.add(Dense(15, activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	return model
 

#estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=5, verbose=2,shuffle=True,use_multiprocessing=True,)
#kfold = KFold(n_splits=2, shuffle=True)
#results = cross_val_score(estimator, X, dummy_y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#%%
Erreur=100
for taille1 in range(5,15,1) :
       
    model=baseline_model(taille1)
    model.fit(X,dummy_y, epochs=100, batch_size=5, verbose=2,shuffle=True,use_multiprocessing=True,validation_data=(X2, dummy_y2))
    
    y=model.predict(X2)
    yy=[]
    for i in y :
        a=np.argmax(i)
        yy.append(f.classes[a])
        
    yy=np.array(yy)
    a=np.array(df2[:,2],dtype='<U17')
    err=len(np.where(a!=yy)[0])/len(df2[:,2])
    print("erreur",err)
    if err<Erreur :
        Erreur=err
        taille_final=taille1
    
#%%
test=pd.read_csv('data/test.csv')
df1=test.values
X3=[]
for (ex,abstract) in enumerate(df1[:,1]):
    print(ex)
    X3.append(f.word2vec(abstract))
  
X3=np.array(X3)

y=model.predict(X3)
yy=[]
for i in y :
    a=np.argmax(i)
    yy.append(f.classes[a])
    

df = pd.DataFrame(yy, columns=["Category"])
df.to_csv('solution.csv')
    