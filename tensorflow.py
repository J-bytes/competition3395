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
from stemming.porter2 import stem 
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
lemmatizer = WordNetLemmatizer() 
mot_interdit=np.loadtxt('1-1000.txt',dtype=str)[0:250]
#clf = svm.SVC(kernel='poly')
svc=svm.SVC()
from nltk.tokenize import RegexpTokenizer


class LinearModel:
    def __init__(self,train_inputs,train_labels):
        interdiction=[]
        for interdit in mot_interdit :
            interdiction.append(stem(lemmatizer.lemmatize(interdit)))
        self.temp_word=[]
        for abstract in train_inputs[:,1] :
                 abstract=re.sub(r"[^a-zA-Z]+", ' ', abstract)
                 ''.join([i for i in abstract if not i.isdigit()])
                 temp=[]
                 for word in abstract.lower().split() :
                     word=lemmatizer.lemmatize(word)
                     word=stem(word)
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
    def word2vec(self,abstracts,abstracts2) :
        abstracted=[]
        for abstract in abstracts :
            abstract=''.join([i for i in abstract if not i.isdigit()])
            
            abstract=''.join([" "+lemmatizer.lemmatize(i) for i in abstract.split() ])
           
            abstracted.append(re.sub(r"[^a-zA-Z]+", ' ', abstract))
        """
        words=abstract.lower().split()
       
      
        vecteur=np.zeros(self.n_dict)
        for word in words :
            word=lemmatizer.lemmatize(word)
            word=stem(word)
            if word in self.dict :
                vecteur[np.where(self.dict==word)[0]]+=1/len(words)
        
        #vecteur=np.matmul(np.transpose(self.A),vecteur[:,np.newaxis])
        #vecteur=np.array(np.fft.ifft(vecteur),dtype=np.float32)
        maximum=np.max(vecteur) # normalisation
        vecteur/=maximum
        #vecteur=np.matmul(np.transpose(self.A),vecteur)
        #vecteur=np.array(np.fft.ifft(vecteur),dtype=np.float32)
        """
  
        cv=TfidfVectorizer(r'[a-z]')
        vecteur=cv.fit_transform(abstracted).toarray()
        
        
        abstracted=[]
        for abstract in abstracts2 :
            abstract=''.join([i for i in abstract if not i.isdigit()])
            
            abstract=''.join([" "+lemmatizer.lemmatize(i) for i in abstract.split() ])
           
            abstracted.append(re.sub(r"[^a-zA-Z]+", ' ', abstract))
        """
        words=abstract.lower().split()
       
      
        vecteur=np.zeros(self.n_dict)
        for word in words :
            word=lemmatizer.lemmatize(word)
            word=stem(word)
            if word in self.dict :
                vecteur[np.where(self.dict==word)[0]]+=1/len(words)
        
        #vecteur=np.matmul(np.transpose(self.A),vecteur[:,np.newaxis])
        #vecteur=np.array(np.fft.ifft(vecteur),dtype=np.float32)
        maximum=np.max(vecteur) # normalisation
        vecteur/=maximum
        #vecteur=np.matmul(np.transpose(self.A),vecteur)
        #vecteur=np.array(np.fft.ifft(vecteur),dtype=np.float32)
        """
  
        #cv=TfidfVectorizer(r'[a-z]')
        vecteur2=cv.transform(abstracted).toarray()
        return vecteur,vecteur2
    
    def answer2vec(self,y) :
        a=np.where(self.classes==y)[0]
        
        return a
train=pd.read_csv('data/train.csv')


df1=train.values[0:7200,:]
df2=train.values[7200:7500,:]
f=LinearModel(df1,df1[:,2])
test=pd.read_csv('data/test.csv')
df3=test.values


y=[]
for (ex,abstract) in enumerate(df1[:,1]):
    print(ex)
   
    y.append(f.answer2vec(df1[ex,2]))
X,X3=f.word2vec(train.values[:,1],test.values[:,1])
X=np.array(X)
y=np.array(y)

X2=X[7200:7500]

X2=np.array(X2)
X=X[0:7200]
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

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(15, input_dim=24414, activation='softmax'))
	#model.add(Dense(10, activation='softmax'))
	#model.add(Dense(4, activation='softmax'))
	#model.add(Dense(15, activation='sigmoid'))
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
#epoch=50
Erreur=100

       
model=baseline_model()
model.fit(X,dummy_y, epochs=50, batch_size=15, verbose=2,shuffle=True,use_multiprocessing=True,validation_data=(X2, dummy_y2))

y=model.predict(X2)
yy=[]
for i in y :
    a=np.argmax(i)
    yy.append(f.classes[a])
    
yy=np.array(yy)
a=np.array(df2[:,2],dtype='<U17')
err=len(np.where(a!=yy)[0])/len(df2[:,2])
print("erreur",err)
   
    
#%%

X3=np.array(X3)

y=model.predict(X3)
yy=[]
for i in y :
    a=np.argmax(i)
    yy.append(f.classes[a])
    

df = pd.DataFrame(yy, columns=["Category"])
df.to_csv('solution.csv')
    