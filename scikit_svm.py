# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 09:09:55 2020

@author: joeda
"""

from sklearn import svm,preprocessing
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import regex as re
from nltk.stem import WordNetLemmatizer 

param_grid = [
  
  {'C':  [100000,1000000], 'gamma': [0.00001,0.0000001], 'kernel': ['rbf']},

 ]

lemmatizer = WordNetLemmatizer() 
mot_interdit=np.loadtxt('1-1000.txt',dtype=str)[0:150]
#clf = svm.SVC()
svc=svm.SVC()
clf = GridSearchCV(svc, param_grid,n_jobs=8)


class LinearModel:
    def __init__(self,train_inputs,train_labels):
        self.temp_word=[]
        for abstract in train_inputs[:,1] :
                 abstract=re.sub(r"[^a-zA-Z]+", ' ', abstract)
                 
                 temp=[]
                 for word in abstract.lower().split() :
                     word=lemmatizer.lemmatize(word)
                     temp+=[word]
                     
                 self.temp_word+=temp
                 for interdit in mot_interdit :
                    try :
                        temp.remove(interdit)
                    except : 
                        pass
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
        
        
        self.A=np.random.random(size=(self.n_dict,int(self.n_dict/10)))
        for i in range(0,len(self.classes)) :
            self.train_dict.append({})
            
            
    def word2vec(self,abstract) :
      
        abstract=re.sub(r"[^a-zA-Z]+", ' ', abstract)
        abstract=abstract.replace("0123456789", ' ')
        words=abstract.lower().split()
        
        for interdit in mot_interdit :
            try :
                words.remove(interdit)
            except : 
                pass
       
      
        vecteur=np.zeros(self.n_dict)
        for word in words :
            word=lemmatizer.lemmatize(word)
            if word in self.dict :
                vecteur[np.where(self.dict==word)[0]]+=1
        
      
        vecteur=np.matmul(np.transpose(self.A),vecteur[:,np.newaxis])
        maximum=np.max(vecteur) # normalisation
        vecteur/=maximum
        return vecteur


train=pd.read_csv('data/train.csv')


df1=train.values[0:6000,:]
df2=train.values[6000:7500,:]
f=LinearModel(df1,df1[:,2])

X=[]
y=[]
for (ex,abstract) in enumerate(df1[:,1]):
    print(ex)
    X.append(f.word2vec(abstract))
    y.append(df1[ex,2])
X=np.array(X)[:,:,0]

clf.fit(X, y)
best_param=clf.best_params_
answer=[]
for (ex,abstract) in enumerate(df2[:,1]):
    print(ex)
    answer.append(clf.predict([f.word2vec(abstract)[:,0]])[0])
    
print('erreur : ',len(np.where(answer!=df2[:,2])[0])/len(df2[:,2]))