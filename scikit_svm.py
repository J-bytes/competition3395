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
import matplotlib.pyplot as plt
from stemming.porter2 import stem 
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
lemmatizer = WordNetLemmatizer() 
mot_interdit=np.loadtxt('1-1000.txt',dtype=str)[0:350]
#clf = svm.SVC(kernel='poly')
#from thundersvm import SVC
svc=svm.SVC()
#svc=SVC
trans_table = {ord(c): None for c in string.punctuation + string.digits}  
from nltk.tokenize import RegexpTokenizer

param1=np.array([0.1,1,2,5,10])
param2=np.array([0.1,0.5,0.8,1,2,5])
param_grid = [
  { 'degree' :param1,'C' :param2,'gamma' :['scale'], 'kernel' : ['poly']},
  #{'C':  param1, 'gamma': param2, 'kernel': ['rbf']},
  #{'C':[0.01,0.1,1,10,100],'kernel' :['linear']}
 ]

lemmatizer = WordNetLemmatizer() 
mot_interdit=np.loadtxt('1-1000.txt',dtype=str)[0:150]
clf = svm.SVC(kernel='linear')
svc=svm.SVC(verbose=5)
clf = GridSearchCV(svc, param_grid,verbose=5,n_jobs=8)

def tokenize(text):
        # my text was unicode so I had to use the unicode-specific translate function. If your documents are strings, you will need to use a different `translate` function here. `Translated` here just does search-replace. See the trans_table: any matching character in the set is replaced with `None`
        tokens = [word for word in nltk.word_tokenize(text) if len(word) > 1] #if len(word) > 1 because I only want to retain words that are at least two characters before stemming, although I can't think of any such words that are not also stopwords
    
        stems = [lemmatizer.lemmatize(re.sub(r"[^a-zA-Z]+", ' ',item.lower())) for item in tokens]
        return stems

class LinearModel:
    def __init__(self,train_inputs,train_labels):
        pass
            
            
    def word2vec(self,abstracts,abstracts2) :
        abstracted=[]
        for abstract in abstracts :
           abstracted.append(abstract)
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
  
        cv=TfidfVectorizer(r'[a-z]',tokenizer=tokenize,min_df=15, max_df=0.2)
        vecteur=cv.fit_transform(abstracted).toarray()
        
        
        abstracted=[]
        for abstract in abstracts2 :
            abstracted.append(abstract)
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


train=pd.read_csv('data/train.csv')
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
df1=train.values[0:7500,:]
df2=train.values[6000:7500,:]
f=LinearModel(df1,df1[:,2])

X=[]
y=[]
for (ex,abstract) in enumerate(df1[:,1]):
   

    y.append(df1[ex,2])
    
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

y= keras.utils.to_categorical(encoded_Y)
y=np.argmax(y,axis=1)
X,X2=f.word2vec(df1[:,1],df2[:,1])
print('Now fitting')
clf.fit(X, y)

# We extract just the scores




def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    #ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel(r'Taux de réussite sur $D_{validation}$', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')

# Calling Method 
plot_grid_search(clf.cv_results_,param1 ,param2 , 'degree','C')


#%%
answer=clf.predict(X2)
answer2=[]
for ans in answer :
    answer2.append(f.classes[ans])
print('erreur : ',len(np.where(answer2!=df2[:,2])[0])/len(df2[:,2]))