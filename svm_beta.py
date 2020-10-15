# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 19:09:04 2020

@author: Jonathan Beaulieu-Emond
"""
import numpy as np
import pandas as pd
import regex as re
from scipy.sparse import identity,csr_matrix
train=pd.read_csv('data/train.csv')
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
       
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)



class Bayes_naif:
    def __init__(self,train_inputs,train_labels):
        self.temp_word=[]
        for abstract in train_inputs[:,1] :
                abstract=re.sub(r"[^a-zA-Z0-9]+", ' ', abstract)
             
                self.temp_word+=abstract.lower().split()
                
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
        
        self.w=identity(self.n_dict)*np.random.random()
        self.b=np.random.random()
        for i in range(0,len(self.classes)) :
            self.train_dict.append({})
            
            
    def word2vec(self,abstract) :
        abstract=re.sub(r"[^a-zA-Z0-9]+", ' ', abstract)
             
        words=abstract.lower().split()
        vecteur=np.zeros(self.n_dict)
        for word in words :
            if word in self.dict :
                vecteur[np.where(self.dict==word)[0]]+=1
        
        return vecteur
    
   
            
    def train(self):
        #le but est de minimiser norme(w)?
        for (ex,abstract) in enumerate(self.train_inputs[:,1]) :
            vecteur=self.word2vec(abstract)
            print(ex)
            
            y=np.round(np.max(csr_matrix.dot(np.transpose(self.w),vecteur[:,np.newaxis])+self.b),0)
          
            C=np.where(self.classes==self.train_inputs[ex,2])[0][0]
            
            delta=(y-C)**2
            
            dw=self.w*delta
            self.w+=dw
            self.b+=delta
            
        
           
      
       
        

    def compute_predictions(self, test_data):
            y=[]
            for abstract in test_data[:,1] :
               vecteur=self.word2vec(abstract)
               results=np.round(np.max(self.w*vecteur+self.b),0)
               y.append(self.classes[int(np.round(results,0))])
            return y
            
def error_rate(train,val_data):
     erreur_min=1
    
   
     f=Bayes_naif(train,train[:,2])
     f.train()
     # print(f.train_dict)
     y=f.compute_predictions(val_data)
     errors=len(np.where(y!=val_data[:,2])[0])/len(val_data[:,2])
     
     return errors,y

    


df1=train.values[0:444,:]
df2=train.values[444:7500,:]
erreur,y=error_rate(df1,df2)
test=pd.read_csv('data/test.csv')
#%%
f=Bayes_naif(train.values,train.values[:,2])
f.train()
# print(f.train_dict)
y=f.compute_predictions(test.values,8)

df = pd.DataFrame(y, columns=["Category"])
df.to_csv('solution.csv')
