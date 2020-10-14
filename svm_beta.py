# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 19:09:04 2020

@author: Jonathan Beaulieu-Emond
"""
import numpy as np
import pandas as pd

train=pd.read_csv('data/train.csv')
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
       
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)



def f(x,y,w,lambd) :
    n=len(x)
    return 1/n*np.sum(np.max([0,1-y*(w*x-b)]))+lambd*np.linalg.norm(w)**2

class Bayes_naif:
    def __init__(self,train_inputs,train_labels):
        
        self.n_classes = len(np.unique(train_labels))
        self.classes=np.unique(train_labels)
        self.train_inputs=train_inputs
        self.train_dict=[]
        nn=np.zeros((self.n_classes))
        for (ex,c) in enumerate(self.classes) :
            nn[ex]+=len(np.where(train_inputs[:,2]==c)[0])
        self.n_byclasses=nn
        for i in range(0,len(self.classes)) :
            self.train_dict.append({})
            
            
    def word2vec(self) :
        temp_word=[]
        for abstract in self.train_inputs[:,1] :
                temp_word+=abstract.split()
        nbMot=len(temp_word.unique())
        
        dict_vect=np.zeros((nbMot,nbMot))
        for (ex,word) in temp_word.unique() :
            dict_vect[ex][ex]=1
        
        return dict_vect,nbMot,temp_word.unique()
            
    def train(self,w,lambd):
        
       
        
           
      
       
        

    def compute_predictions(self, test_data,hyper_param):
         
            return y
            
def error_rate(train,val_data):
    erreur_min=1
    
    for hyper_param in np.arange(0,100,1) :
        f=Bayes_naif(train,train[:,2])
        f.train()
       # print(f.train_dict)
        y=f.compute_predictions(val_data,hyper_param)
        errors=len(np.where(y!=val_data[:,2])[0])/len(val_data[:,2])
        if errors<erreur_min:
            yy=y
            erreur_min=errors
            hype=hyper_param
    return erreur_min,yy,hype

    


df1=train.values[0:6000,:]
df2=train.values[6000:7500,:]
#erreur,y,hype=error_rate(df1,df2)
test=pd.read_csv('data/test.csv')
#%%
f=Bayes_naif(train.values,train.values[:,2])
f.train()
# print(f.train_dict)
y=f.compute_predictions(test.values,8)

df = pd.DataFrame(y, columns=["Category"])
df.to_csv('solution.csv')
