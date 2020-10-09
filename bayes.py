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
            
            
        
    def train(self):
        
        for (ex,c) in enumerate(self.classes) :
            word_bank=[]
            for abstract in self.train_inputs[np.where(self.train_inputs[:,2]==c)][:,1] :
                word_bank+=abstract.split()
            for word in word_bank:
                
                 if (word in self.train_dict[ex]) :
                     self.train_dict[ex][word]+=1/len(word_bank)
                 else :
                     self.train_dict[ex].update([(word,1/len(word_bank))])
            
           
      
       
        

    def compute_predictions(self, test_data):
            y=[]
            for (i,abstract) in enumerate(test_data[:,1]) :
                abstract=abstract.split()
                answers=np.zeros(self.n_classes)
                
                for word in abstract:
                    for ex in range(0,(self.n_classes)) :
                        if word in self.train_dict[ex] :
                            answers[ex]+=self.train_dict[ex][word]
                        
              
                y.append(self.classes[np.argmax(answers)])
                    

            return y
            
def error_rate(train,val_data):
    f=Bayes_naif(train,train[:,2])
    f.train()
   # print(f.train_dict)
    y=f.compute_predictions(val_data)
    errors=len(np.where(y!=val_data[:,2])[0])/len(val_data[:,2])
    return errors,y

    
filter1=np.zeros(7500,dtype=np.int32)
filter1[np.random.random_integers(0,7499,5000)]+=1
filter2=np.abs(filter1-1)


df1=train.values[0:5000,:]
df2=train.values[5000:7500,:]
erreur,y=error_rate(df1,df2)