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
                abstract=re.sub(r"[^a-zA-Z0-9]+", ' ', abstract)
                word_bank+=abstract.lower().split()
                
            for word in word_bank:
                 word=word.lower()
                 #word=word.isalnum()
                 if (word in self.train_dict[ex]) :
                     self.train_dict[ex][word]+=1/len(word_bank)#pour une raison X il faut multiplier ici...
                 else :                                               #python doit avoir de la misere a gerer les trop petits nombres
                     self.train_dict[ex].update([(word,1/len(word_bank))])
            
           
      
       
        

    def compute_predictions(self, test_data,hyper_param):
            global abstract
            y=[]
            for (i,abstract) in enumerate(test_data[:,1]) :
                abstract=re.sub(r"[^a-zA-Z0-9]+", ' ', abstract)
                abstract=abstract.lower().split()
                answers=np.zeros(self.n_classes)
                abstract=np.array(abstract)
                for word in abstract:
                    word=word.lower()
                    #word=word.isalnum()
                    for ex in range(0,(self.n_classes)) :
                        if answers[ex]==0 and word in self.train_dict[ex]:
                             p=self.train_dict[ex][word]
                             answers[ex]=p
                        if word in self.train_dict[ex] :
                            Pmot=0
                            for ex2 in range(0,(self.n_classes)) :
                                try : 
                                    Pmot+=self.train_dict[ex2][word]
                                except :
                                    Pmot+=0
                                    
                            p=self.train_dict[ex][word]*self.n_byclasses[ex]/Pmot
                            
                            answers[ex]*=(p+0.01)/hyper_param #!!!! overflow
                            #print(answers[ex])
                            
                #print(answers)
                y.append(self.classes[np.argmax(answers)])
                    

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
