# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 19:09:04 2020

@author: Jonathan Beaulieu-Emond
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
train=pd.read_csv('data/train.csv')
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
       
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)

mot_interdit=np.loadtxt('1-1000.txt',dtype=str)[0:300]
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
                 abstract=re.sub(r"[^a-zA-Z]+", ' ', abstract)
                 ''.join([i for i in abstract if not i.isdigit()])
                 word_bank+=abstract.lower().split()
                
            for word in word_bank:
                
                 #word=word.isalnum()
                 if (word in self.train_dict[ex]) :
                     self.train_dict[ex][word]+=1/len(word_bank)#pour une raison X il faut multiplier ici...
                 else :                                               #python doit avoir de la misere a gerer les trop petits nombres
                     self.train_dict[ex].update([(word,1/len(word_bank))])
        
        #lets remove item which have a really low frequency
        for i in range(1,2) :
            try :
                liste=list(self.train_dict[ex].keys())[list(self.train_dict[ex].values()).index(i/len(word_bank))]
            except :
                liste=[]
            for item in liste :
                self.train_dict[ex].pop(item, None)
      
       
        

    def compute_predictions(self, test_data):
           
            y=[]
            for (i,abstract) in enumerate(test_data[:,1]) :
                abstract=re.sub(r"[^a-zA-Z]+", ' ', abstract)
                abstract=abstract.split()
                answers=np.zeros(self.n_classes)
                abstract=np.array(abstract)
                for word in abstract:
                    word=word.lower()
                    #word=word.isalnum()
                    for ex in range(0,(self.n_classes)) :
                        if answers[ex]==0 and word in self.train_dict[ex] and not word in mot_interdit and len(word)>2 :
                             p=self.train_dict[ex][word]
                             answers[ex]=np.log(p+1)
                        if word in self.train_dict[ex] :
                            Pmot=0
                            for ex2 in range(0,(self.n_classes)) :
                                try : 
                                    Pmot+=self.train_dict[ex2][word]
                                except :
                                    Pmot+=0
                                    
                            p=self.train_dict[ex][word]*self.n_byclasses[ex]/Pmot
                            
                            answers[ex]+=np.log(p+1) #!!!! overflow
                            #print(answers[ex])
                            
                #print(answers)
                y.append(self.classes[np.argmax(answers)])
                    

            return y
            
      
def error_rate(data,data_range):
     n=data_range
     order=np.argsort(np.random.random(size=n))
     erreur=[]
     for k in range(0,10) :
         val_data=data[order][int(n/10*k):int(n/10*(k+1))]
         train=np.concatenate((data[order,:][0:int(n/10*k),:],data[order,:][int(n/10*(k+1))::,:]))
         print(train.shape,val_data.shape)
         f=Bayes_naif(train,train[:,2])
         f.train()
         # print(f.train_dict)
         y=f.compute_predictions(val_data)
         erreur.append(len(np.where(y==val_data[:,2])[0])/len(val_data[:,2]))
        
    
     return erreur,y,f

    



for data_range in [0] :
   
    
    nbdata=7500-data_range
    erreur,y,f=error_rate(train.values,nbdata)
    print(erreur)
    erreur_err=np.std(erreur)
    erreurmean=np.mean(erreur)
    plt.errorbar(nbdata,erreurmean,yerr=erreur_err,fmt='.')
plt.grid()
plt.xlabel('Taille du jeu de données d\'entrainement')
plt.ylabel(r'Taux de réussite de classification($D_{val}$)')

#%%
test=pd.read_csv('data/test.csv')
f=Bayes_naif(train.values,train.values[:,2])
f.train()
# print(f.train_dict)
y=f.compute_predictions(test.values)

df = pd.DataFrame(y, columns=["Category"])
df.to_csv('solution.csv')