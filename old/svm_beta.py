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
gamma=0.1
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
       
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)



class LinearModel:
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
        
        self.w=np.random.random((15,15))
        self.b=np.random.random(size=(15,15))
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
    
    def error_rate(self, X, y,ex1,ex2): 
        """Retourne le taux d'erreur pour un batch X
        """
        y=np.array(y)[:,np.newaxis]
        return np.mean(y*self.predict(X,ex1,ex2)  < 0)

    # les méthodes loss et gradient seront redéfinies dans les classes enfants
    def loss(self, X, y): 
        return 0
    
    def predict(self,X,ex1,ex2) :
        aa=np.dot(X,self.w[ex1,ex2])+self.b[ex1,ex2]
        return -np.sign(aa)
    def gradient(self, X, y,ex1,ex2): 
        y=np.array(y)
        #yy=self.w[ex1,ex2]*y
        #return np.max([yy,np.zeros(len(y))],axis=0)
        return ((self.predict(X,ex1,ex2) - y[:, np.newaxis]) * X).mean(axis=0)
    def train(self,n_steps,stepsize):
        #le but est de minimiser norme(w)?
        
        #Étape1 : Convertir tous les abstract en vecteur
        X=[]
        y=[]
        for (ex1,c1) in enumerate(self.classes) :
            for (ex2,c2) in enumerate(self.classes) :
                print(ex1,ex2)
                if c1!=c2 :
                    for (ex,abstract) in enumerate(self.train_inputs[:,1][np.where(self.train_inputs[:,2]==c1)]) :
                        vecteur=self.word2vec(abstract)
                        vecteur=(-vecteur+1)**len(vecteur)
                        X.append(vecteur)
                        y.append(-1)
                        
                    for (ex,abstract) in enumerate(self.train_inputs[:,1][np.where(self.train_inputs[:,2]==c2)]) :
                        vecteur=self.word2vec(abstract)
                        vecteur=(vecteur+1)**len(vecteur)
                        X.append(vecteur)
                        y.append(+1)
                    
                        
                        
                    
                   
                    losses = []
                    errors = []
            
                    for i in range(n_steps):
                        # Gradient Descent
                        self.w[ex1,ex2] -= stepsize *np.mean(self.gradient(X, y,ex1,ex2))
                        self.b[ex1,ex2] -= stepsize*self.b[ex1,ex2]
                        # Update losses
                        losses += [self.loss(X, y)]
                        print(self.w[ex1,ex2],self.error_rate(X, y,ex1,ex2))
                        # Update errors
                        errors += [self.error_rate(X, y,ex1,ex2)]
               
                    X=[]
                    y=[]
                    
               
            
    
    def compute_predictions(self, test_data):
            y=[]
            results=np.zeros(15)
            for abstract in test_data[:,1] :
               vecteur=self.word2vec(abstract)
               for ex1 in range(0,15) :
                   for ex2 in range(0,15) :
                       if ex1!=ex2 :
                           f=self.predict(vecteur,ex1,ex2)
                           if f<0 :
                               results[ex1]+=1
                           else :
                               results[ex2]+=1
                               
               print(results)
               y.append(self.classes[np.argmax(results)])
            return y
            


class LinearRegression(LinearModel):

    def __init__(self, w0, reg):
        super().__init__(w0, reg)

    def loss(self, X, y):
        """Calcule la perte moyenne pour une batch X. 
        Prend en entrée une matrice X et le vecteur y et retourne un scalaire.
        """
        return 0.5 * np.mean((self.predict(X) - y) ** 2) + self.reg * 0.5 * np.sum(self.w ** 2)

    def gradient(self, X, y):
        """Calcule le gradient de la fonction de perte par rapport a w pour un batch X.
        Prend en entrée une matrice X et le vecteur y.
        Retourne un vecteur de la meme taille que w.
        """
        return ((self.predict(X) - y)[:, np.newaxis] * X).mean(axis=0) + self.reg * self.w
    


df1=train.values[0:6000,:]
df2=train.values[6000:7500,:]
#erreur,y=error_rate(df1,df2)
test=pd.read_csv('data/test.csv')
#%%
f=LinearModel(df1,df1[:,2])
f.train(100,5)
# print(f.train_dict)
y=f.compute_predictions(df2)
print(len(np.where(y!=df2[:,2])[0]))
#y=f.compute_predictions(test.values,8)

df = pd.DataFrame(y, columns=["Category"])
df.to_csv('solution.csv')
