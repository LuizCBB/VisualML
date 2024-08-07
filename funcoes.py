import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
import streamlit as st
import matplotlib.pyplot as plt

    
def validação_cruzada(instancias_geradas, X, y):    
    modelos_acuracias = []   # definindo uma variável que vai guardar os modelos e seus respectivos valores de acurácia média
    
    # cria uma barra de progresso
    texto = "Executando a validação cruzada ... " 
    barra_progresso = st.progress(0, text= texto)
    contador = 0
    n_iter = len(instancias_geradas)
    
    for ig in instancias_geradas:
        # atualiza a barra de progresso
        contador += 1
        barra_progresso.progress(contador/n_iter, text = texto + str(contador) + " de " + str(n_iter))
        
        # validação cruzada de 10 vezes
        cv = RepeatedStratifiedKFold(n_splits = 10)
        n_scores = cross_val_score(ig[1], X, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
        
        # guardando o modelo e o valor da acurácia média de de validação cruzada
        modelos_acuracias.append([ig[0], ig[1], mean(n_scores)])
   
    return modelos_acuracias
    
def treina_sem_validação_cruzada(instancias_geradas, X, y):    
    modelos_acuracias = []   # definindo uma variável que vai guardar os modelos e seus respectivos valores de acurácia sobre a base de treino
    
    # cria uma barra de progresso
    texto = "Executando o treinamento sem fração de validação ... " 
    barra_progresso = st.progress(0, text= texto)
    contador = 0
    n_iter = len(instancias_geradas)
    for ig in instancias_geradas:
        # atualiza a barra de progresso
        contador += 1
        barra_progresso.progress(contador/n_iter, text = texto + str(contador) + " de " + str(n_iter))
        
        # Treina o modelo
        modelo = ig[1].fit(X, y)

        # Faz previsões no dataset conjunto X
        y_pred = modelo.predict(X)

        # Avalia a acurácia do modelo
        acuracia = accuracy_score(y, y_pred)
        
        # guardando o modelo e o valor da acurácia
        modelos_acuracias.append([ig[0], ig[1], acuracia])   
        
    # Retorna os modelos e suas respctivas acurácias
    return modelos_acuracias      

def testa_melhor_modelo(melhor_modelo, X, y):   
    resultados = {}
    # faz predições sobre a base de dados teste (X_teste)
    yhat = melhor_modelo.predict(X)
    resultados["yhat"] = yhat
    
    # Acurácia do modelo sobre os dados teste
    resultados["acuracia"] = accuracy_score(y, yhat)
    
    # Precisão do modelo sobre os dados teste
    resultados["precisao"] = precision_score(y, yhat, average="macro")
    
    # Recall do modelo sobre os dados teste
    resultados["recall"] = recall_score(y, yhat, average="macro")
    
    # f1 score do modelo sobre os dados teste
    resultados["f1"] = f1_score(y, yhat, average="macro")
    
    # Matriz de confusão
    matriz = confusion_matrix(y, yhat)
    fig, ax = plt.subplots()
    sns.heatmap(matriz, cmap='coolwarm', annot=True, linewidth=1, fmt='d', ax=ax)
    resultados["fig_matriz_confusao"] = fig
    
    return resultados   

def cria_modelo(instancia, X, y):
    return instancia.fit(X, y)
    


