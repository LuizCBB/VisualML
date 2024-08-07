# Imports
import time
import numpy as np
import pandas as pd
import streamlit as st
import sklearn.metrics
import sklearn.datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from PIL import Image
from sklearn.model_selection import train_test_split
import funcoes
from numpy import argmax

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def carrega_binaria():
    # CSS
    st.markdown(""" <style>
        #div_algorithm_name {
          background-color: lightblue;
          text-align: left;
          height: 40px;
          max-width: 2000px; 
          border-radius: 10px;
          padding-left: 10px;
        }
         
        #algorithm_name_title{
            font-size:24px;
            font-family: 'Arial Narrow';
            color: black;
        }
    </style> """, unsafe_allow_html=True)
    
    if 'step' not in st.session_state:
        st.session_state.step = 'step1'
    
    # Logo
    col_logo1, col_logo2, col_logo3 = st.columns(3)
    with col_logo1:
        st.write(' ')
    with col_logo2:
        st.image("logo_visualML2.jpg")
    with col_logo3:
        st.write(' ')
    
    #Título
    st.markdown('<div id="div_algorithm_name"> <p id = "algorithm_name_title"> Classificação Binária</p> </div></br>', unsafe_allow_html=True)
    st.write('Em desenvolvimento ')







                                
                
            



