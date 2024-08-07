import streamlit as st
from streamlit_option_menu import option_menu
import inicial, classificacao_multiclasse, classificacao_binaria, regressao, nao_supervisionado, aprendizado_profundo
st.set_page_config(layout="wide")

with st.sidebar:
    selected = option_menu("", ["Inicial", "Regressão", "Classificação Binária", "Classificação multiclasse", "Aprendizado não supervisionado", "Aprendizado profundo"], default_index = 0)
    

if selected == "Inicial":
    inicial.carrega_inicial()
elif selected == "Classificação multiclasse":
    classificacao_multiclasse.carrega_multiclasse()
elif selected == "Classificação Binária":
    classificacao_binaria.carrega_binaria()
elif selected == "Regressão":
    regressao.carrega_regressao()
elif selected == "Aprendizado não supervisionado":
    nao_supervisionado.carrega_nao_supervisionado()
elif selected == "Aprendizado profundo":
    aprendizado_profundo.carrega_aprendizado_profundo()
    
    
    
    