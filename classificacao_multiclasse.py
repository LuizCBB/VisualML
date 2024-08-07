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
from sklearn import svm

def carrega_multiclasse():
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
    st.markdown('<div id="div_algorithm_name"> <p id = "algorithm_name_title"> Classificação Multiclasse</p> </div></br>', unsafe_allow_html=True)
    
    if (st.session_state.step == 'step1'):

        st.write("**ETAPA 1: definir o banco de dados**")

        with st.expander(":blue[**1 - Selecionar um banco de dados**]"):
            
                
            dataset_selectbox = st.selectbox('**Escolha um banco de dados:**',('Iris', 'Wine'))
            st.session_state.dataset = dataset_selectbox
            if dataset_selectbox == 'Iris':
                dataset = sklearn.datasets.load_iris()
                st.markdown("""<p style='text-align: justify; font-size: 15px;'>
                O <i>dataset</i> Iris é um conjunto de dados multivariados que consiste de amostras de cada uma de três espécies de plantas
                do gênero Iris (<i>Iris setosa</i>, <i>Iris virginica</i> e <i>Iris versicolor</i>). Quatro variáveis foram medidas em cada amostra: 
                comprimento da sépala (sepal length), largura da sépala (sepal width), comprimento da pétala (petal length) e a largura da pétala (petal width).
                Todas as medidas estão em centímetros. Veja mais informações sobre este <i>dataset</i> <a href='https://archive.ics.uci.edu/dataset/53/iris'>aqui</a>.</p>
                """, unsafe_allow_html=True)
                    
            elif dataset_selectbox == 'Wine':
                dataset = sklearn.datasets.load_wine()
                st.markdown("""<p style='text-align: justify; font-size: 15px;'>
                O <i>dataset</i> Wine é resultado de uma análise química de vinhos cultivados na mesma região da Itália, mas derivados de três cultivares
                diferentes. Dessa forma, tem-se três categorias (identificadas no <i>dataset</i> como class_0, class_1 e class_2) para a variável a ser predita (variável
                <i>target</i>). A análise determinou as quantidades de 13 constituintes (que representam as variávels preditoras ou explicativas) encontrados em cada um dos três tipos
                de vinhos: álcool (alcohol), ácido málico (malic_acid), cinzas (ash), alcalinidade das cinzas (alcalinity_of_ash), magnésio (magnesium),
                fenóis totais (total_phenols), flavonóides (flavanoids), fenóis não flavonóides (nonflavanoid_phenols), proantocianinas (proanthocyanins),
                intensidade de cor (color_intensity), matiz (hue), OD280/OD315 de vinhos diluídos (OD280/OD315_of_diluted_wines), prolina (proline). Veja mais 
                informações sobre este <i>dataset</i> <a href='https://archive.ics.uci.edu/dataset/109/wine'>aqui</a>.</p>
                """, unsafe_allow_html=True)

        

        with st.expander(":blue[**2 - Visualizar o banco de dados**]"):
            # Extrai a variável alvo
            targets = dataset.target_names

            # Prepara o dataframe com os dados
            dataframe = pd.DataFrame (dataset.data, columns = dataset.feature_names)
            estatisticas = dataframe.describe()
            dataframe['target'] = pd.Series(dataset.target)
            dataframe['target labels'] = pd.Series(targets[i] for i in dataset.target)

            # Mostra o dataset selecionado pelo usuário
            st.write(dataframe)
            st.write("Número de observações: ", str(dataframe.shape[0]))
            st.write("Rótulos (labels): ", str(dataframe['target labels'].unique()).replace("[", "").replace("]","").replace(" ",", ").replace("'",""))
            st.write("Rótulos (labels) na forma numérica: ", str(dataframe['target'].unique()).replace("[", "").replace("]","").replace(" ",", ")) 
            var_descritivas = ""
            for i in range(len(estatisticas.columns.values.tolist())):
                if i == len(estatisticas.columns.values.tolist())-1:
                    var_descritivas += estatisticas.columns.values.tolist()[i]
                else:
                    var_descritivas += estatisticas.columns.values.tolist()[i] + ", "
            st.write("Variáveis descritivas: ", var_descritivas) 
            st.write("Resumo estatístico de cada variável descritiva:", estatisticas.drop(["count", "25%", "50%","75%"]).set_index([pd.Index(["Média", "Desvio padrão", "Mínimo", "Máximo"])]))
            
        with st.expander(":blue[**3 - Realizar a divisão dos dados em um banco de dados de treino e teste**]"):
            split = st.slider('Escolha o percentual dos dados que ficará para teste:', 10, 50, 30)
            split = split/100
            # Extrai os dados de treino e teste            
            X_treino, X_teste, y_treino, y_teste = train_test_split(dataset.data, dataset.target, test_size = float(split))
            
            # Coloca as bases de dados de treino e teste na sessão
            st.session_state.Xtreino = X_treino
            st.session_state.ytreino = y_treino
            st.session_state.Xteste = X_teste
            st.session_state.yteste = y_teste
            
            # Imprime o número de observações nos bancos de dados treino e teste
            st.write("Número de observações no banco de dados de treino: ", str(X_treino.shape[0]))
            st.write("Número de observações no banco de dados de teste: ", str(X_teste.shape[0]))
        
        
        if 'bt_etapa1_concluida_disabled' not in st.session_state:
            st.session_state.bt_etapa1_concluida_disabled = False
        if 'bt_etapa1_concluida_label' not in st.session_state:
            st.session_state.bt_etapa1_concluida_label = "Finalizar esta etapa"
            
        # Função para direcionar a etapa 2
        def update_step2():
            st.session_state.step = 'step2'
        
        bt_etapa1_concluida = st.button("Continuar", on_click  = update_step2)
               
    elif (st.session_state.step == 'step2'):
        st.write("**ETAPA 2: treinar, otimizar e selecionar um modelo**")
        with st.expander(":blue[**1 - Selecionar um ou mais algoritmos**]"):
            algoritmos = st.multiselect(
                "**Selecione um ou mais algoritmos:**",
                ["Multinomial Logistic Regression (MLR)", "Multi-layer Perceptron (MLP)", "K-nearest Neighbors (KNN)", "Support Vector Machines (SVM)", "Decision Trees (DT)"],
                ["Multinomial Logistic Regression (MLR)"]
            )
        
        with st.expander(":blue[**2 - Selecionar os hiperparâmetros**]"):
            
            if (not algoritmos):
                st.write("**Você deve selecionar ao menos um algoritmo.**")
            else:
                hiperparametros_selectbox = st.selectbox('**Hiperparâmetros:**',('Padrão', 'Customizado', 'Grid search'))   
                if (hiperparametros_selectbox == 'Padrão'):
                
                    # Gera as instâncias dos algoritmos de acordo com as escolhas do usuário
                    instancias_geradas = []
                    for a in algoritmos:                        
                        if (a == "Multinomial Logistic Regression (MLR)"):
                            instancias_geradas.append([a, LogisticRegression(multi_class='multinomial')])
                        if (a == "Multi-layer Perceptron (MLP)"):
                            instancias_geradas.append([a, MLPClassifier()])
                        if (a == "K-nearest Neighbors (KNN)"):
                            instancias_geradas.append([a, KNeighborsClassifier()])
                        if (a == "Support Vector Machines (SVM)"):
                            instancias_geradas.append([a, svm.SVC()])
                        if (a == "Decision Trees (DT)"):               
                            instancias_geradas.append([a, DecisionTreeClassifier()])
                
                elif (hiperparametros_selectbox == 'Customizado'):
                    st.write("**Alguns hiperparâmetros para testar:**")                   
                    
                    # Gera as instâncias dos algoritmos de acordo com as escolhas do usuário                    
                    instancias_geradas = []
                    for a in algoritmos:                        
                        if (a == "Multinomial Logistic Regression (MLR)"):
                            col_hp1, col_hp2, col_hp3, col_hp4 = st.columns(4)
                            with col_hp1:
                                st.write("**" + a + "**")  
                            with col_hp2:    
                                solver_MLR = st.selectbox("MLR: solver", ("lbfgs", "newton-cg", "sag", "saga"), index=0)
                            with col_hp3:                                
                                C_MLR = st.selectbox("MLR: C", np.arange(0.1, 10, 0.1), index=9)
                            with col_hp4:                                
                                max_iter_MLR =  st.selectbox("MLR: max_iter", np.arange(1, 1000, 1), index=99)
                                
                            string_instancia_MLR =  "LogisticRegression(multi_class='multinomial', solver='"+solver_MLR+ "', C="+str(C_MLR)+", max_iter="+ str(max_iter_MLR)+")"                            
                            instancias_geradas.append([a, eval(string_instancia_MLR)])
                                
                        elif (a == "Multi-layer Perceptron (MLP)"):
                            col_hp1, col_hp2, col_hp3, col_hp4 = st.columns(4)
                            with col_hp1:
                                st.write("**" + a + "**")  
                            with col_hp2:    
                                solver_MLP = st.selectbox("MLP: solver", ("lbfgs", "sgd", "adam"), index=2)
                            with col_hp3:                                
                                hidden_layer_sizes_1 = st.selectbox("MLP: hidden_layer_sizes (1ª)", np.arange(1, 1000, 1), index=99)
                                hidden_layer_sizes_2 = st.selectbox("MLP: hidden_layer_sizes (2ª)", np.arange(1, 1000, 1), index=49)
                                hidden_layer_sizes = (hidden_layer_sizes_1,hidden_layer_sizes_2)
                            with col_hp4:                                
                                max_iter_MLP =  st.selectbox("MLP: max_iter(epochs)", np.arange(1, 1000, 1), index=199)
                                
                            string_instancia_MLP =  "MLPClassifier(solver='"+solver_MLP+ "', hidden_layer_sizes="+ str(hidden_layer_sizes)+", max_iter="+ str(max_iter_MLP)+")"                              
                            instancias_geradas.append([a, eval(string_instancia_MLP)])   

                        elif (a == "K-nearest Neighbors (KNN)"):
                            col_hp1, col_hp2, col_hp3, col_hp4 = st.columns(4)
                            with col_hp1:
                                st.write("**" + a + "**")  
                            with col_hp2:    
                                n_neighbors = st.selectbox("KNN: n_neighbors", np.arange(3, 15, 1), index=2)
                            with col_hp3:                                
                                metric = st.selectbox("KNN: metric", ("minkowski", "euclidean", "manhattan"), index=0)
                            with col_hp4:                                
                                algorithm =  st.selectbox("KNN: algorithm", ("auto", "ball_tree", "kd_tree", "brute"), index=0) 
                                
                            string_instancia_KNN =  "KNeighborsClassifier(n_neighbors="+str(n_neighbors)+ ", metric='"+ metric+"', algorithm='"+algorithm+"')"                              
                            instancias_geradas.append([a, eval(string_instancia_KNN)])
                        
                        elif (a == "Support Vector Machines (SVM)"):
                            col_hp1, col_hp2, col_hp3, col_hp4 = st.columns(4)
                            with col_hp1:
                                st.write("**" + a + "**")  
                            with col_hp2:    
                                kernel = st.selectbox("SVM: kernel", ("linear", "poly", "rbf", "sigmoid"), index=2)
                            with col_hp3:                                
                                C_SVM = st.selectbox("SVM: C", np.arange(0.1, 10, 0.1), index=9)
                            with col_hp4:                                
                                max_iter_SVM =  st.selectbox("SVM: max_iter", np.arange(1, 2000, 1), index=199)
                                
                            string_instancia_SVM =  "svm.SVC(kernel='"+kernel+ "', C="+ str(C_SVM)+", max_iter="+ str(max_iter_SVM)+")"                             
                            instancias_geradas.append([a, eval(string_instancia_SVM)])
                      
                        elif (a == "Decision Trees (DT)"):
                            col_hp1, col_hp2, col_hp3, col_hp4 = st.columns(4)
                            with col_hp1:
                                st.write("**" + a + "**")  
                            with col_hp2:    
                                criterion = st.selectbox("DT: criterion", ("gini", "entropy", "log_loss"), index=0)
                            with col_hp3:                                
                                max_depth = st.selectbox("DT: max_depth", np.arange(1, 20, 1), index=2)
                            with col_hp4:                                
                                splitter =  st.selectbox("DT: splitter", ("best", "random"), index=0) 
                            
                            string_instancia_DT =  "DecisionTreeClassifier(criterion='"+criterion+ "', max_depth="+ str(max_depth)+", splitter='"+splitter+"')"                              
                            instancias_geradas.append([a, eval(string_instancia_DT)])   
                            
                    # imprime a legenda dos hiperparâmetros    
                    if ("Multi-layer Perceptron (MLP)" in algoritmos or "Multinomial Logistic Regression (MLR)" in algoritmos):
                        st.write("- solver: algoritmo de otimização")
                    if ("Multi-layer Perceptron (MLP)" in algoritmos or "Multinomial Logistic Regression (MLR)" in algoritmos or "Support Vector Machines (SVM)" in algoritmos):              
                        st.write("- max_iter: número máximo de iterações necessárias para que o solver convirja")
                    if ("Multi-layer Perceptron (MLP)" in algoritmos):
                        st.write("- hidden_layer_sizes: número de neurônios na i-ésima camada oculta")
                    if ("K-nearest Neighbors (KNN)" in algoritmos):
                        st.write("- n_neighbors: número de vizinhos a serem usados para as consultas")
                        st.write("- metric: métrica a ser usada para o cálculo de distância")   
                        st.write("- algorithm: algoritmo usado para calcular os vizinhos mais próximos")
                    if ("Support Vector Machines (SVM)" in algoritmos):
                        st.write("- kernel - especifica o tipo de kernel a ser usado no algoritmo")
                        st.write("- C - parâmetro de regularização (a força da regularização é inversamente proporcional ao valor de C, ou seja, valores menores especificam uma regularização mais forte)")                       
                    if ("Decision Trees (DT)" in algoritmos):
                        st.write("- criterion: função para medir a qualidade de uma divisão.")   
                        st.write("- max_depth: a profundidade máxima da árvore.")
                        st.write("- splitter: a estratégia usada para escolher a divisão em cada nó.")
                        st.write("")
                        
                elif (hiperparametros_selectbox == 'Grid search'): 
                    instancias_geradas = []
                    for a in algoritmos:                        
                        if (a == "Multinomial Logistic Regression (MLR)"):
                            col_hp1, col_hp2, col_hp3, col_hp4 = st.columns(4)
                            with col_hp1:
                                st.write("**" + a + "**")
                            with col_hp2:
                                op_solver_MLR = st.multiselect("MLR: solver", ["lbfgs", "newton-cg", "sag", "saga"],["lbfgs"])
                            with col_hp3:            
                                op_C_MLR = st.multiselect("MLR: C", [str(i) for i in np.arange(0.1, 10, 0.1)],["1.0"])  
                            with col_hp4:            
                                op_max_iter_MLR = st.multiselect("MLR: max_iter", [str(i) for i in range(10,200,90)],["100"])                               
                                                        
                            if not op_solver_MLR:
                                solver_MLR = "lbfgs"
                                if not op_C_MLR:
                                    C_MLR = 1.0
                                    if not op_max_iter_MLR:
                                        max_iter_MLR = 100
                                        string_instancia_MLR =  "LogisticRegression(multi_class='multinomial', solver='"+solver_MLR+ "', C="+str(C_MLR)+", max_iter="+ str(max_iter_MLR)+")"                            
                                        instancias_geradas.append([a, eval(string_instancia_MLR)])                                                
                                    else:
                                        for max_iter_MLR in op_max_iter_MLR:
                                            string_instancia_MLR =  "LogisticRegression(multi_class='multinomial', solver='"+solver_MLR+ "', C="+str(C_MLR)+", max_iter="+ str(max_iter_MLR)+")"                            
                                            instancias_geradas.append([a, eval(string_instancia_MLR)]) 
                                else:
                                    for C_MLR in op_C_MLR:
                                        if not op_max_iter_MLR:
                                            max_iter_MLR = 100
                                            string_instancia_MLR =  "LogisticRegression(multi_class='multinomial', solver='"+solver_MLR+ "', C="+str(C_MLR)+", max_iter="+ str(max_iter_MLR)+")"                            
                                            instancias_geradas.append([a, eval(string_instancia_MLR)])                                                
                                        else:
                                            for max_iter_MLR in op_max_iter_MLR:
                                                string_instancia_MLR =  "LogisticRegression(multi_class='multinomial', solver='"+solver_MLR+ "', C="+str(C_MLR)+", max_iter="+ str(max_iter_MLR)+")"                            
                                                instancias_geradas.append([a, eval(string_instancia_MLR)])  
                                            
                            else:
                                for solver_MLR in op_solver_MLR:
                                    if not op_C_MLR:
                                        C_MLR = 1.0
                                        if not op_max_iter_MLR:
                                            max_iter_MLR = 100
                                            string_instancia_MLR =  "LogisticRegression(multi_class='multinomial', solver='"+solver_MLR+ "', C="+str(C_MLR)+", max_iter="+ str(max_iter_MLR)+")"                            
                                            instancias_geradas.append([a, eval(string_instancia_MLR)])                                                
                                        else:
                                            for max_iter_MLR in op_max_iter_MLR:
                                                string_instancia_MLR =  "LogisticRegression(multi_class='multinomial', solver='"+solver_MLR+ "', C="+str(C_MLR)+", max_iter="+ str(max_iter_MLR)+")"                            
                                                instancias_geradas.append([a, eval(string_instancia_MLR)]) 
                                    else:
                                        for C_MLR in op_C_MLR:
                                            if not op_max_iter_MLR:
                                                max_iter_MLR = 100
                                                string_instancia_MLR =  "LogisticRegression(multi_class='multinomial', solver='"+solver_MLR+ "', C="+str(C_MLR)+", max_iter="+ str(max_iter_MLR)+")"                            
                                                instancias_geradas.append([a, eval(string_instancia_MLR)])                                                
                                            else:
                                                for max_iter_MLR in op_max_iter_MLR:
                                                    string_instancia_MLR =  "LogisticRegression(multi_class='multinomial', solver='"+solver_MLR+ "', C="+str(C_MLR)+", max_iter="+ str(max_iter_MLR)+")"                            
                                                    instancias_geradas.append([a, eval(string_instancia_MLR)])   
                                                   
                               
                        elif (a == "Multi-layer Perceptron (MLP)"):
                            col_hp1, col_hp2, col_hp3, col_hp4 = st.columns(4)
                            with col_hp1:
                                st.write("**" + a + "**")
                            with col_hp2:
                                op_solver_MLP = st.multiselect("MLP: solver", ["lbfgs", "sgd", "adam"],["adam"])
                            with col_hp3:            
                                op_hidden_layer_sizes = st.multiselect("MLP: hidden_layer_sizes (1ª)", [str(i) for i in np.arange(1, 1000, 1)],["100"])
                            with col_hp4:            
                                op_max_iter_MLP = st.multiselect("MLP: max_iter(epochs)", [str(i) for i in np.arange(1, 1000, 1)],["200"])
                        
                            if not op_solver_MLP:
                                solver_MLP = "adam"
                                if not op_hidden_layer_sizes:
                                    hidden_layer_sizes = "(100,)"
                                    if not op_max_iter_MLP:
                                        max_iter_MLP = 200
                                        string_instancia_MLP =  "MLPClassifier(solver='"+solver_MLP+ "', hidden_layer_sizes="+ str(hidden_layer_sizes)+", max_iter="+ str(max_iter_MLP)+")"                              
                                        instancias_geradas.append([a, eval(string_instancia_MLP)])                                                 
                                    else:
                                        for max_iter_MLP in op_max_iter_MLP:
                                            string_instancia_MLP =  "MLPClassifier(solver='"+solver_MLP+ "', hidden_layer_sizes="+ str(hidden_layer_sizes)+", max_iter="+ str(max_iter_MLP)+")"                              
                                            instancias_geradas.append([a, eval(string_instancia_MLP)])  
                                else:
                                    for hidden_layer_sizes in op_hidden_layer_sizes:
                                        if not op_max_iter_MLP:
                                            max_iter_MLP = 200
                                            string_instancia_MLP =  "MLPClassifier(solver='"+solver_MLP+ "', hidden_layer_sizes="+ str(hidden_layer_sizes)+", max_iter="+ str(max_iter_MLP)+")"                              
                                            instancias_geradas.append([a, eval(string_instancia_MLP)])                                                 
                                        else:
                                            for max_iter_MLP in op_max_iter_MLP:
                                                string_instancia_MLP =  "MLPClassifier(solver='"+solver_MLP+ "', hidden_layer_sizes="+ str(hidden_layer_sizes)+", max_iter="+ str(max_iter_MLP)+")"                              
                                                instancias_geradas.append([a, eval(string_instancia_MLP)])   
                                            
                            else:
                                for solver_MLP in op_solver_MLP:
                                    if not op_hidden_layer_sizes:
                                        hidden_layer_sizes = "(100,)"
                                        if not op_max_iter_MLP:
                                            max_iter_MLP = 200
                                            string_instancia_MLP =  "MLPClassifier(solver='"+solver_MLP+ "', hidden_layer_sizes="+ str(hidden_layer_sizes)+", max_iter="+ str(max_iter_MLP)+")"                              
                                            instancias_geradas.append([a, eval(string_instancia_MLP)])                                                
                                        else:
                                            for max_iter_MLP in op_max_iter_MLP:
                                                string_instancia_MLP =  "MLPClassifier(solver='"+solver_MLP+ "', hidden_layer_sizes="+ str(hidden_layer_sizes)+", max_iter="+ str(max_iter_MLP)+")"                              
                                                instancias_geradas.append([a, eval(string_instancia_MLP)])  
                                    else:
                                        for hidden_layer_sizes in op_hidden_layer_sizes:
                                            if not op_max_iter_MLP:
                                                max_iter_MLP = 200
                                                string_instancia_MLP =  "MLPClassifier(solver='"+solver_MLP+ "', hidden_layer_sizes="+ str(hidden_layer_sizes)+", max_iter="+ str(max_iter_MLP)+")"                              
                                                instancias_geradas.append([a, eval(string_instancia_MLP)])                                                 
                                            else:
                                                for max_iter_MLP in op_max_iter_MLP:
                                                    string_instancia_MLP =  "MLPClassifier(solver='"+solver_MLP+ "', hidden_layer_sizes="+ str(hidden_layer_sizes)+", max_iter="+ str(max_iter_MLP)+")"                              
                                                    instancias_geradas.append([a, eval(string_instancia_MLP)])  
             
                        elif (a == "K-nearest Neighbors (KNN)"):
                            col_hp1, col_hp2, col_hp3, col_hp4 = st.columns(4)
                            with col_hp1:
                                st.write("**" + a + "**")
                            with col_hp2:
                                op_n_neighbors = st.multiselect("KNN: n_neighbors", [str(i) for i in np.arange(3, 15, 1)],["5"])
                            with col_hp3:            
                                op_metric = st.multiselect("KNN: metric", ["minkowski", "euclidean", "manhattan"],["minkowski"])
                            with col_hp4:            
                                op_algorithm = st.multiselect("KNN: algorithm", ["auto", "ball_tree", "kd_tree", "brute"],["auto"])
                            
                            if not op_n_neighbors:
                                n_neighbors = "5"
                                if not op_metric:
                                    metric = "minkowski"
                                    if not op_algorithm:
                                        algorithm = "auto"
                                        string_instancia_KNN =  "KNeighborsClassifier(n_neighbors="+str(n_neighbors)+ ", metric='"+ metric+"', algorithm='"+algorithm+"')"                              
                                        instancias_geradas.append([a, eval(string_instancia_KNN)])                                                 
                                    else:
                                        for algorithm in op_algorithm:
                                            string_instancia_KNN =  "KNeighborsClassifier(n_neighbors="+str(n_neighbors)+ ", metric='"+ metric+"', algorithm='"+algorithm+"')"                              
                                            instancias_geradas.append([a, eval(string_instancia_KNN)]) 
                                else:
                                    for metric in op_metric:
                                        if not op_algorithm:
                                            algorithm = "auto"
                                            string_instancia_KNN =  "KNeighborsClassifier(n_neighbors="+str(n_neighbors)+ ", metric='"+ metric+"', algorithm='"+algorithm+"')"                              
                                            instancias_geradas.append([a, eval(string_instancia_KNN)])                                                 
                                        else:
                                            for algorithm in op_algorithm:
                                                string_instancia_KNN =  "KNeighborsClassifier(n_neighbors="+str(n_neighbors)+ ", metric='"+ metric+"', algorithm='"+algorithm+"')"                              
                                                instancias_geradas.append([a, eval(string_instancia_KNN)])   
                                            
                            else:
                                for n_neighbors in op_n_neighbors:
                                    if not op_metric:
                                        metric = "minkowski"
                                        if not op_algorithm:
                                            algorithm = "auto"
                                            string_instancia_KNN =  "KNeighborsClassifier(n_neighbors="+str(n_neighbors)+ ", metric='"+ metric+"', algorithm='"+algorithm+"')"                              
                                            instancias_geradas.append([a, eval(string_instancia_KNN)])                                               
                                        else:
                                            for algorithm in op_algorithm:
                                                string_instancia_KNN =  "KNeighborsClassifier(n_neighbors="+str(n_neighbors)+ ", metric='"+ metric+"', algorithm='"+algorithm+"')"                              
                                                instancias_geradas.append([a, eval(string_instancia_KNN)])  
                                    else:
                                        for metric in op_metric:
                                            if not op_algorithm:
                                                algorithm = "auto"
                                                string_instancia_KNN =  "KNeighborsClassifier(n_neighbors="+str(n_neighbors)+ ", metric='"+ metric+"', algorithm='"+algorithm+"')"                              
                                                instancias_geradas.append([a, eval(string_instancia_KNN)])                                                 
                                            else:
                                                for algorithm in op_algorithm:
                                                    string_instancia_KNN =  "KNeighborsClassifier(n_neighbors="+str(n_neighbors)+ ", metric='"+ metric+"', algorithm='"+algorithm+"')"                              
                                                    instancias_geradas.append([a, eval(string_instancia_KNN)])
                          
                        elif (a == "Support Vector Machines (SVM)"):
                            col_hp1, col_hp2, col_hp3, col_hp4 = st.columns(4)
                            with col_hp1:
                                st.write("**" + a + "**")
                            with col_hp2:
                                op_kernel = st.multiselect("SVM: kernel", ["linear", "poly", "rbf", "sigmoid"],["rbf"])
                            with col_hp3:            
                                op_C_SVM = st.multiselect("SVM: C", [str(i) for i in np.arange(0.1, 10, 0.1)],["1.0"])
                            with col_hp4:            
                                op_max_iter_SVM = st.multiselect("SVM: max_iter", [str(i) for i in np.arange(1, 2000, 1)],["200"])
                            
                            if not op_kernel:
                                kernel = "rbf"
                                if not op_max_iter_SVM:
                                    max_iter_SVM = "-1"
                                    if not op_C_SVM:
                                        C_SVM = "1.0"
                                        string_instancia_SVM =  "svm.SVC(kernel='"+kernel+ "', C="+ str(C_SVM)+", max_iter="+ str(max_iter_SVM)+")"                             
                                        instancias_geradas.append([a, eval(string_instancia_SVM)])                                                 
                                    else:
                                        for C_SVM in op_C_SVM:
                                            string_instancia_SVM =  "svm.SVC(kernel='"+kernel+ "', C="+ str(C_SVM)+", max_iter="+ str(max_iter_SVM)+")"                             
                                            instancias_geradas.append([a, eval(string_instancia_SVM)]) 
                                else:
                                    for max_iter_SVM in op_max_iter_SVM:
                                        if not op_C_SVM:
                                            C_SVM = "1.0"
                                            string_instancia_SVM =  "svm.SVC(kernel='"+kernel+ "', C="+ str(C_SVM)+", max_iter="+ str(max_iter_SVM)+")"                             
                                            instancias_geradas.append([a, eval(string_instancia_SVM)])                                                 
                                        else:
                                            for C_SVM in op_C_SVM:
                                                string_instancia_SVM =  "svm.SVC(kernel='"+kernel+ "', C="+ str(C_SVM)+", max_iter="+ str(max_iter_SVM)+")"                             
                                                instancias_geradas.append([a, eval(string_instancia_SVM)])   
                                            
                            else:
                                for kernel in op_kernel:
                                    if not op_max_iter_SVM:
                                        max_iter_SVM = "-1"
                                        if not op_C_SVM:
                                            C_SVM = "1.0"
                                            string_instancia_SVM =  "svm.SVC(kernel='"+kernel+ "', C="+ str(C_SVM)+", max_iter="+ str(max_iter_SVM)+")"                             
                                            instancias_geradas.append([a, eval(string_instancia_SVM)])                                               
                                        else:
                                            for C_SVM in op_C_SVM:
                                                string_instancia_SVM =  "svm.SVC(kernel='"+kernel+ "', C="+ str(C_SVM)+", max_iter="+ str(max_iter_SVM)+")"                             
                                                instancias_geradas.append([a, eval(string_instancia_SVM)])  
                                    else:
                                        for max_iter_SVM in op_max_iter_SVM:
                                            if not op_C_SVM:
                                                C_SVM = "1.0"
                                                string_instancia_SVM =  "svm.SVC(kernel='"+kernel+ "', C="+ str(C_SVM)+", max_iter="+ str(max_iter_SVM)+")"                             
                                                instancias_geradas.append([a, eval(string_instancia_SVM)])                                                 
                                            else:
                                                for C_SVM in op_C_SVM:
                                                    string_instancia_SVM =  "svm.SVC(kernel='"+kernel+ "', C="+ str(C_SVM)+", max_iter="+ str(max_iter_SVM)+")"                             
                                                    instancias_geradas.append([a, eval(string_instancia_SVM)])
                            
                            
                        elif (a == "Decision Trees (DT)"):
                            col_hp1, col_hp2, col_hp3, col_hp4 = st.columns(4)
                            with col_hp1:
                                st.write("**" + a + "**")
                            with col_hp2:
                                op_criterion = st.multiselect("DT: criterion", ["gini", "entropy", "log_loss"],["gini"])
                            with col_hp3:            
                                op_max_depth = st.multiselect("DT: max_depth", [str(i) for i in np.arange(1, 20, 1)],["3"])
                            with col_hp4:            
                                op_splitter = st.multiselect("DT: splitter", ["best", "random"],["best"])
                            
                            if not op_criterion:
                                criterion = "gini"
                                if not op_max_depth:
                                    max_depth = "None"
                                    if not op_splitter:
                                        splitter = "best"
                                        string_instancia_DT =  "DecisionTreeClassifier(criterion='"+criterion+ "', max_depth="+ str(max_depth)+", splitter='"+splitter+"')"                              
                                        instancias_geradas.append([a, eval(string_instancia_DT)])                                                 
                                    else:
                                        for splitter in op_splitter:
                                            string_instancia_DT =  "DecisionTreeClassifier(criterion='"+criterion+ "', max_depth="+ str(max_depth)+", splitter='"+splitter+"')"                              
                                            instancias_geradas.append([a, eval(string_instancia_DT)]) 
                                else:
                                    for max_depth in op_max_depth:
                                        if not op_splitter:
                                            splitter = "best"
                                            string_instancia_DT =  "DecisionTreeClassifier(criterion='"+criterion+ "', max_depth="+ str(max_depth)+", splitter='"+splitter+"')"                              
                                            instancias_geradas.append([a, eval(string_instancia_DT)])                                                 
                                        else:
                                            for splitter in op_splitter:
                                                string_instancia_DT =  "DecisionTreeClassifier(criterion='"+criterion+ "', max_depth="+ str(max_depth)+", splitter='"+splitter+"')"                              
                                                instancias_geradas.append([a, eval(string_instancia_DT)])   
                                            
                            else:
                                for criterion in op_criterion:
                                    if not op_max_depth:
                                        max_depth = "None"
                                        if not op_splitter:
                                            splitter = "best"
                                            string_instancia_DT =  "DecisionTreeClassifier(criterion='"+criterion+ "', max_depth="+ str(max_depth)+", splitter='"+splitter+"')"                              
                                            instancias_geradas.append([a, eval(string_instancia_DT)])                                               
                                        else:
                                            for splitter in op_splitter:
                                                string_instancia_DT =  "DecisionTreeClassifier(criterion='"+criterion+ "', max_depth="+ str(max_depth)+", splitter='"+splitter+"')"                              
                                                instancias_geradas.append([a, eval(string_instancia_DT)])  
                                    else:
                                        for max_depth in op_max_depth:
                                            if not op_splitter:
                                                splitter = "best"
                                                string_instancia_DT =  "DecisionTreeClassifier(criterion='"+criterion+ "', max_depth="+ str(max_depth)+", splitter='"+splitter+"')"                              
                                                instancias_geradas.append([a, eval(string_instancia_DT)])                                                
                                            else:
                                                for splitter in op_splitter:
                                                    string_instancia_DT =  "DecisionTreeClassifier(criterion='"+criterion+ "', max_depth="+ str(max_depth)+", splitter='"+splitter+"')"                              
                                                    instancias_geradas.append([a, eval(string_instancia_DT)])
                            
        with st.expander(":blue[**3 - Realizar o treinamento:**]"):
            if (algoritmos):                
                validacao_cruzada_10X = st.checkbox("**Aplicar validação cruzada de 10X**")      
                bt_treinar = st.button("Iniciar treinamento")
                if(bt_treinar):                   
                    if (validacao_cruzada_10X):                        
                        modelos_acuracias = funcoes.validação_cruzada(instancias_geradas, st.session_state.Xtreino, st.session_state.ytreino) 
                        # Info de sucesso
                        st.success("Treinamento terminado!")
                        st.write("**Valor de acurácia média obtida após o treinamento com validação cruzada de 10X:**")
                    else: 
                        modelos_acuracias = funcoes.treina_sem_validação_cruzada(instancias_geradas, st.session_state.Xtreino, st.session_state.ytreino)
                        # Info de sucesso
                        st.success("Treinamento terminado!")
                        st.write("**Valor de acurácia obtida após o treinamento sem frações de validação:**")
                    
                    valores_acuracia = []
                    for ma in modelos_acuracias:                         
                        st.write(str(ma[1]),":",str(ma[2]))
                        valores_acuracia.append(str(ma[2]))
                        
                    pos_melhor_modelo = argmax(valores_acuracia)
                    
                    if (len(valores_acuracia) > 1):                            
                        st.write("**Melhor modelo obtido:**", " ", str(modelos_acuracias[pos_melhor_modelo][1]))
                        st.session_state.melhor_modelo = modelos_acuracias[pos_melhor_modelo][1]
                        
                    else:
                        st.session_state.melhor_modelo = modelos_acuracias[pos_melhor_modelo][1]      
            else:
                st.write("**Você deve selecionar ao menos um algoritmo.**")  
        
        col_bt1, col_bt2 = st.columns(2)
        with col_bt1:
            # Função para redirecionar a etapa 1
            def update_step1():
                st.session_state.step = 'step1'
            # Botão para redirecionar a etapa 1
            bt_etapa1 = st.button("Recomeçar", on_click  = update_step1)
            
        with col_bt2:
            # Verifica se mostra ou não botão para continuar para a próxima etapa
            if 'melhor_modelo' in st.session_state:
                # Função para direcionar a etapa 3
                def update_step3():
                    st.session_state.step = 'step3'
                # Botão para direcionar a etapa 3
                bt_etapa2_concluida = st.button("Continuar", on_click  = update_step3)
    
    elif (st.session_state.step == 'step3'):
        st.write("**ETAPA 3: testar o melhor modelo**")
        with st.expander(":blue[**1 - Realizar predições sobre a base de dados teste (dados não presentes no treino)**]"):
            st.write("Melhor modelo em uso: ", str(st.session_state.melhor_modelo))
            
            bt_testar = st.button("Testar o modelo")
            if(bt_testar): 
                
                #melhor_modelo = eval(st.session_state.melhor_modelo).fit(st.session_state.Xtreino, st.session_state.ytreino)
                melhor_modelo = funcoes.cria_modelo(st.session_state.melhor_modelo, st.session_state.Xtreino, st.session_state.ytreino)
                predicoes_metricas = funcoes.testa_melhor_modelo(melhor_modelo, st.session_state.Xteste, st.session_state.yteste)
                yhat = predicoes_metricas["yhat"]
                # Info de sucesso
                st.success("Predições feitas com sucesso sobre a base de dados de teste")
                st.write("**Métricas de avaliação do modelo:**")
                st.write("Acurácia: ", str(predicoes_metricas["acuracia"]))
                st.write("Precisão: ", str(predicoes_metricas["precisao"]))
                st.write("Recall: ", str(predicoes_metricas["recall"]))
                st.write("f1-score: ", str(predicoes_metricas["f1"]))
                
                col_mc1, col_mc2 = st.columns(2)
                # matriz de confusão
                with col_mc1:
                    st.write("Matriz de confusão: ")
                    st.pyplot(predicoes_metricas["fig_matriz_confusao"])
                with col_mc2:
                    st.write(" ")
                
        col_bt1, col_bt2 = st.columns(2)
        with col_bt1:
            # Função para direcionar a etapa 1
            def update_step1():
                st.session_state.step = 'step1'
                # Botão para redirecionar a etapa 1
            bt_etapa1 = st.button("Recomeçar", on_click  = update_step1)
            
        with col_bt2:
            # Função para direcionar a etapa 4
            def update_step4():
                st.session_state.step = 'step4'
                # Botão para direcionar a etapa 4
            bt_etapa3_concluida = st.button("Continuar", on_click  = update_step4)   
            
    elif (st.session_state.step == 'step4'):
        st.write("**ETAPA 4: realizar predições com o melhor modelo utilizando novos dados**")            
        dataset = st.session_state.dataset
        st.write("**Selecione os valores e faça uma predição utilizando o melhor modelo selecionado:**")
        st.write("Melhor modelo em uso: ", str(st.session_state.melhor_modelo))
        if dataset == 'Iris':
            # Pega os dados do usuário
            col_n1, col_n2, col_n3, col_n4 = st.columns(4)
            with col_n1:            
                valores_sepal_length = np.arange(4.3, 7.9, 0.3)
                sepal_length = st.selectbox('sepal length (cm)', valores_sepal_length)
            with col_n2:                
                valores_sepal_width = np.arange(2 , 4.4, 0.2)
                sepal_width = st.selectbox('sepal width (cm)', valores_sepal_width)   
            with col_n3:                
                valores_petal_length  = np.arange(1, 6.9, 0.4)
                petal_length = st.selectbox('petal length (cm)', valores_petal_length)
            with col_n4:                
                valores_petal_width  = np.arange(0.1, 2.5, 0.1)
                petal_width = st.selectbox('petal width (cm)', valores_petal_width)  
            
            # Prepara os dados para a predição
            consulta = np.array([[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]])
            
            
            # Botão para realizar a predição            
            bt_prever = st.button("Realizar previsão")
            if(bt_prever):          
                # Melhor_modelo
                melhor_modelo = funcoes.cria_modelo(st.session_state.melhor_modelo, st.session_state.Xtreino, st.session_state.ytreino)
                # Realiza a predição
                predicao = melhor_modelo.predict(consulta)
                if predicao[0] == 0:
                    p = "setosa"
                elif predicao[0] == 1:
                    p = "versicolor"
                elif predicao[0] == 2:
                    p = "virginica"
                
                # Info do resultado
                st.success("Previsão realizada: " + p)
                
        
            # Função para direcionar a etapa 1
            def update_step1():
                st.session_state.step = 'step1'
            # Botão para redirecionar a etapa 1
            bt_etapa1 = st.button("Recomeçar", on_click  = update_step1)
            
        elif dataset == 'Wine':
            st.write("**Selecione os valores e faça uma predição utilizando o melhor modelo selecionado:**")
            col_n1, col_n2, col_n3, col_n4, col_n5, col_n6, col_n7 = st.columns(7)
            with col_n1:            
                valores_alcohol = np.arange(11.0, 15.0, 0.1)
                alcohol = st.selectbox('alcohol', valores_alcohol)
            with col_n2:                
                valores_malic_acid = np.arange(0.7 , 6.0, 0.1)
                malic_acid = st.selectbox('malic_acid', valores_malic_acid)   
            with col_n3:                
                valores_ash  = np.arange(1.3, 3.3, 0.1)
                ash = st.selectbox('ash', valores_ash)
            with col_n4:                
                valores_alcalinity_of_ash  = np.arange(10.0, 30.0, 0.5)
                alcalinity_of_ash = st.selectbox('alcalinity_of_ash', valores_alcalinity_of_ash)
            with col_n5:                
                valores_magnesium  = np.arange(70, 162, 5)
                magnesium = st.selectbox('magnesium', valores_magnesium)
            with col_n6:                
                valores_total_phenols  = np.arange(0.9, 3.9, 0.1)
                total_phenols = st.selectbox('total_phenols', valores_total_phenols)
            with col_n7:                
                valores_flavanoids  = np.arange(0.3, 5.1, 0.1)
                flavanoids = st.selectbox('total_phenols', valores_flavanoids)
            with col_n1:            
                valores_nonflavanoid_phenols = np.arange(0.1, 0.7, 0.01)
                nonflavanoid_phenols = st.selectbox('nonflavanoid_phenols', valores_nonflavanoid_phenols)
            with col_n2:            
                valores_proanthocyanins = np.arange(0.4, 3.6, 0.1)
                proanthocyanins = st.selectbox('proanthocyanins', valores_proanthocyanins)
            with col_n3:            
                valores_color_intensity = np.arange(1.2, 13.0, 0.3)
                color_intensity = st.selectbox('color_intensity', valores_color_intensity)
            with col_n4:            
                valores_hue = np.arange(0.4, 1.75, 0.01)
                hue = st.selectbox('hue', valores_hue)
            with col_n5:            
                valores_od280_od315_of_diluted_wines = np.arange(1.26, 4.0, 0.01)
                od280_od315_of_diluted_wines = st.selectbox('od280/od315_of_diluted_wines', valores_od280_od315_of_diluted_wines)
            with col_n6:            
                valores_proline = np.arange(278, 1680, 50)
                proline = st.selectbox('proline', valores_proline)
            
            # Prepara os dados para a predição
            consulta = np.array([[float(alcohol), float(malic_acid), float(ash), float(alcalinity_of_ash), float(magnesium), float(total_phenols), float(flavanoids),
                                float(nonflavanoid_phenols), float(proanthocyanins), float(color_intensity), float(hue), float(od280_od315_of_diluted_wines), float(proline)]])
            
            
            # Botão para realizar a predição            
            bt_prever = st.button("Realizar previsão")
            if(bt_prever):          
                # Melhor_modelo
                melhor_modelo = funcoes.cria_modelo(st.session_state.melhor_modelo, st.session_state.Xtreino, st.session_state.ytreino)
                # Realiza a predição
                predicao = melhor_modelo.predict(consulta)
                if predicao[0] == 0:
                    p = "class_0"
                elif predicao[0] == 1:
                    p = "class_1"
                elif predicao[0] == 2:
                    p = "class_2"
                
                # Info do resultado
                st.success("Previsão realizada: " + p)
                
            # Função para direcionar a etapa 1
            def update_step1():
                st.session_state.step = 'step1'
            # Botão para redirecionar a etapa 1
            bt_etapa1 = st.button("Recomeçar", on_click  = update_step1)
        
      
   
    
    
 









                                
                
            



