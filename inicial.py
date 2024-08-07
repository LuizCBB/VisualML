import streamlit as st

def carrega_inicial():
    # Logo
    col_logo1, col_logo2, col_logo3 = st.columns(3)
    with col_logo1:
        st.write(' ')
    with col_logo2:
        st.image("logo_visualML2.jpg")
    with col_logo3:
        st.write(' ')
        
    # CSS
    st.markdown(""" <style>
        .div, p {
          text-align: justify;
          height: float;
          max-width: 2000px; 
          border-radius: 10px;
          padding-left: 10px;
        }
    </style> """, unsafe_allow_html=True)
        
    st.markdown("""<div><p>Atualmente, o aprendizado de máquina (<i>Machine Learning</i> - ML) está presente em variados dispositivos e serviços que fazem parte da vida humana, como serviços de recomendação, diagnóstico
    de cuidados de saúde, veículos autônomos, ambientes virtuais de aprendizagem, dentre outros. Assim, investir no ensino de aprendizado de máquina é essencial para
    qualquer país que deseja se destacar no cenário global e preparar a sociedade para os desafios do futuro. O aprendizado de máquina é uma das áreas mais dinâmicas da
    inteligência artificial, impulsionando a inovação em setores como saúde, finanças, agricultura e manufatura. Ao investir na educação e formação de profissionais capacitados
    em aprendizado de máquina, um país pode liderar o desenvolvimento de tecnologias avançadas, criando soluções inovadoras para problemas complexos e melhorando a eficiência
    em diversos setores. O desenvolvimento de aplicações habilitadas para o aprendizado de máquina, entretanto, não é trivial, requerendo a compreensão de algoritmos e processos
    de trabalho complexos, exigindo que o estudante tenha certo nível de habilidades de programação e fique imerso em uma gama cada vez maior de arquiteturas e <i>frameworks</i>.
    Em consequência disso, a aprendizagem de alunos iniciantes nesta área é uma tarefa difícil. Para popularizar o aprendizado de máquina é desejável reduzir o esforço cognitivo 
    de alunos iniciantes. Neste contexto, desenvolvemos o <b><i>VisualML</i></b>, uma ferramenta interativa baseada em fluxo de dados para o estudo de conceitos de aprendizado de máquina.
    </p></div>""", unsafe_allow_html=True)
    
    with st.expander(":blue[**Relação entre Inteligência artificial, aprendizado de máquina e aprendizado profundo**]"):
        st.markdown("""<div><p>A Inteligência Artificial (IA) é uma área que se concentra na criação de sistemas capazes de realizar tarefas que normalmente requerem inteligência 
        humana. A IA pode ser vista como a capacidade de uma máquina imitar funções cognitivas humanas, como aprendizado e resolução de problemas. O Aprendizado de Máquina 
        (<i>Machine Learning</i> - ML) é um subcampo da IA que se concentra no desenvolvimento de algoritmos e modelos que permitem que os computadores aprendam a partir de dados.
        Em vez de serem explicitamente programados para realizar uma tarefa, os sistemas de aprendizado de máquina usam dados para treinar modelos que podem fazer previsões ou 
        tomar decisões. O Aprendizado Profundo (<i>Deep Learning</i> - DL) é uma subcategoria do aprendizado de máquina que utiliza redes neurais artificiais com muitas camadas
        para modelar padrões complexos em grandes volumes de dados. É comum utilizar-se a expressão “aprendizado de máquina clássico” para referir-se às abordagens não
        baseadas em aprendizado profundo. Embora a popularidade do aprendizado profundo tenha aumentado significativamente nos últimos anos, impulsionada principalmente após 
        o desenvolvimento dos grandes modelos de linguagem (<i>Large Language Models</i> - LLMs) como BERT e GPT, o aprendizado de máquina clássico (incluindo algoritmos como
        árvores de decisão, regressão logística, K-vizinhos mais próximos, redes neurais artificiais, e muitos outros) ainda é muito importante em diferentes áreas e contextos.
        A IA, o aprendizado de máquina e o aprendizado profundo são áreas interconectadas que, juntas, estão transformando a maneira como interagimos com a 
        tecnologia.""", unsafe_allow_html=True)
            
    with st.expander(":blue[**Aprendizado supervisionado**]"):
        st.markdown("""<div><p>Em desenvolvimento</p></div>""", unsafe_allow_html=True)
    with st.expander(":blue[**Aprendizado não supervisionado**]"):
        st.markdown("""<div><p>Em desenvolvimento</p></div>""", unsafe_allow_html=True)
    
  
    
    