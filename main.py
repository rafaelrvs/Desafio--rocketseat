import os
from datetime import datetime
import yfinance as yf
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import streamlit as st
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Função para buscar preços de ações e resumir os dados
def fetch_stock_price(ticket):
    try:
        stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
        if stock.empty:
            raise ValueError("Nenhum dado foi encontrado para o ticket fornecido.")
        
        # Resumindo os dados para evitar ultrapassar o limite de tokens
        stock_summary = stock[['Close']].resample('M').mean()  # Média mensal do preço de fechamento
        return stock_summary
    except Exception as e:
        st.error(f"Erro ao buscar dados de ações: {e}")
        return None

# Configuração do ambiente
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("A chave da API OpenAI não foi encontrada. Defina-a na variável de ambiente 'OPENAI_API_KEY'.")

# Usando diretamente o ChatOpenAI para modelos de chat
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)

# Definindo o template do prompt
prompt_template = """
You are an experienced stock market analyst. Analyze the historical stock price of {ticket}, considering the provided data, and generate a brief report of the price trends and the influencing factors.

Stock data: {stock_data}

Output: Write a short analysis highlighting key price movements and any potential market factors that influenced these movements.

RETORNE OS DADOS EM PORTUGUES
"""

# Função para executar a análise usando o ChatOpenAI diretamente
def run_analysis(ticket):
    stock_data = fetch_stock_price(ticket)
    if stock_data is None:
        return "Análise não pôde ser realizada devido a erros ao buscar dados."
    
    # Resumindo ainda mais os dados se necessário
    stock_data_summary = stock_data.to_dict()
    
    # Preparando o prompt
    prompt = ChatPromptTemplate.from_template(prompt_template).format(ticket=ticket, stock_data=stock_data_summary)
    
    # Criando mensagens de sistema e humano
    messages = [
        SystemMessage(content="You are an experienced stock market analyst."),
        HumanMessage(content=prompt)
    ]
    
    # Executando o modelo de chat diretamente
    analysis = llm(messages)
    
    return analysis.content

# Interface do usuário com Streamlit
with st.sidebar:
    st.header('Insira o ticket da ação')

    with st.form(key='research_form'):
        topic = st.text_input("Selecione o ticket da ação")
        submit_button = st.form_submit_button(label="Executar Pesquisa")

if submit_button:
    if not topic:
        st.error("Por favor, preencha o campo do ticket.")
    else:
        st.info("Buscando dados e gerando análise, por favor aguarde...")
        analysis_result = run_analysis(topic)
        st.subheader("Resultados da sua pesquisa:")
        st.write(analysis_result)
