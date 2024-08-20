import json
import os
from datetime import datetime
import yfinance as yf
from langchain.tools import Tool
from langchain_openai import ChatOpenAI  # Corrigido aqui
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Função para buscar preços de ações
def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock

# Configuração da ferramenta de busca de preços
yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="Fetches stock prices for {ticket} from the last year from Yahoo Finance API",
    func=lambda ticket: fetch_stock_price(ticket)
)

# Configuração do ambiente
os.environ['OPENAI_API_KEY'] = ""
llm = ChatOpenAI(model="gpt-3.5-turbo")  # Corrigido aqui

# Configuração da ferramenta de busca de notícias
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

# Definindo o template do prompt
prompt_template = """
You are an experienced stock market analyst. Analyze the historical stock price of {ticket}, considering the provided data, and generate a brief report of the price trends and the influencing factors.

Stock data: {stock_data}

Output: Write a short analysis highlighting key price movements and any potential market factors that influenced these movements.
"""

# Criando a cadeia de análise de preço de ações
stock_price_analysis_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template),
    verbose=True
)

# Executando a análise
ticket = "AAPL"
stock_data = fetch_stock_price(ticket)
analysis = stock_price_analysis_chain.run({"ticket": ticket, "stock_data": stock_data})

print(analysis)
