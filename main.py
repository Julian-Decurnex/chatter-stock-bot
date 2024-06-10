import os
from dotenv import load_dotenv

# Remove the PINECONE_API_KEY environment variable if it exists to ensure a clean state.
if 'PINECONE_API_KEY' in os.environ:
    del os.environ['PINECONE_API_KEY']

# Load environment variables from a .env file.
load_dotenv()

# Import necessary modules and classes for the chatbot functionality.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_tool_calling_agent
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools import tool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
import pandas as pd
from pandas.tseries.offsets import DateOffset
import requests
import chainlit as cl
from fuzzywuzzy import fuzz
import yfinance as yf

# Setup for Langsmith tracing.
from langsmith import Client
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "agent-chain"
client = Client()

# Retrieve Upstash Redis configuration from environment variables.
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")  # URL for the Upstash Redis REST API.
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")  # Authentication token for the Upstash Redis REST API.

# Establish dates for news comparison
date = pd.Timestamp.today() - DateOffset(days=10)
date = date.strftime('%Y%m%d')

def get_session_history(session_id: str) -> UpstashRedisChatMessageHistory:
    """Creates a chat message history instance using Upstash Redis.

    Keyword arguments:
    session_id -- ID of the chat session to retrieve history for.

    Return: An instance of UpstashRedisChatMessageHistory connected to the specified session.
    """
    return UpstashRedisChatMessageHistory(
        url=UPSTASH_REDIS_REST_URL,
        token=UPSTASH_REDIS_REST_TOKEN,
        session_id=session_id,
        ttl=0
    )

@tool("Search_Stock_News")
def query_news(symbol):
    """ Searches and returns updated (last 10 days) news dataframe for market sentiment about a specific SYMBOL/TICKER.

    Keyword arguments:
    symbol -- The financial ticker for the company to research.

    Return: The latest financial news results in the form a dataframe with the following columns:
    "title", "summary", "url", "relevance" and "sentiment".

    sentiment score definition: x <= -0.35: Bearish; -0.35 < x <= -0.15: Somewhat-Bearish; -0.15 < x < 0.15: Neutral; 0.15 <= x < 0.35: Somewhat_Bullish; x >= 0.35: Bullish"
    relevance score definition: 0 < x <= 1, with a higher score indicating higher relevance."
    """


    alphavantage_apikey = os.getenv("ALPHAVANTAGE_API_KEY")

    if symbol is None:
        url = 'https://www.alphavantage.co/query' \
              '?function=NEWS_SENTIMENT' \
              '&sort=RELEVANCE' \
              '&time_from=%sT0000' \
              '&topics=financial_markets' \
              '&limit=20' \
              '&apikey=%s' % (date, alphavantage_apikey)

    else:
        url = 'https://www.alphavantage.co/query' \
              '?function=NEWS_SENTIMENT' \
              '&sort=RELEVANCE' \
              '&time_from=%sT0000' \
              '&limit=20' \
              '&tickers=%s' \
              '&apikey=%s' % (date, symbol, alphavantage_apikey)

    if alphavantage_apikey == "demo":
        url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=demo'
        symbol = "AAPL"

    r = requests.get(url)
    data = r.json()

    df = pd.DataFrame(data['feed'])
    df = df.drop_duplicates(subset=['title'])

    def sentiment_filter(sentiments):
      return [x for x in sentiments if x['ticker'] == symbol][0]

    def split_relevance(x):
        return x['relevance_score']

    def split_sentiment(x):
        return x['ticker_sentiment_score']

    if symbol:
        df["ticker_sentiment"] = df["ticker_sentiment"].apply(sentiment_filter)
        df["relevance"] = df["ticker_sentiment"].apply(split_relevance)
        df["sentiment"] = df["ticker_sentiment"].apply(split_sentiment)
    else:
        df["relevance"] = 0.0
        df["sentiment"] = 0.0

    del df['ticker_sentiment']

    news_df = df[['title', 'summary', 'url', 'relevance', 'sentiment']]

    news_df['relevance'] = pd.to_numeric(news_df['relevance'])

    news_df = news_df[news_df['relevance'] > 0.4]

    return news_df

@tool("Find_Ticker_Symbol")
def find_ticker(word):
    """
    This function finds the closest company name or exact ticker to a given word and returns its ticker symbol.
    Use in case of uncertainty of a company name or ticker symbol

    Args:
        word (str): The word to search for in the company names or ticker symbols. This can be either a single-word company name or a ticker symbol.

    Returns:
        str: The ticker symbol of the closest company name or the exact ticker, or None if no close match is found.

    Example:
        >>> find_ticker("AAPL")
        'AAPL'
        >>> find_ticker("Apple")
        'AAPL'
    """

    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv("stock_info.csv")

    # Make all company names and tickers lowercase for case-insensitive matching
    data["Name"] = data["Name"].str.lower()
    data["Ticker"] = data["Ticker"].str.lower()
    word = word.lower()

    # First, check if the word matches any ticker exactly
    if word in data["Ticker"].values:
        return word.upper()

    # Calculate similarity scores using fuzzywuzzy
    # Select companies with a similarity score above a threshold
    threshold = 40
    data["Similarity Score"] = data["Name"].apply(lambda name: fuzz.ratio(name, word))
    data_filtered = data[data["Similarity Score"] >= threshold]

    # If no companies meet the threshold, return None
    if data_filtered.empty:
        return None

    # Get the top 5 companies with the highest similarity scores
    most_similar = data_filtered.sort_values(by="Similarity Score", ascending=False).iloc[:5]

    # Check if any of the top 5 companies contains the search word
    exact_match = most_similar[most_similar["Name"].str.contains(word)]
    if not exact_match.empty:
        return exact_match["Ticker"].iloc[0].upper()

    # If no match containing the word, return the ticker of the most similar company
    return most_similar["Ticker"].iloc[0].upper()

@tool("Get_Stock_Price")
def get_stock_price(company):
    """
    This function retrieves the latest stock price for a given company.

    Args:
        company (str): The name of the company (single word) to search for.

    Returns:
        float: The last stock price of the company, or None if the company or its ticker symbol is not found.

    Example:
        >>> get_stock_price("Nvidia")
        190.75  # (example output)
    """
    ticker = find_ticker(company)
    if ticker is None:
        return None

    company_data = yf.Ticker(ticker)
    return company_data.fast_info['lastPrice']

def initialize_new_retriever(url: str):
    """Initializes a new retriever from the provided URL by loading and splitting documents.

    Keyword arguments:
    url -- URL to load the documents from.

    Return: A retriever tool initialized with documents from the specified URL.
    """
    loader = WebBaseLoader(url)  # Load documents from the specified URL.
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)  # Split documents into chunks for embedding.
    splitdocs = splitter.split_documents(docs)
    model_name = "text-embedding-3-small"  # Model name for the OpenAI embeddings.
    embeddings = OpenAIEmbeddings(model=model_name)  # Embedding model for vector representation.
    vectorstore = FAISS.from_documents(splitdocs, embedding=embeddings)  # FAISS vector store to store and search vectors.
    retriever = vectorstore.as_retriever(search_kwargs={"10": 3})  # Retriever tool configured to return top 3 results.
    return retriever

def initialize_pinecone_retriever():
    """Initializes a Pinecone-based vector store retriever.

    Return: A retriever tool backed by a Pinecone vector store.
    """
    index_name = os.getenv('PINECONE_INDEX_NAME')  # Name of the Pinecone index.
    pinecone_api_key = os.getenv("PINECONE_API_KEY")  # API key for Pinecone.
    pc = Pinecone(api_key=pinecone_api_key)  # Initialize Pinecone with the API key.
    model_name = "text-embedding-3-small"  # Model name for the OpenAI embeddings.
    embeddings = OpenAIEmbeddings(model=model_name)  # Embedding model for vector representation.
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)  # Pinecone vector store for storing and retrieving vectors.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # Retriever tool configured to return top 3 results.

    # Initialize the chat prompt template for rephrasing follow-up questions
    prompt_retriever = ChatPromptTemplate.from_messages([
        ("system", "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    model = "gpt-3.5-turbo-0125"  # Model identifier for the OpenAI chat model.
    llm = ChatOpenAI(model=model)  # Initialize the chat model.
    # Create a history-aware retriever to handle the conversation context
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=prompt_retriever
    )
    return retriever

def initialize_chatbot(retriever, web_retriever, session_id):
    """Initializes the chatbot with specified retrievers and session ID.

    Keyword arguments:
    retriever -- The Pinecone vector store-based retriever.
    web_retriever -- The web-based retriever.
    session_id -- ID of the chat session.

    Return: An initialized AgentExecutor with chatbot functionality.
    """
    model = "gpt-3.5-turbo-0125"  # Model identifier for the OpenAI chat model.
    llm = ChatOpenAI(model=model)  # Initialize the chat model.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial advisor called chatter-stock-bot. You give precise recommendations for investing." 
         "You have 5 tools at your disposal. Use then jointly. "
         "When requested about financial information from dates ranging from 2013 to 2022, refer to 'Retriever_Financial_Search'."
         "For market sentiment, use 'Search_Stock_News' who takes in a ticker symbol referring to the Company to be researched. Use the sentiment score to determine if the market is bullish or not. If uncertain of the symbol to use, use the Find_Ticker_Symbol tool first."
         "For latest stock price, use Get_Stock_Price."
         "When unsure about a ticker symbol, use Find_Ticker_symbol, which takes in a one word company name."
         "When asked about current dollar exchange rate with the peso, use Retriever_Dollar_Search."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    history = get_session_history(session_id)  # Retrieve chat history for the session.
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        chat_memory=history
    )

    @tool("Trading_Research")
    def researcher(query: str) -> str:
        """Fetches up-to-date financial news for major companies using Yahoo Finance.

        Keyword arguments:
        query -- The financial query or company name to research.

        Return: The latest financial news results from Yahoo Finance.
        """
        Yfinance = YahooFinanceNewsTool()  # Yahoo Finance news tool for financial queries.
        return Yfinance.run(query)

    

    retriever_dollar_search = create_retriever_tool(
        retriever=web_retriever,
        name="Retriever_Dollar_Search",
        description="Use this tool when asked for information regarding current dollar price exchange value, include Blue, Oficial, and others. Specify which type of dollar."
    )
    retriever_pinecone_search = create_retriever_tool(
        retriever=retriever,
        name="Retriever_Financial_Search",
        description="Use this tool when searching for financial information from 2013 to 2022. This includes any major company."
    )

    tools = [retriever_pinecone_search, retriever_dollar_search, query_news, get_stock_price, find_ticker]  # List of tools available to the chatbot.

    agent = create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=tools
    )

    agentExecutor = AgentExecutor(
        agent=agent,
        tools=tools
    )

    runnable_with_history = RunnableWithMessageHistory(
        agentExecutor,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    return runnable_with_history

def process_chat(agentExecutor, user_input, session_id):
    """Processes a chat input and returns the agent's response.

    Keyword arguments:
    agentExecutor -- The executor managing the agent's responses.
    user_input -- The user's input string to be processed.
    session_id -- The current chat session ID.

    Return: The generated response from the agent.
    """
    response = agentExecutor.invoke(
        {"input": user_input}, 
        config={"configurable": {"session_id": session_id}},
    )
    return response["output"]

# @cl.set_starters
# async def starters():
#     return [
#         cl.Starter(
#             label="Company Financial Overview",
#             message="Can you provide a financial overview of Apple from 2019 to 2021, including key metrics from their financial statements?",
#             #icon="/icons/finance-history.svg",
#             ),
#         cl.Starter(
#             label="Current Market Information",
#             message="What is the latest stock price of Tesla?",
#             #icon="/icons/stock.svg",
#             ),
#         cl.Starter(
#             label="Exchange Rate Inquiry",
#             message="What is the current exchange rate between the US dollar and the Argentine peso?",
#             #icon="/icons/dollar.svg",
#             ),
#         cl.Starter(
#             label="Company News and Analysis",
#             message="Tell me the latest news about Microsoft and how it might impact their stock price. Should I buy or sell?",
#             #icon="/icons/market-info.svg",
#             )
#         ]

@cl.on_chat_start
async def on_chat_start():
    """Function to initialize chat session including retrievers and chatbot when a chat session starts."""
    url = "https://bluedollar.net/"  # URL for web-based retriever initialization.
    web_retriever = initialize_new_retriever(url)  # Initialize a new web retriever.
    retriever = initialize_pinecone_retriever()  # Initialize a Pinecone retriever.
    session_id = "chat-3"  # Session ID for the chat session.
    agentExecutor = initialize_chatbot(retriever, web_retriever, session_id)  # Initialize the chatbot.
    cl.user_session.set("agentExecutor", agentExecutor)  # Store the agentExecutor in the session.
    cl.user_session.set("session_id", session_id)  # Store the session ID in the session.

@cl.on_message
async def on_message(message: cl.Message):
    """Handles incoming messages by processing them and sending responses.

    Keyword arguments:
    message -- The incoming chat message to be processed.

    Return: None.
    """
    agentExecutor = cl.user_session.get("agentExecutor")  # Retrieve the agentExecutor from the session.
    session_id = cl.user_session.get("session_id")  # Retrieve the session ID from the session.
    response = process_chat(agentExecutor, message.content, session_id)  # Process the user's message.
    await cl.Message(content=response).send()  # Send the response back to the user.

if __name__ == '__main__':
    cl.run()

"""Explanation
Initialization Functions: These functions remain unchanged. They initialize the retrievers and the chatbot.
Langsmith is used for trace viewing. ( I suggest creating a Langsmith API key)
Upstash is used for conversation memory based on session id. (Free & Serverless)


.env file content:
LANGCHAIN_API_KEY
UPSTASH_REDIS_REST_TOKEN
UPSTASH_REDIS_REST_URL
TAVILY_API_KEY
OPENAI_API_KEY
PINECONE_API_KEY


Current tools for the agent are:
- Tavily search (not used) 
- Yahoo finance (for updated news)
- pinecone retriever 
- url retriever (example url is a simple date and time page just to see its integration, for example, what is the current date?)


Chainlit Event Handlers:

@cl.on_chat_start: This function initializes the retrievers and the chatbot when a chat session starts. It sets the agentExecutor and session_id in the user session.
@cl.on_message: This function processes incoming messages. It retrieves the agentExecutor and session_id from the user session, processes the user's input, and sends the response back to the user.
Running the Application: The if __name__ == '__main__': block runs the Chainlit application.

To run:
modify pinecone index_name
chainlit run agent-chain.py


To do's:
- Add error handling across the chatbot.
- Enhance the system prompt.
- Add tools.
- Combine multiple tools (may be included in prompt I think).
- Enhance chainlit.md 
- 
"""