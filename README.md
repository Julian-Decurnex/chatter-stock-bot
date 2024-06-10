# Financial Advisor Chatbot

## Overview

This project develops a chatbot named `chatter-stock-bot`, designed to function as a financial advisor. It provides users with financial advice, market sentiment analysis, stock price updates, and news tailored to specific companies or general market conditions. The chatbot integrates multiple tools and data sources to provide accurate and timely information.

## Features

- **Financial Information Retrieval**: Offers historical financial data for companies from 2013 to 2022.
- **Market Sentiment Analysis**: Analyzes news sentiment and relevance for particular stocks.
- **Stock Price Updates**: Provides the latest stock prices.
- **Ticker Symbol Identification**: Identifies company ticker symbols based on fuzzy name matching.
- **Exchange Rate Information**: Retrieves current exchange rate data, particularly focusing on USD to ARS rates.

## Technical Stack

- **LangChain**: Powers the conversational AI capabilities.
- **Pinecone**: Manages vector storage for efficient document retrieval.
- **FAISS**: Facilitates efficient similarity search in large datasets.
- **OpenAI**: Supplies language models and embeddings.
- **Upstash Redis**: Provides conversation history management.
- **Chainlit**: Manages the chatbot application and lifecycle.
- **Yahoo Finance & Alphavantage APIs**: Supply financial data and news.

## Installation

1. **Clone the repository**:
   ```
   git clone <repository-url>
   ```
2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```
3. **Set up environment variables**:
   Include the following in your `.env` file:
   ```
   LANGCHAIN_API_KEY=<your_langchain_api_key>
   UPSTASH_REDIS_REST_TOKEN=<your_upstash_redis_rest_token>
   UPSTASH_REDIS_REST_URL=<your_upstash_redis_rest_url>
   TAVILY_API_KEY=<your_tavily_api_key>
   OPENAI_API_KEY=<your_openai_api_key>
   PINECONE_API_KEY=<your_pinecone_api_key>
   ALPHAVANTAGE_API_KEY=<your_alphavantage_api_key>
   ```

## Configuration

- **Modify the Pinecone Index Name**:
  Ensure that the Pinecone index name is set correctly in your `.env` file to match your configuration.

## Running the Application

Execute the chatbot with the following command:
```
chainlit run agent-chain.py
```

## Usage

Once the application is running, you can interact with the chatbot through the Chainlit interface. The chatbot can answer various queries about financial data, stock prices, and market sentiment.

## Contributing

Contributions are welcome! Please read the `CONTRIBUTING.md` for instructions on how to make contributions.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgements

- LangChain for the conversational AI framework.
- OpenAI for the powerful language models.
- Pinecone for the vector search engine capabilities.