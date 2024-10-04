# streamlit_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
import plotly.express as px
import os
from datetime import datetime, timedelta

from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import requests  # For API calls

def main():
    st.title("Stock Analysis App with GPT-4 and LangChain")

    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        return

    # User inputs
    ticker_input = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, BHP, RELIANCE)", "AAPL")
    exchange = st.selectbox(
        "Select Exchange",
        ["US", "ASX", "India (NSE)", "India (BSE)", "Singapore", "Hong Kong", "China (Shanghai)", "China (Shenzhen)"]
    )

    # Exchange suffix mapping
    exchange_suffixes = {
        "US": "",
        "ASX": ".AX",
        "India (NSE)": ".NS",
        "India (BSE)": ".BO",
        "Singapore": ".SI",
        "Hong Kong": ".HK",
        "China (Shanghai)": ".SS",
        "China (Shenzhen)": ".SZ"
    }

    suffix = exchange_suffixes[exchange]
    ticker = ticker_input.upper() + suffix

    # Retrieve the company's full name
    try:
        ticker_object = yf.Ticker(ticker)
        company_info = ticker_object.info
        company_name = company_info.get('longName') or company_info.get('shortName') or ticker_input
    except Exception as e:
        company_name = ticker_input  # Default to ticker_input if company name not found

    # Date range selection
    st.subheader("Select Time Period for Analysis")
    today = datetime.today()
    default_start = today - timedelta(days=365)
    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", today)

    if start_date > end_date:
        st.error("Error: End date must fall after start date.")
        return

    if ticker:
        # Fetch stock data for the selected time period
        data = fetch_stock_data(ticker, start_date, end_date)
        if data.empty:
            st.error("No data found for the given ticker and date range.")
            return

        st.subheader(f"Closing Price of {ticker}")
        st.line_chart(data['Close'])

        # Calculate technical indicators
        data = calculate_technical_indicators(data)

        # Exclude original columns to get only technical indicators
        original_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        indicator_columns = [col for col in data.columns if col not in original_columns]

        # Display technical indicators
        st.subheader("Technical Indicators")
        st.dataframe(data[indicator_columns])

        # Plot selected indicators
        st.subheader("Technical Indicator Plots")

        # Select indicators to plot
        selected_indicators = st.multiselect(
            "Select Technical Indicators to Plot",
            options=['Close'] + indicator_columns,  # Include 'Close' price
            default=['Close', 'trend_sma_fast', 'trend_sma_slow']
        )

        # Plot the selected indicators over the selected time period
        if selected_indicators:
            fig = px.line(data.reset_index(), x='Date', y=selected_indicators)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one indicator to plot.")

        # Fetch internet buzz (sentiment analysis)
        st.subheader("Internet Buzz")
        news = get_news(company_name)
        if news:
            for article in news:
                st.write(f"**[{article['title']}]({article['url']})**")
                st.write(article['description'])
        else:
            st.write("No news articles found.")

        # Prepare data for GPT-4 analysis
        latest_data = data.tail(1).to_dict('records')[0]

        # Use selected indicators for recommendation
        technicals_for_recommendation = {k: v for k, v in latest_data.items() if k in selected_indicators and k != 'Close'}
        news_descriptions = " ".join([article['description'] for article in news])

        # Generate recommendation using GPT-4 via LangChain
        st.subheader("Recommendation")
        with st.spinner("Generating recommendation..."):
            recommendation = get_recommendation_gpt4(ticker_input, exchange, technicals_for_recommendation, news_descriptions)
        st.write(f"Our recommendation is:\n\n{recommendation}")

        # Sidebar for fetching indicator descriptions
        st.sidebar.subheader("Learn About Technical Indicators")
        indicator_to_describe = st.sidebar.selectbox(
            "Select an Indicator to Get Its Description",
            options=indicator_columns
        )

        if st.sidebar.button("Get Description"):
            with st.spinner("Fetching description..."):
                description = get_indicator_description(indicator_to_describe)
            st.sidebar.markdown(f"### Description of {indicator_to_describe}")
            st.sidebar.markdown(description)

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def calculate_technical_indicators(data):
    data = dropna(data)
    data = add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume")
    return data

def get_news(company_name):
    # Load environment variables from .env file
    news_api_key = os.getenv("NEWS_API_KEY")

    if not news_api_key:
        st.error("News API key not found. Please set NEWS_API_KEY in your .env file.")
        return []

    # Adjust the query to include the company name in quotes for exact matching
    query = f'"{company_name}"'

    # Use NewsAPI to fetch news articles
    url = (
        'https://newsapi.org/v2/everything?'
        f'qInTitle={query}&'
        'sortBy=publishedAt&'
        'language=en&'
        'pageSize=10&'
        f'apiKey={news_api_key}'
    )

    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        # Filter articles that mention the company name
        news = [
            {
                'title': article['title'],
                'description': article['description'] or '',
                'url': article['url']
            }
            for article in articles
            if company_name.lower() in (article['title'] + article['description']).lower()
        ]
        return news
    else:
        st.error(f"Error fetching news: {response.status_code}")
        return []

def get_recommendation_gpt4(ticker_input, exchange, technicals, news):
    # Load environment variables from .env file
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        return "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file."

    # Initialize the OpenAI LLM with GPT-4 via LangChain
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4", openai_api_key=openai_api_key)

    # Prepare the prompt
    ticker = f"{ticker_input} ({exchange})"
    prompt_template = """
You are a financial analyst. Based on the following technical indicators and news sentiment for {ticker}, provide a recommendation (Buy, Sell, Hold) with a brief explanation.

Technical Indicators:
{technicals}

News Sentiment:
{news}

Recommendation:
"""

    prompt = PromptTemplate(
        input_variables=["ticker", "technicals", "news"],
        template=prompt_template,
    )

    # Format the prompt
    final_prompt = prompt.format(
        ticker=ticker,
        technicals=technicals,
        news=news
    )

    # Create a LangChain message object
    from langchain.schema import HumanMessage
    message = HumanMessage(content=final_prompt)

    # Get the recommendation from GPT-4 using 'invoke' method
    try:
        response = llm.invoke([message])
        recommendation = response.content.strip()
        return recommendation
    except Exception as e:
        return f"An error occurred while fetching the recommendation: {e}"

@st.cache_data(show_spinner=True)
def get_indicator_description(indicator_name):
    # Load environment variables from .env file
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        return "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file."

    # Initialize the OpenAI LLM with GPT-4 via LangChain
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-4", openai_api_key=openai_api_key)

    # Prepare the prompt
    prompt = f"""
Provide a detailed explanation of the technical indicator '{indicator_name}' used in stock market analysis. Include:

- A description of what the indicator measures.
- The mathematical formula or calculation steps involved.
- How traders typically interpret this indicator.
- Any considerations or limitations when using this indicator.

Please present the information in clear, concise paragraphs with appropriate headings if necessary.
"""
    # Create a LangChain message object
    from langchain.schema import HumanMessage
    message = HumanMessage(content=prompt)
    # Get the description from GPT-4
    try:
        response = llm.invoke([message])
        description = response.content.strip()
        return description
    except Exception as e:
        return f"An error occurred while fetching the description: {e}"

if __name__ == "__main__":
    main()
