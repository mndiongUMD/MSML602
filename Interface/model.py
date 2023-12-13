#!/usr/bin/env python
# coding: utf-8

# In this project, we are predicting trends in technology adoption and interest based on social media (Twitter) data. Specifically, the model aims to forecast the following:
# 
# 1. **Volume of Discussions**: Predicting the number of tweets or social media posts related to specific technologies, gadgets, or software within a given time frame in the future (e.g., daily, weekly). This serves as an indicator of public interest and awareness levels.
# 
# 2. **Sentiment Trends**: Forecasting the overall sentiment (positive, negative, neutral) associated with these technologies in the social media discourse. This could involve predicting the average sentiment score or the proportion of tweets falling into each sentiment category for upcoming days.
# 
# 3. **Combination of Volume and Sentiment**: A more comprehensive approach might involve predicting both the volume of discussion and the sentiment concurrently. This dual prediction can provide a more nuanced understanding of how public interest and perception might evolve over time.
# 
# ### Example Predictions
# - **Before a Product Launch**: If there's an upcoming release of a new gadget, the model might predict an increase in the volume of discussion and potentially the sentiment trend leading up to and following the launch.
# - **Emerging Technology Trends**: For emerging tech like augmented reality, blockchain, or new software platforms, the model could forecast how discussions (both in volume and sentiment) about these technologies will trend in the short-term future.
# 
# ### Purpose of These Predictions
# - **Market Insight**: These predictions can provide valuable insights for businesses, marketers, and technologists about consumer interest and sentiment trends, aiding in strategic planning and decision-making.
# - **Product Strategy**: For tech companies, understanding how public interest and sentiment are likely to shift can inform product development, marketing strategies, and customer engagement plans.
# - **Investment Decisions**: Investors in technology sectors might use these predictions to gauge potential market reactions to new technologies or products.
# 
# The predictions, therefore, are not just about the raw data but also about interpreting the data to extract meaningful trends and insights that can inform various strategic decisions in the technology domain.

# In[1]:


# Essential imports
import pandas as pd
import numpy as np
import tweepy
import nltk
import sqlite3
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

# ## 1. Data Collection
# Sources: Gather data from social media. We will be using Twitter API to search and get tweets with relevant keywords
# 
# Keywords: Identify relevant keywords for each technology (e.g., "artificial intelligence", "augmented reality", "blockchain").

# In[2]:


# Twitter API keys

# Consumer Keys
# MSML apis
api_key = 'fQyQfxNjgLk8NDoVt339h8K0g'
api_secret_key = '5wHUc4mrkVn1R9pR7tVaNXkKuB6Le1qIpSqKA3nb9H70rEVqiz'

# Authentication Tokens
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAJu0rQEAAAAAgyq9jSqX7eeOOvgs2RmwfRHVzgE%3Djo5sGF0vv7XmtfYAgK71rOh70224Z0dmPXFvOrXjekfqJ8XnY3'
access_token = '2931998159-ngeYrsqwmVvs1jYjpZcCFBzO2xm0j2wsqokBLK6'
access_token_secret = 'CGo43zg5cX2KDdyACKDVIUtrULMV1SCBjPVNogCW1UKKs'

# Authenticate
auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

client = tweepy.Client(bearer_token=bearer_token, 
              consumer_key=api_key, 
              consumer_secret=api_secret_key, 
              access_token=access_token, 
              access_token_secret=access_token_secret)


# In[3]:

def fetch_tweets():
    # Getting the tweets from twitter
    query = 'artificial intelligence -is:retweet'
    tweets = client.search_recent_tweets(query=query,
                                    tweet_fields=['context_annotations', 'created_at'],
                                    max_results=100)

    # Extract data from the response
    data = [{'Tweet': tweet.text, 'Timestamp': tweet.created_at} for tweet in tweets.data]

    # Create a DataFrame
    tweets_df = pd.DataFrame(data)

    # Display the DataFrame
    return tweets_df


# In[4]:

def add_tweets_to_database(tweets_df):
    # Create a SQLite database connection
    conn = sqlite3.connect('tweets_database.db')

    # Write the DataFrame to a SQLite table
    tweets_df.to_sql('tweets', conn, if_exists='replace', index=False)

    # Optionally, read the table back from the database to verify
    tweets_df_from_sql = pd.read_sql('SELECT * FROM tweets', conn)

    # Display the DataFrame read from the database
    print(tweets_df_from_sql)

    # Close the database connection
    conn.close()
    
# In[5]:


# For fetching the data in the database later
def get_tweets_by_query():
    conn = sqlite3.connect('tweets_database.db')
    cur = conn.cursor()

    # Select tweets that match the query
    cur.execute("SELECT Tweet FROM tweets")
    all_tweets = cur.fetchall()
    
    print(all_tweets)

    conn.close()
    return all_tweets

## 2. Data Preprocessing

# In[6]:

# Define the tweet cleaning function
def clean_text(text):
    """Remove URLs, hashtags, mentions, and special characters."""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # Remove mentions and hashtags
    text = re.sub(r"[^a-zA-Z']", " ", text)  # Keep only alphabets and apostrophes
    return text

def tokenize(text):
    """Tokenize the text into words."""
    return word_tokenize(text)

def remove_stop_words(tokens):
    """Remove stop words from the list of tokens."""
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def lemmatize(tokens):
    """Lemmatize the tokens (words)."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def drop_irrelevant_cols(tweets_df):
    """Drop irrelevant columns from the DataFrame."""
    relevant_cols = ['Tweet', 'Timestamp']  # Assuming these are the relevant columns
    return tweets_df[relevant_cols]

def filter_rows(tweets_df):
    """Filter rows based on the presence of specific keywords."""
    query_words = ['Artificial Intelligence', 'ai']
    mask = tweets_df['Tweet'].str.contains('|'.join(query_words), case=False, na=False)
    return tweets_df[mask]

def preprocess_tweet(tweet):
    """Preprocess a single tweet text."""
    tweet = tweet.lower()  # Convert to lowercase
    tweet = clean_text(tweet)
    tokens = tokenize(tweet)
    tokens = remove_stop_words(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)

def preprocessing_tweets(tweets_df):
    """Complete preprocessing pipeline for tweets."""
    tweets_df = drop_irrelevant_cols(tweets_df)
    tweets_df = filter_rows(tweets_df)
    tweets_df['Cleaned_Tweet'] = tweets_df['Tweet'].apply(preprocess_tweet)
    return tweets_df

## 3. Sentiment Analysis
# Sentiment Detection Tool: Use pre-built libraries like TextBlob.
# Classification: Classify the sentiment of each piece of text as positive, negative, or neutral.

# In[9]:

# Function to apply sentiment analysis
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'positive', polarity
    elif polarity == 0:
        return 'neutral', polarity
    else:
        return 'negative', polarity

def analyze_sentiments(tweets_df):
    # Apply the function to each tweet
    tweets_df['Sentiment'], tweets_df['Polarity'] = zip(*tweets_df['Cleaned_Tweet'].apply(analyze_sentiment))

# In[10]:

def plot_sentiment_distribution(tweets_df, filename='static/sentiment_distribution.png'):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Sentiment', data=tweets_df)
    plt.title('Sentiment Distribution in Tweets')
    plt.savefig(filename)
    plt.close()

def plot_sentiment_over_time(tweets_df, filename='static/sentiment_over_time.png'):
    plt.figure(figsize=(10, 6))
    tweets_df.groupby('Timestamp')['Sentiment'].value_counts().unstack().plot(kind='line', marker='o')
    plt.title('Sentiment Over Time')
    plt.ylabel('Number of Tweets')
    plt.xlabel('Date')
    plt.savefig(filename)
    plt.close()

# In[12]:

def cumulative_sentiment_trends(tweets_df):
    # Convert 'Timestamp' to datetime if not already
    tweets_df['Timestamp'] = pd.to_datetime(tweets_df['Timestamp'])

    # Filter tweets from 2022 to present date
    tweets_df_filtered = tweets_df[tweets_df['Timestamp'].dt.year >= 2022]

    # Ensure 'Sentiment' column exists and contains categorical data in the filtered DataFrame
    if 'Sentiment' in tweets_df_filtered.columns and tweets_df_filtered['Sentiment'].isin(['positive', 'neutral', 'negative']).all():
        plt.figure(figsize=(10, 6))

        # Filter and group by Timestamp and Sentiment, then calculate cumulative sum
        cumulative_sentiment = tweets_df_filtered.groupby(['Timestamp', 'Sentiment']).size().unstack().cumsum()

        # Plot cumulative sentiment trends
        cumulative_sentiment.plot(figsize=(10, 6), marker='o')
        plt.title('Cumulative Sentiment Trends (2022 to Present)')
        plt.ylabel('Cumulative Number of Tweets')
        plt.xlabel('Date')
        plt.legend(title='Sentiment')
        plt.show()
    else:
        print("Error: 'Sentiment' column not found or does not contain categorical data.")


# In[13]:

def word_cloud(tweets_df):
    # Word Cloud (for positive sentiment tweets as an example)
    positive_tweets = ' '.join(tweets_df[tweets_df['Sentiment'] == 'positive']['Cleaned_Tweet'])
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(positive_tweets)
    
    if tweets_df['Sentiment'].dtype == 'object':
        tweets_df['Sentiment'] = tweets_df['Sentiment'].astype('category')

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Positive Sentiment Tweets')
    plt.show()

# In[14]:

# ## 4. Time Series Analysis
# In[15]:

def time_series_analysis():
    # Convert 'Timestamp' to datetime and set as index
    tweets_df['Timestamp'] = pd.to_datetime(tweets_df['Timestamp'])

    # Set 'Timestamp' as the index
    tweets_df.set_index('Timestamp', inplace=True)

    # Resample and aggregate sentiment scores
    # calculating daily mean sentiment
    daily_sentiment = tweets_df['Polarity'].resample('D').mean()

    # Resample sentiment scores on a weekly basis (calculating mean polarity)
    weekly_sentiment = tweets_df['Polarity'].resample('W').mean()

    # Plot daily sentiment
    plt.figure(figsize=(12, 6))
    daily_sentiment.plot()
    plt.title('Daily Average Sentiment Polarity')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Polarity')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plotting weekly sentiment
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_sentiment.index, weekly_sentiment.values, marker='o', linestyle='-', color='orange', label='Weekly')
    plt.title('Weekly Average Sentiment Polarity Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Polarity')
    plt.grid(True)
    plt.legend()
    plt.show()

# In[16]:
def decomposition_and_trend_analysis():
    # Decompose the time series data
    decomposition = seasonal_decompose(tweets_df['Polarity'], period=30)  # Adjust period for seasonality

    # Plot decomposed components
    plt.figure(figsize=(12, 8))
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(tweets_df.index, tweets_df['Polarity'], label='Original Data')
    plt.legend()

    plt.subplot(412)
    plt.plot(tweets_df.index, trend, label='Trend')
    plt.legend()

    plt.subplot(413)
    plt.plot(tweets_df.index, seasonal, label='Seasonal')
    plt.legend()

    plt.subplot(414)
    plt.plot(tweets_df.index, residual, label='Residuals')
    plt.legend()

    plt.tight_layout()
    plt.show()

def calculating_rolling_mean_std():
    # Reset index to convert 'Timestamp' back to a column
    tweets_df.reset_index(inplace=True)

    # Convert 'Timestamp' to datetime if not already
    tweets_df['Timestamp'] = pd.to_datetime(tweets_df['Timestamp'])

    # Set 'Timestamp' as the index
    tweets_df.set_index('Timestamp', inplace=True)

    # Calculate rolling mean and rolling standard deviation
    rolling_mean = tweets_df['Polarity'].rolling(window=30).mean()  # Adjust window size as needed
    rolling_std = tweets_df['Polarity'].rolling(window=30).std()  # Adjust window size as needed

    # Plot original data with rolling mean and rolling standard deviation
    plt.figure(figsize=(12, 6))
    plt.plot(tweets_df.index, tweets_df['Polarity'], label='Original Data', color='blue')
    plt.plot(tweets_df.index, rolling_mean, label='Rolling Mean (30-day)', color='red')
    plt.plot(tweets_df.index, rolling_std, label='Rolling Std (30-day)', color='green')
    plt.title('Trend Analysis: Original Data with Rolling Statistics')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Polarity')
    plt.legend()
    plt.grid(True)
    plt.show()

## 5. Forecasting
def forecasting(tweets_df):
    # Reset index to convert 'Timestamp' back to a column
    tweets_df.reset_index(inplace=True)
    
    # Split the data into training and testing sets (adjust according to your data)
    train_size = int(len(tweets_df) * 0.8)
    train_data, test_data = tweets_df[:train_size], tweets_df[train_size:]

    # Fit ARIMA model (adjust p, d, q values)
    p, d, q = 5, 1, 0  # Example values, tune based on autocorrelation analysis
    model = ARIMA(train_data['Polarity'], order=(p, d, q))
    arima_model = model.fit()

    # Forecast future values
    forecast_values = arima_model.forecast(steps=len(test_data))  # Adjust steps for forecasting horizon

    # Evaluation (calculate RMSE)
    mse = mean_squared_error(test_data['Polarity'], forecast_values)
    rmse = np.sqrt(mse)

    # Plotting actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(test_data['Timestamp'], test_data['Polarity'], label='Actual')
    plt.plot(test_data['Timestamp'], forecast_values, label='Predicted', color='red')
    plt.title('ARIMA Forecasting: Actual vs Predicted Sentiment')
    plt.xlabel('Timestamp')
    plt.ylabel('Polarity')
    plt.legend()
    plt.show()

    # Convert forecasted values to a list and prepare a response dictionary

    # Instead of plt.show(), save the plot and return its URL if necessary
    filename = 'forecast_plot.png'
    plt.savefig('static/' + filename)
    plt.close()

     # Pair each forecasted value with its corresponding timestamp
    forecasted_values_with_timestamps = list(zip(test_data['Timestamp'], forecast_values))

    return forecasted_values_with_timestamps

## 6. Visualization
def visualiation():
    # Assuming 'tweets_df' contains your time series data with a 'Timestamp' column and 'Polarity' values
    # Convert 'Timestamp' to datetime if not already
    tweets_df['Timestamp'] = pd.to_datetime(tweets_df['Timestamp'])

    # Set 'Timestamp' as the index
    tweets_df.set_index('Timestamp', inplace=True)

    # Ensure the index is monotonic and has a frequency (e.g., 'D' for daily, 'M' for monthly, etc.)
    # Replace 'D' with the appropriate frequency for your data
    tweets_df.index = pd.date_range(start=tweets_df.index[0], periods=len(tweets_df), freq='D')

    # Lambda function for data loading
    load_data = lambda: pd.DataFrame({
                            'Timestamp': pd.date_range(start='2023-01-01', periods=100, freq='D'),
                            'Sentiment': np.random.randint(-1, 2, 100)
                        })

    data = load_data()

    # Plotting with Plotly
    fig = px.line(data, x='Timestamp', y='Sentiment', title='Sentiment Trend')
    fig.show()