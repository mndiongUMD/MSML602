from flask import Flask, jsonify, url_for
import model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/analyze_forecast', methods=['GET'])
def analyze_forecast():
    # Fetch tweets and preprocess
    tweets_df = model.fetch_tweets()
    preprocessed_tweets_df = model.preprocessing_tweets(tweets_df)

    # Perform sentiment analysis on the preprocessed DataFrame
    model.analyze_sentiments(preprocessed_tweets_df)

    # Generate and save plots based on the analyzed data
    model.plot_sentiment_distribution(preprocessed_tweets_df)
    model.plot_sentiment_over_time(preprocessed_tweets_df)

    # Perform forecasting
    forecast_results = model.forecasting(preprocessed_tweets_df)

    # Construct URLs for the saved images
    distribution_url = url_for('static', filename='sentiment_distribution.png')
    over_time_url = url_for('static', filename='sentiment_over_time.png')

    # Combine the results
    results = {
        'distribution_plot': distribution_url,
        'over_time_plot': over_time_url,
        'forecast_results': forecast_results
    }

    # Return all results as JSON
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
