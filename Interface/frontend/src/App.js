import React, { useState, useEffect } from "react";

function App() {
  const [plots, setPlots] = useState({
    distributionPlot: "",
    overTimePlot: "",
  });
  const [forecastResults, setForecastResults] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/analyze_forecast");
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const fetchedData = await response.json();
        console.log("Fetched Data:", fetchedData); // Log the fetched data
        setPlots({
          distributionPlot: fetchedData.distribution_plot,
          overTimePlot: fetchedData.over_time_plot,
        });
        setForecastResults(fetchedData.forecast_results);
      } catch (error) {
        console.error("Fetch error:", error);
        setError(error.message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (isLoading) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <p>Predicting the Sentiments of Artificial Intelligence</p>

      {/* Render the sentiment plots */}
      {plots.distributionPlot && (
        <img
          src={plots.distributionPlot}
          alt="Distribution of Sentiments in Tweets"
        />
      )}

      {plots.overTimePlot && (
        <img src={plots.overTimePlot} alt="Sentiments Over Time" />
      )}

      {/* Check if forecastResults is not empty and render the forecasted data */}
      {forecastResults && forecastResults.length > 0 && (
        <div>
          <h2>Forecasted Values</h2>
          <ul>
            {forecastResults.map((item, index) => {
              // Determine sentiment based on the value
              let sentiment;
              if (item[1] > 0) {
                sentiment = "positive";
              } else if (item[1] < 0) {
                sentiment = "negative";
              } else {
                sentiment = "neutral";
              }

              return (
                <li key={index}>
                  {`${item[0]} - ${item[1].toFixed(6)} (${sentiment})`}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
