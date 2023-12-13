function SentimentPlots({ plots }) {
  return (
    <div>
      {plots.distributionPlot && (
        <img
          src={plots.distributionPlot}
          alt="Distribution of Sentiments in Tweets"
        />
      )}
      {plots.overTimePlot && (
        <img src={plots.overTimePlot} alt="Sentiments Over Time" />
      )}
    </div>
  );
}

export default SentimentPlots;
