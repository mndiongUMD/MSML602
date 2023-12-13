// Example React component

import React, { useState } from "react";

function TweetAnalyzer() {
  const [tweet, setTweet] = useState("");
  const [result, setResult] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ tweet: tweet }),
      });
      const data = await response.json();
      setResult(data.sentiment);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <textarea value={tweet} onChange={(e) => setTweet(e.target.value)} />
        <button type="submit">Analyze</button>
      </form>
      {result && <p>Sentiment: {result}</p>}
    </div>
  );
}

export default TweetAnalyzer;
