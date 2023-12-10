const express = require("express");
const router = express.Router();
const app = express();
const cors = require("cors");

app.use(cors());

app.get("/", (req, res) => {
  res.send("Hello World!");
});

const port = 3000;
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}/`);
});

app.use(express.static("public"));

app.post("/predict", (req, res) => {
  // Process the input using your ML model
  // Return the prediction
});

router.get("/climate-data", async (req, res) => {
  try {
    const data = await getClimateData(); // Replace with your method to get data
    res.json(data);
  } catch (error) {
    res.status(500).send(error.message);
  }
});

module.exports = router;
