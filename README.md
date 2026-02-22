<p align="center">
  <img src="assets/FRAUD.png" alt="Project Banner" width="100%">
</p>

![Python version](https://img.shields.io/badge/Python%20version-3.10%2B-lightgrey)
![GitHub repo size](https://img.shields.io/github/repo-size/coder-red/Financial-volatility-forecasting)
![GitHub last commit](https://img.shields.io/github/last-commit/coder-red/Financial-volatility-forecasting)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=chainlink&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-000000?style=flat&logo=graph-dot-org&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-f55036?style=flat&logo=speedtest&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2ECC71?style=flat&logo=anaconda&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)


# Key findings: 

**Key findings: Transactions within high-risk categories such as online shopping and groceries, as well as those occurring during late-night hours (22:00–03:00), were significantly more likely to be fraudulent.**


## Author

- [@coder-red](https://www.github.com/coder-red)

## Table of Contents

  - [Business context](#business-context)
  - [Data source](#data-source)
  - [Methods](#methods)
  - [Tech Stack](#tech-stack)
  - [Quick glance at the results](#quick-glance-at-the-results)
  - [Lessons learned and recommendation](#lessons-learned-and-recommendation)
  - [Limitation and what can be improved](#limitation-and-what-can-be-improved)
  - [Repository structure](#repository-structure)


## Business context
This model identifies fraudulent credit card transactions in real time by combining XGBoost machine learning with an Agentic LLM investigator. By integrating LangGraph to handle "grey area" cases, the system is designed to catch sophisticated fraud patterns—like late-night online shopping bursts—that traditional rule-based systems often miss. Risk operations teams and fintech institutions use this to automate high-volume triage, reduce financial loss from chargebacks, and provide a "human-in-the-loop" layer for complex investigative decisions

## Data source

- https://www.kaggle.com/datasets/kartik2112/fraud-detection

## Methods

- **Sentiment Engineering (NLP):** Extracted contextual sentiment from financial headlines using FinBERT. The model was optimized via ONNX to reduce inference latency, allowing for rapid processing of large-scale historical RSS archives.
- **Feature engineering and sentiment extraction:** Integrated RSS feeds and Google Trends as exogenous variables, applied NLP based sentiment scoring to quantify market "panic"
- **Benchmarking: Evaluated three distinct models:** GARCH(1,1), EGARCH-X, and XGBoost.
- **Chronological Train/Test Split:** Used a fixed  split i.e 80% train / 20% test to preserve the time dependent structure of the market data and prevent random shuffling.
- **Look-ahead Bias Prevention:** All exogenous features (Sentiments and trends) were lagged to ensure predictions rely strictly on information available at the time of the forecast.


## Tech Stack

- **Python:** Core logic (refer to requirement.txt for the packages used in this project)
- **FinBERT + ONNX Runtime:** Leveraged FinBERT (specialized BERT for finance) exported to ONNX format for high-speed sentiment inference on RSS and news data.
- **Scikit-learn and XGBoost:** machine learning & evaluation
- **Arch Library:** Used for GARCH(1,1) as the baseline and EGARCH-X for modelling with exogenous inputs
- **NLP & APIs:** yfinance for market data; Bloomberg, cnbc, ft, and wall street journal, Google Trends for exogenous inputs



## Quick glance at the results

Target distribution between the features.

![Bar chart](assets/target.png)

Confusion matrix.

![Confusion Matrix](assets/confusion.png)

Feature importance.

![Bar Chart](assets/features.png)

- ***Metrics used: rmse, mae, R²***


### Model Evaluation Strategy

**Primary Metric: RMSE (Root Mean Squared Error)**
Volatility forecasting requires precise predictions since small errors can compound in risk calculations.RMSE penalizes large forecast errors more heavily than MAE, making it best for identifying models that avoid dangerous outliers in volatility estimates.


**Supporting Metrics: MAE (Mean Absolute Error), R²**
- **MAE** shows the model's average size of forecasting error.
- **R²** indicates how much variation in volatility the model explains.


## Lessons Learned and Recommendations

**What I found:**

- **EGARCH-X vs. XGBoost Performance:** While XGBoost is better at capturing non-linearities, EGARCH-X performed better due to its assymetry modelling combined with reacting to exogenous sentiment

- **Walk-forward validation with XGBoost performed better:** I compared with standard xgboost and walk-forward validation performed better. This might be because retraining could have added more signal. SPY volatility dynamics were stable during the test period

- **Historical volatility dominates prediction:** The 20-day rolling mean of absolute returns (`rolling_abs_return_mean_20d`) was by far the strongest predictor. This confirms volatility persistence.  This is because instead of looking at one noisy day’s move, it looks at the average size of moves over the last 20 days. This helps the model see how turbulent the market has been recently rather than reacting to a single spike.

- **ARCH-style features (abs_return, return_squared) underperformed expectations:** `return_squared` had zero importance in XGBoost. This is likely due to the presence of lagged volatility feature which makes it add little incremental information. The model already captures volatility dynamics through historical rolling volatility.

- **Lagged returns showed limited value:** Lagged returns added very limited incremental value because the rolling volatility feature already captures past returns. Since `rolling_abs_return_mean_20d` is calculated from the last 20 days of returns, individual lagged returns become redundant.



**Recommendation:**
- Recommendation would be to regularly re train the model on new data and use a simple check to see if the market is in a calm or crazy period, then use settings that fit that period.


## Limitation and What Can Be Improved
**Limitation**
- The model is mostly looking at what happened yesterday to predict today. If there is a major sudden market crash or spike, the model may be one day late to react because it hasn't seen the news/pattern yet.


**What Can Be Improved**
- Dynamic Re-training: Implement an automated pipeline to regularly re-train the model on a sliding window.


## Repository structure

<details>
  <summary><strong>Repository Structure (click to expand)</strong></summary>

```text

Live-Fraud-Agent/
├── README.md                    # Demo GIF + docker-compose up instructions
├── requirements.txt
├── .env.example
├── docker-compose.yml          # API + DB one-command demo
├── Dockerfile
│
├── app/                        # FastAPI production API
│   ├── main.py
│   └── routes.py
│
├── agents/                     # RAG agentic fraud reasoning
│   ├── fraud_agent.py         # LLM decision layer
│   ├── prompts.py
│   └── tools.py
│
├── src/                        # ML pipeline
│   ├── inference.py           # XGBoost 
│   ├── features.py
│   └── data.py
│
├── notebooks/                  # Proof of work
│   ├── 01_eda.ipynb
│   └── 02_training.ipynb
│
├── tests/                      # Production reliability
│   └── test_end_to_end.py
├── data/                       # Sample transactions
│   └── sample_fraud.csv
└── models/
    └── xgboost_fraud.pkl



Transaction → API Call → AI Reasoning → Decision
                                            ↓
                                     Is it BLOCK?
                                     ↙          ↘
                                  YES          NO
                                   ↓            ↓
                            Ask Human      Auto-approve
                            (HITL)            (Done)
