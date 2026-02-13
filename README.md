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
│   ├── inference.py           # XGBoost (0.73 AUC)
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
