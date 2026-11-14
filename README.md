# FraudGraph AI

![Transfer network from the hosted demo](docs/images/transfer-network.png)

Look at the four accounts at the bottom of that graph. They pass around tiny amounts, 45 to 55 a hop, nowhere near the 1,800 volume threshold. Any rule that only watches how much money moves would ignore them. They're flagged because the money goes in a circle, and that's the point of this project: some fraud isn't visible in any single transaction. You only see it in the shape of the network.

FraudGraph AI puts two ways of looking at fraud in one app. A model scores individual card transactions. A graph view flags accounts that sit on loops or push unusual volume, with a reason attached to each flag. It started as a course project for *Machine Learning for Data Analytics* at SRM Institute of Science and Technology.

[Hosted demo](https://fraud-graph-ai-fullstack.vercel.app/) · [API and dataset reference](docs/API_AND_DATASETS.md) · [Frontend setup](fraudgraph-frontend/README.md)

The hosted demo is a lighter version than the code in this repo: you upload a CSV, it trains a logistic regression on the spot, and it shows a demo transfer graph. The saved XGBoost model, the PaySim graph, investigation cases and the Data Lab described below are in the repo, not the hosted demo.

## How the pieces fit

```mermaid
flowchart LR
    B["Browser<br/>React + Vite"] -->|"/api"| P["Node proxy<br/>adds the API key"]
    P --> A["FastAPI<br/>src/api"]
    A --> M[("models/<br/>model, metrics,<br/>graph payload")]
    A --> S[("SQLite<br/>uploads, cases")]
    C["Colab notebook<br/>training, SHAP,<br/>graph build"] -.->|"copy files over"| M
```

The browser never sees the API key. The proxy adds it on the way to FastAPI. And the heavy steps (training, SHAP, building the full graph) run in Colab, so the API stays light enough for a laptop.

## Two datasets that refuse to merge

The best public datasets for this problem don't overlap.

The [credit-card dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) has 284,807 real transactions, but every feature is an anonymised PCA component and there are no account IDs. You can score a transaction. You can't ask who it's connected to.

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) has accounts and transfers between them, so you can build a network. But it's synthetic, so nothing about it proves how real fraud looks.

We kept them separate on purpose, and the app says so. The model only ever sees credit-card features. The graph only ever sees PaySim transfers. The graph score is a transparent heuristic built from how active an account is, how widely it fans money out, how much flows through it and when. Simulator fraud labels appear as context but never feed the score.

## A quick look at the hosted demo

<p>
  <img src="docs/images/demo-overview.png" width="49%" alt="Overview page of the hosted demo">
  <img src="docs/images/demo-prediction.png" width="49%" alt="Fraud Prediction page after training">
</p>

Left: the overview, with the workflow from uploading data to reviewing flagged accounts. Right: the prediction page after training on a small demo file. Every prediction screen carries the same reminder, that a predicted fraud label is a screening signal and not proof of fraud.

## Accuracy is a trap here

Only 492 of the 284,807 transactions are fraud. A model that answers "legitimate" every time is over 99.8% accurate and catches nothing. The saved model's own metadata carries that warning, so we don't use accuracy.

![Confusion matrix and model comparison](docs/images/results.png)

The saved model is a class-weighted XGBoost. On a held-out test set of 56,962 transactions with 98 frauds, it catches 83 and misses 15, with 13 false alarms (precision 0.865, recall 0.847). Its decision threshold is 0.325, not the default 0.5. We picked it on a separate validation split where a missed fraud counts ten times as costly as a false alarm, so the test set never influenced it.

The right-hand chart shows average precision from cross-validation for four candidates. The top three are close, and logistic regression trails them clearly.

The hosted demo's baseline shows why the choice of metric matters. Trained on a small 1,500-row demo file, its logistic regression catches 15 of 18 frauds but raises 54 false alarms:

![Model evaluation page of the hosted demo](docs/images/model-evaluation.png)

## Where it's still weak

- 98 fraud cases is a small test set, and a few cases either way would move every metric above.
- The evaluation is an internal holdout from one dataset, not external validation. The split is random, so it ignores time order, and a deployed model would have to predict the future.
- The credit-card data covers two days of European card transactions from 2013. Fraud patterns today may look different.
- PaySim is simulated, and the graph score has no ground truth behind it. It's a way to decide what to look at first, not a verdict. The sample graph in the hosted demo flags 10 of its 18 accounts, which is a lot of review work.
- It's a single-user local app with no multi-user authorisation. Keep it on localhost.

## What's in the repo

- Credit-card transaction scoring with the saved model, per row or across a whole uploaded CSV
- A PaySim account graph with suspicion scores, reasons and source-row evidence
- Investigation cases stored in SQLite, so they survive restarts
- A Data Lab for uploading CSVs, with a quality summary
- Per-user API keys, with only a hash stored
- SHAP analysis and the Colab notebook for the heavy computation

```
src/
  api/        FastAPI app, routes, investigations, background jobs
  ml/         training, evaluation, prediction
  graph/      graph building, graph features, suspicious-group scoring
  data/       dataset handling, preprocessing, SQLite storage
  auth.py     per-user API keys
scripts/      key management, precomputing graph and transactions, SHAP
models/       saved model, metadata, evaluation report
colab/        notebook for the heavy computation
fraudgraph-frontend/   React dashboard and the authenticated proxy
tests/        backend tests
```

## Run it

Put the datasets in `data/raw/`:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
kaggle datasets download -d ealaxi/paysim1 -p data/raw --unzip
mv data/raw/PS_20174392719_1491204439457_log.csv data/raw/paysim.csv
```

Start the API from the repository root:

```bash
python3 -m pip install -r requirements.txt
FRAUDGRAPH_API_KEY=<server-side-key> python3 -m uvicorn src.api.main:app --reload
```

Then start the dashboard. It needs the proxy running too, and [`fraudgraph-frontend/README.md`](fraudgraph-frontend/README.md) walks through all three terminals:

```bash
cd fraudgraph-frontend
npm install
npm run dev
```

Tests: `pytest -q`
