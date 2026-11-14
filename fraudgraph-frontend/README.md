# FraudGraph AI

A local fraud analysis dashboard for the two supplied CSV datasets. The React app talks to a same-origin Node proxy; the proxy adds the API key server-side and forwards requests to FastAPI. The dark fintech interface is retained.

## What each page uses

- **Overview, Transactions, Payment Activity:** actual rows and labels from `FraudGraph-AI/data/raw/creditcard.csv`. `Class` is the CSV's observed label. A requested per-row prediction uses that row's `Time`, `V1`–`V28`, and `Amount` in the saved model's exact feature order.
- **Model Performance:** backend model metadata and the saved evaluation artifact, including its dataset/model provenance. It is a separate holdout report and never recomputed on uploaded data.
- **Fraud Network:** PaySim transaction rows only (`step`, `type`, `amount`, `nameOrig`, `nameDest`, and simulator labels). Graph scores are explainable structural heuristics; simulator labels are displayed as retrospective evidence and do not contribute to scores.
- **Investigations:** cases are stored in the backend SQLite database and link to actual credit-card CSV transaction records. The cases survive API restarts on this local installation.
- **Data Lab:** uploaded CSVs are stored under generated server-side IDs, registered in SQLite, and analyzed in chunks. Credit-card-compatible uploads can run the saved model over all eligible rows; Transactions then shows results from that selected upload only. Missing or non-numeric rows are reported, and any `Class` label stays separate from model inputs and predictions. PaySim-compatible uploads can open their own account graph; evaluation remains unavailable for uploads.
- Uploads can be deleted from Data Lab, which removes the CSV and its registry record.

Credit-card transactions and PaySim accounts are separate datasets and are not joined. Upload records and investigation cases are available to users who can reach this single-user local API; this app does not implement multi-user identity or tenant-level authorization. Keep it bound to localhost and do not expose it to an untrusted network.

The notebook `FraudGraph_AI.ipynb` is a legacy Colab/setup notebook, not the runtime entry point. No upload is used to train or evaluate the model.

## Required files

Place the supplied files at:

```text
FraudGraph-AI/data/raw/creditcard.csv
FraudGraph-AI/data/raw/paysim.csv
FraudGraph-AI/models/transaction_model.joblib
```

Rename `PS_20174392719_1491204439457_log.csv` to `paysim.csv`. The API also needs the checked-in model metadata and evaluation report under `models/`. Raw data, uploads, SQLite databases, and secrets are git-ignored.

## Setup

Create a strong key and place the same value in two server-side environments. Do not use a `VITE_` prefix or put the key in browser code.

```sh
cd ~/Desktop/fraudgraph-frontend
cp .env.example .env
```

Edit `.env` and set `FRAUDGRAPH_API_KEY` to a newly generated value. The frontend reads only `VITE_API_BASE_URL=/api` from `.env.local`; keep the secret only in `.env`.

Install backend dependencies:

```sh
cd ~/Desktop/FraudGraph-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install frontend dependencies once:

```sh
cd ~/Desktop/fraudgraph-frontend
npm install
```

## Run locally

Use three terminals.

Terminal 1, API:

```sh
cd ~/Desktop/FraudGraph-AI
set -a; source ../fraudgraph-frontend/.env; set +a
source .venv/bin/activate
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Terminal 2, authenticated proxy:

```sh
cd ~/Desktop/fraudgraph-frontend
set -a; source .env; set +a
export FRAUDGRAPH_BACKEND_URL=http://127.0.0.1:8000
node proxy.mjs
```

Terminal 3, dashboard:

```sh
cd ~/Desktop/fraudgraph-frontend
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`). The proxy is restricted to `127.0.0.1:8001`; it streams uploads up to 1 GB plus multipart overhead and allows up to 30 minutes for large uploads. Check API health through it with `curl -i http://127.0.0.1:8001/health`.

## Test and build

```sh
cd ~/Desktop/FraudGraph-AI
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider

cd ~/Desktop/fraudgraph-frontend
npm run lint
npm run build
node --test proxy-body.test.mjs graph-selection.test.mjs
```
