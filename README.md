FraudGraph AI: Dataset Documentation

Runtime scope: the current app supports local credit-card transaction analysis, PaySim graph investigation, persistent SQLite investigation cases, and CSV quality uploads. Uploads never train or evaluate the model. Per-row risk checks are enabled only when an uploaded CSV has all required numeric credit-card model features. PaySim graph analysis remains separate and uses `data/raw/paysim.csv` only.

Authentication

Per-user API-key authentication is enabled by default. Create or revoke keys from the
repository root with `python scripts/manage_api_keys.py create --owner-id USER_ID` and
`python scripts/manage_api_keys.py revoke --key-id KEY_ID`. Creation prints the raw key
once; SQLite stores only its SHA-256 hash. Set `FRAUDGRAPH_SINGLE_KEY_MODE=1` only for
explicit legacy single-key deployments using `FRAUDGRAPH_API_KEY`.

The datasets and investigations SQLite stores migrate existing rows by adding an
`owner_id` column and assigning rows created before this migration to `legacy`. New
rows are assigned to the authenticated key owner. Existing database files are migrated
on first connection; back them up before deployment and ensure the process can write
the database files.

Overview

FraudGraph AI uses two separate datasets for two different tasks:

- Part 1: Transaction Classification: Detect fraudulent credit card transactions using machine learning.
- Part 2: Account-Transfer Graph Analysis: Analyze relationships between accounts to identify potentially suspicious transaction patterns.

These datasets serve different purposes and will not be merged.

Data Storage

- data/raw/: Stores downloaded original datasets.
- data/processed/: Stores cleaned or transformed datasets, if needed.

Raw and processed datasets are excluded from Git to avoid committing large files.

⸻

Part 1: Credit Card Fraud Detection

Purpose: Train a machine learning model to classify credit card transactions as fraudulent or legitimate.

Dataset source: Kaggle: Credit Card Fraud Detection

Expected file: data/raw/creditcard.csv

Dataset Information

Property Details
Total transactions 284,807
Fraudulent transactions 492
Legitimate transactions 284,315
Fraud percentage Approximately 0.17%
Total features 30
Target column Class

Columns

Column Description
Time Seconds elapsed since the first transaction in the dataset
V1 to V28 Anonymized numerical features obtained through Principal Component Analysis (PCA)
Amount Transaction amount
Class Target label: 0 = legitimate, 1 = fraudulent

Download Instructions

Download the dataset manually from Kaggle and place the extracted CSV file here:

data/raw/creditcard.csv

Alternatively, use the Kaggle CLI if it is installed and configured with your Kaggle API credentials:

kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip

Preprocessing

The preprocessing pipeline will:

1. Validate the expected columns and check for missing values.
2. Separate the features from the target variable.
3. Split the dataset into stratified training and testing sets.
4. Fit RobustScaler on the training data only.
5. Apply the fitted scaler to the training and testing features.

Class-imbalance handling, including any resampling, will be performed during model training in a later phase. The test set will never be resampled.

Limitation: This dataset does not contain account IDs or explicit sender-receiver relationships. It will be used only for transaction classification, not for constructing the account-transfer graph.

⸻

Part 2: Account-Transfer Graph Analysis

Purpose: Analyze transaction relationships between accounts and investigate potentially suspicious groups or transaction patterns.

PaySim is the selected dataset for this graph workflow. IBM AML remains an
unselected alternative described for reference; the current API does not load it.

PaySim

Dataset source: Kaggle: PaySim Synthetic Financial Dataset

Expected file: data/raw/paysim.csv

Original filename: PS_20174392719_1491204439457_log.csv

Dataset Information

PaySim contains approximately 6.36 million simulated financial transactions.

Column Description
step Time unit; each step represents one hour
type Transaction type, such as CASH_IN, CASH_OUT, DEBIT, PAYMENT, or TRANSFER
amount Transaction amount
nameOrig Originating account ID
oldbalanceOrg Sender’s balance before the transaction
newbalanceOrig Sender’s balance after the transaction
nameDest Destination account ID
oldbalanceDest Recipient’s balance before the transaction
newbalanceDest Recipient’s balance after the transaction
isFraud Indicates whether the transaction is labelled fraudulent
isFlaggedFraud Indicates whether the simulator’s rule flagged the transaction

Download Instructions

kaggle datasets download -d ealaxi/paysim1 -p data/raw --unzip

Rename the extracted CSV file:

mv data/raw/PS_20174392719_1491204439457_log.csv data/raw/paysim.csv

Option B: IBM Anti-Money Laundering Dataset

Dataset source: Kaggle: IBM Transactions for Anti-Money Laundering

Expected transaction file: data/raw/HI-Small_Trans.csv

The smaller dataset variant is recommended for initial development.

Important Columns

Column Description
Timestamp Transaction timestamp
From Bank Identifier of the sender’s bank
Account Sender account identifier in the original duplicated-header schema
To Bank Identifier of the receiving bank
Account.1 Receiver account identifier after pandas renames the duplicate column
Amount Received Amount received
Receiving Currency Currency received
Amount Paid Amount paid
Payment Currency Currency used for payment
Payment Format Payment method, such as Wire, ACH, or Cheque
Is Laundering Indicates whether a transaction is labelled as part of a laundering pattern

Download Instructions

kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -p data/raw --unzip

The exact extracted filenames may depend on the dataset archive.

An optional patterns file, if included in the downloaded dataset, may also be useful:

data/raw/HI-Small_Patterns.txt

⸻

Dataset Selection

PaySim is the selected account-transfer dataset. Its `nameOrig`, `nameDest`,
`step`, `type`, `amount`, and `isFraud` fields build the directed account graph.
The API samples at most 250,000 eligible TRANSFER/CASH_OUT rows (the dashboard
requests 100,000) to bound memory. The response reports the PaySim dataset
version and whether the result is sampled. The credit-card dataset is never
joined to PaySim; it has no account identifiers or transfer relationships.

Data Privacy and Version Control

- Do not commit raw datasets to Git.
- Do not commit processed datasets to Git.
- Keep downloaded datasets inside the appropriate data/ directories.
- Commit source code, documentation, configuration files, and tests instead.
- Document any preprocessing steps required to reproduce the experiments.

Backend API

Install dependencies and run the API from the repository root:

```sh
python3 -m pip install -r requirements.txt
FRAUDGRAPH_API_KEY=<server-side-key> python3 -m uvicorn src.api.main:app --reload
```

The API requires `X-API-Key` on its protected endpoints. Keep
`FRAUDGRAPH_API_KEY` on the server; a browser client should use a same-origin
backend proxy rather than embedding the key in frontend code.

Laptop-light mode

The MacBook API defaults to `FRAUDGRAPH_LAPTOP_MODE=1`. It does not read raw
credit-card or PaySim CSVs, build graphs, train, run SHAP, or run full upload
analysis. Set `FRAUDGRAPH_LAPTOP_MODE=1` and
`FRAUDGRAPH_GRAPH_LIVE_BUILD=0` on the Mac. Run the heavy commands in Colab,
then copy these files into the Mac repository's `models/` directory:

- `transaction_model.joblib`
- `model_metadata.json`
- `evaluation_report.json`
- `transactions.parquet`
- `graph_payload.json`
- `graph_evaluation.json`
- SHAP output files when explanations are needed

Use `FRAUDGRAPH_LAPTOP_MODE=0` to restore the previous local-computation
behavior. Laptop uploads are limited to 20 MB by default; larger files and
analysis must be processed in Colab.

Offloading to Google Colab

Open `colab/fraudgraph_colab.ipynb`, set the Drive paths, and run its cells.
The notebook clones the repository, installs `requirements-colab.txt`, copies
the raw CSVs to local Colab storage, and writes all generated artifacts to the
Drive output folder. Copy the files listed above back into `models/` on the
Mac before starting FastAPI.

Authenticated endpoints:

- `GET /datasets` lists uploaded CSVs; `POST /datasets/upload` validates and
  records a CSV up to 1 GB; uploads are streamed into a temporary file before
  bounded chunked inspection and registration. `GET /datasets/{id}` and `/preview` return its
  quality summary and rows; `POST /datasets/{id}/transactions/{row}/risk`
  predicts only for model-compatible uploaded rows. Files are stored under
  generated IDs in `data/uploads/`, with metadata in `data/datasets.sqlite3`.
  `DELETE /datasets/{id}` removes that upload and its registry record.
  These endpoints use the same API key as the rest of the local service; the
  app has no multi-user or tenant authorization.
- `POST /datasets/{id}/analysis` runs the saved model over that compatible
  upload in 10,000-row batches. `GET /datasets/{id}/analysis` reports progress;
  `GET /datasets/{id}/predictions` pages the completed, dataset-scoped results.
  Missing or non-numeric feature rows are rejected, fully blank feature rows
  are skipped, and `Class` is returned only as the separate recorded CSV label.
  The model receives only `Time`, `V1`–`V28`, and `Amount`.

PaySim-compatible uploads can be analyzed from the Data Lab's **Account
connections → Open account graph** action. This uses that upload's own rows and
requires `step`, `type`, `amount`, `nameOrig`, `nameDest`, and `isFraud`. The
graph endpoint accepts `dataset_id`, `account_id`, and `neighbor_limit` query
parameters. The default PaySim view uniformly samples at most `max_rows`
eligible `TRANSFER` and `CASH_OUT` rows (seed 42); responses mark sampling and
include source row evidence. Structural pattern scores do not use simulator
fraud labels.

- `GET /health` and `GET /model/info` report API/model availability and model
  feature information.
- `POST /predict` accepts the 30 model features. `POST /predict/batch` accepts
  1–1,000 feature records.
- `GET /transactions` returns paginated dataset records. Query parameters:
  `limit` (1–500), `offset`, `search` (stable transaction ID substring),
  `min_amount`, `max_amount`, `class_label` (0 or 1), `sort_by`
  (`source_row`, `Time`, `Amount`, or `Class`), and `sort_order` (`asc` or
  `desc`).
- `GET /transactions/{transaction_id}` returns one record.
- `POST /transactions/{transaction_id}/predict` runs the saved model against
  that dataset row's exact `Time`, `V1`–`V28`, and `Amount` features. The
  returned prediction is separate from its ground-truth `Class` label.
- `GET /transactions/summary` returns dataset counts and amount/time summary.
- `GET /transactions/payment-activity` groups labeled records by elapsed-time
  bucket; `interval_seconds` is 60–86,400 (default 3,600), with `limit` and
  `offset` for bucket pagination.
- `GET /evaluation/report` returns the persisted held-out report or an
  explicit unavailable response.
- `GET /investigations` lists persisted cases; `POST /investigations` creates
  a case linked to a real credit-card dataset row; `PATCH /investigations/{id}`
  updates status, severity, or notes. SQLite data is stored at
  `data/investigations.sqlite3` (override with `FRAUDGRAPH_CASES_DB`).
- `GET /graph/paysim` returns a bounded PaySim transfer graph and suspicious
  accounts with reason strings. `max_rows` caps the eligible rows sampled,
  `dataset_id` selects a compatible upload, and `account_id` with
  `neighbor_limit` returns that account's bounded connections and source-row
  evidence. Its score is a transparent heuristic based on
  transfer activity, fan-out, flow amount, and timing; fraud labels are
  retrospective context and are not score inputs. It is not a probability or
  credit-card model output. The PaySim dataset itself is synthetic.

Transaction IDs are dataset-derived: `creditcard-<first-16-SHA256-hex>:row:<1-based-source-row>`.
`Class` is the dataset ground-truth label, not a model prediction. The
credit-card dataset contains only elapsed seconds, amount, anonymized PCA
features, and labels; it does not support account, identity, device, location,
or payment-method inference.

For the paired frontend, keep `FRAUDGRAPH_API_KEY` in an ignored server-side
environment file and run the localhost proxy described in
`fraudgraph-frontend/README.md`. Never expose this key through a `VITE_`
variable or public frontend build. The frontend calls the same-origin `/api`
path and the proxy injects `X-API-Key`.

To reproduce the test-only evaluation with the existing saved model:

```sh
python3 -m src.ml.evaluate
```

The evaluator uses the seed-42 stratified 80/20 split, the saved model's
threshold, and raw features matching the prediction API. It writes
`models/evaluation_report.json` only with metrics when the split counts,
feature schema, model predictions, threshold, and recorded test confusion
matrix/ROC-AUC are consistent. Otherwise it persists an unavailable report
with a reason. Results are an internal holdout from this same dataset, not
external validation; no model training occurs.
