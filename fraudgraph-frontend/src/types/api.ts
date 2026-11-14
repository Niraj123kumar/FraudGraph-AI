export interface HealthResponse {
status: string
model: string | null
}

export interface RiskCutoffs {
low_max_exclusive: number
medium_max_exclusive: number
}

export interface ModelInfoResponse {
model_name: string
model_version: string
threshold: number
feature_count: number
feature_names: string[]
risk_cutoffs: RiskCutoffs
}

export interface TransactionFeatures {
Time: number
V1: number
V2: number
V3: number
V4: number
V5: number
V6: number
V7: number
V8: number
V9: number
V10: number
V11: number
V12: number
V13: number
V14: number
V15: number
V16: number
V17: number
V18: number
V19: number
V20: number
V21: number
V22: number
V23: number
V24: number
V25: number
V26: number
V27: number
V28: number
Amount: number
}

export interface PredictionResponse {
probability: number
is_fraud: boolean
risk_level: 'low' | 'medium' | 'high'
}

export interface BatchResponse {
predictions: PredictionResponse[]
count: number
}

export interface DatasetTransaction {
transaction_id: string
dataset_version: string
source_row: number
Time: number
Amount: number
Class: 0 | 1
}

export interface TransactionsResponse {
data_source: string
dataset_version: string
label_field: 'Class'
label_semantics: string
items: DatasetTransaction[]
total: number
limit: number
offset: number
}

export type TransactionSortField = 'source_row' | 'Time' | 'Amount' | 'Class'
export type SortOrder = 'asc' | 'desc'

export interface TransactionsQuery {
limit?: number
offset?: number
search?: string
min_amount?: number
max_amount?: number
class_label?: 0 | 1
sort_by?: TransactionSortField
sort_order?: SortOrder
}

export interface TransactionSummaryResponse {
data_source: string
dataset_version: string
label_field: string
label_semantics: string
record_count: number
positive_count: number
negative_count: number
positive_rate: number
total_amount: number
mean_amount: number
time_min_seconds: number
time_max_seconds: number
}

export interface PaymentActivityBucket {
period_start_seconds: number
transaction_count: number
fraud_count: number
total_amount: number
mean_amount: number
}

export interface PaymentActivityResponse {
data_source: string
dataset_version: string
time_unit: 'seconds_from_first_transaction'
interval_seconds: number
buckets: PaymentActivityBucket[]
}

export interface PaymentActivityQuery {
interval_seconds?: number
limit?: number
offset?: number
}

export interface EvaluationSplit {
strategy: string | null
test_fraction: number | null
random_seed: number | null
training_partition_count: number | null
threshold_selection_validation_count: number | null
test_count: number | null
transformations: unknown
}

export interface EvaluationMetrics {
roc_auc: number | null
average_precision: number | null
precision: number | null
recall: number | null
f1: number | null
}

export interface EvaluationConfusionMatrix {
labels: [0, 1]
tn: number
fp: number
fn: number
tp: number
}

export interface EvaluationCurves {
roc: { false_positive_rate: number[]; true_positive_rate: number[] }
precision_recall: { recall: number[]; precision: number[] }
}

export interface EvaluationReportResponse {
available: boolean
reason: string | null
evaluation_scope: string | null
evaluation_date_utc: string | null
data_source: string | null
dataset_version: string | null
dataset_sha256: string | null
model_name: string | null
model_version: string | null
model_sha256: string | null
threshold: number | null
split: EvaluationSplit | null
sample_count: number | null
positive_count: number | null
negative_count: number | null
feature_schema: unknown
metrics: EvaluationMetrics | null
confusion_matrix: EvaluationConfusionMatrix | null
curves?: EvaluationCurves | null
}

export interface InvestigationCase {
  id: string
  transaction_id: string
  dataset_version: string
  source_row: number
  Time: number
  Amount: number
  Class: 0 | 1
  severity: 'Low' | 'Medium' | 'High'
  status: 'Open' | 'In Review' | 'Resolved'
  notes: string
  created_at: string
  updated_at: string
}

export interface InvestigationCaseList { items: InvestigationCase[]; total: number }
export interface GraphNode {
  id: string; degree: number; in_degree: number; out_degree: number
  total_amount: number; transaction_count: number; fraud_count: number
  risk_score: number; reasons: string[]
}
export interface GraphEdge {
  source: string; target: string; total_amount: number; transaction_count: number; fraud_count: number
  evidence: { source_row: number; step: number; type: string; amount: number; simulator_fraud_label: 0 | 1 | null }[]
}
export interface TransactionGraphResponse {
  data_source: string; dataset_version: string; sampled: boolean
  nodes: GraphNode[]; edges: GraphEdge[]; suspicious_accounts: GraphNode[]
  focus_account: string | null; neighborhood_truncated: boolean; available_connections: number
}

export interface UploadedDataset {
  dataset_id: string
  filename: string
  uploaded_at: string
  sha256: string
  row_count: number
  rows_read: number
  rows_processed: number
  rows_rejected: number
  rows_skipped: number
  columns: string[]
  dtypes: Record<string, string>
  missing_values: Record<string, number>
  total_missing_values: number
  duplicate_rows: number
  preview?: Record<string, unknown>[] | null
  supports: {
    data_quality: boolean
    transaction_risk: boolean
    account_relationships: boolean
    evaluation: boolean
    reasons: Record<string, string | null>
  }
}

export interface UploadedDatasetList { items: UploadedDataset[]; total: number }
export interface DatasetPreview { dataset_id: string; columns: string[]; offset: number; items: Record<string, unknown>[] }
export interface UploadedPrediction extends PredictionResponse {
  transaction_id: string; dataset_id: string; source_row: number
  model_name: string; model_version: string; transaction: Record<string, unknown>
}

export interface UploadedAnalysisRun {
  dataset_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  rows_total: number
  rows_scanned: number
  rows_processed: number
  rows_rejected: number
  rows_skipped: number
  error: string | null
  started_at: string | null
  completed_at: string | null
}

export interface UploadedDatasetPrediction extends PredictionResponse {
  source_row: number
  Time: number
  Amount: number
  recorded_csv_label: string | null
}

export interface UploadedDatasetPredictionsResponse {
  dataset_id: string
  filename: string
  rows_processed: number
  rows_rejected: number
  rows_skipped: number
  items: UploadedDatasetPrediction[]
  total: number
  limit: number
  offset: number
  model_name: string
  model_version: string
}
