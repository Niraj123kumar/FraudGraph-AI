from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TransactionFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Time: float = Field(ge=0)
    Amount: float = Field(ge=0)

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class PredictionResponse(BaseModel):
    probability: float
    is_fraud: bool
    risk_level: Literal["low", "medium", "high"]


class BatchRequest(BaseModel):
    transactions: list[TransactionFeatures] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class RiskCutoffs(BaseModel):
    low_max_exclusive: float
    medium_max_exclusive: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    feature_count: int
    feature_names: list[str]
    risk_cutoffs: RiskCutoffs


class TransactionRecord(BaseModel):
    transaction_id: str
    dataset_version: str
    source_row: int = Field(ge=1)
    Time: float = Field(ge=0)
    Amount: float = Field(ge=0)
    Class: Literal[0, 1]


class TransactionListResponse(BaseModel):
    data_source: str
    dataset_version: str
    label_field: Literal["Class"]
    label_semantics: str
    items: list[TransactionRecord]
    total: int
    limit: int
    offset: int


class DatasetSummaryResponse(BaseModel):
    data_source: str
    dataset_version: str
    label_field: Literal["Class"]
    label_semantics: str
    record_count: int
    positive_count: int
    negative_count: int
    positive_rate: float
    total_amount: float
    mean_amount: float
    time_min_seconds: float
    time_max_seconds: float


class PaymentActivityBucket(BaseModel):
    period_start_seconds: float
    transaction_count: int
    fraud_count: int
    total_amount: float
    mean_amount: float


class PaymentActivityResponse(BaseModel):
    data_source: str
    dataset_version: str
    time_unit: Literal["seconds_from_first_transaction"]
    interval_seconds: int
    buckets: list[PaymentActivityBucket]


class EvaluationMetrics(BaseModel):
    roc_auc: float
    average_precision: float
    precision: float
    recall: float
    f1: float


class EvaluationSplit(BaseModel):
    strategy: str
    test_fraction: float
    random_seed: int
    training_partition_count: int
    threshold_selection_validation_count: int | None
    test_count: int
    transformations: str


class EvaluationConfusionMatrix(BaseModel):
    labels: list[int]
    tn: int
    fp: int
    fn: int
    tp: int


class EvaluationRocCurve(BaseModel):
    false_positive_rate: list[float]
    true_positive_rate: list[float]


class EvaluationPrecisionRecallCurve(BaseModel):
    recall: list[float]
    precision: list[float]


class EvaluationCurves(BaseModel):
    roc: EvaluationRocCurve
    precision_recall: EvaluationPrecisionRecallCurve


class EvaluationReportResponse(BaseModel):
    available: bool
    reason: str | None
    evaluation_scope: str
    evaluation_date_utc: str | None
    data_source: str | None
    dataset_version: str | None
    dataset_sha256: str | None
    model_name: str | None
    model_version: str | None
    model_sha256: str | None
    threshold: float | None
    split: EvaluationSplit | None
    sample_count: int | None
    positive_count: int | None
    negative_count: int | None
    feature_schema: list[str]
    metrics: EvaluationMetrics | None
    confusion_matrix: EvaluationConfusionMatrix | None
    curves: EvaluationCurves | None = None


class InvestigationCaseCreate(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=200)
    severity: Literal["Low", "Medium", "High"] = "Medium"
    notes: str = Field(default="", max_length=5000)


class InvestigationCaseUpdate(BaseModel):
    status: Literal["Open", "In Review", "Resolved"] | None = None
    severity: Literal["Low", "Medium", "High"] | None = None
    notes: str | None = Field(default=None, max_length=5000)


class InvestigationCase(BaseModel):
    id: str
    transaction_id: str
    dataset_version: str
    source_row: int
    Time: float
    Amount: float
    Class: Literal[0, 1]
    severity: Literal["Low", "Medium", "High"]
    status: Literal["Open", "In Review", "Resolved"]
    notes: str
    created_at: str
    updated_at: str


class InvestigationCaseList(BaseModel):
    items: list[InvestigationCase]
    total: int


class GraphNode(BaseModel):
    id: str
    degree: int
    in_degree: int
    out_degree: int
    total_amount: float
    transaction_count: int
    fraud_count: int
    risk_score: float
    reasons: list[str]


class GraphTransactionEvidence(BaseModel):
    source_row: int
    step: float
    type: str
    amount: float
    simulator_fraud_label: Literal[0, 1] | None


class GraphEdge(BaseModel):
    source: str
    target: str
    total_amount: float
    transaction_count: int
    fraud_count: int
    evidence: list[GraphTransactionEvidence]


class TransactionGraphResponse(BaseModel):
    data_source: str
    dataset_version: str
    sampled: bool
    sampling_ratio: float = 1.0
    focus_account: str | None = None
    neighborhood_truncated: bool = False
    available_connections: int = 0
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    suspicious_accounts: list[GraphNode]


class DatasetSupport(BaseModel):
    data_quality: bool
    transaction_risk: bool
    account_relationships: bool
    evaluation: bool
    reasons: dict[str, str | None]


class UploadedDataset(BaseModel):
    dataset_id: str
    filename: str
    uploaded_at: str
    sha256: str
    row_count: int
    rows_read: int
    rows_processed: int
    rows_rejected: int
    rows_skipped: int
    columns: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    total_missing_values: int
    duplicate_rows: int
    preview: list[dict[str, Any]] | None = None
    supports: DatasetSupport


class DatasetListResponse(BaseModel):
    items: list[UploadedDataset]
    total: int


class DatasetPreviewResponse(BaseModel):
    dataset_id: str
    columns: list[str]
    offset: int
    items: list[dict[str, Any]]


class UploadedPredictionResponse(PredictionResponse):
    transaction_id: str
    dataset_id: str
    source_row: int
    model_name: str
    model_version: str
    transaction: dict[str, Any]


class UploadedAnalysisRun(BaseModel):
    dataset_id: str
    status: Literal["queued", "running", "completed", "failed"]
    rows_total: int
    rows_scanned: int
    rows_processed: int
    rows_rejected: int
    rows_skipped: int
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    heartbeat_at: str | None = None


class UploadedDatasetPrediction(PredictionResponse):
    source_row: int
    Time: float
    Amount: float
    recorded_csv_label: str | None


class UploadedDatasetPredictionsResponse(BaseModel):
    dataset_id: str
    filename: str
    rows_processed: int
    rows_rejected: int
    rows_skipped: int
    items: list[UploadedDatasetPrediction]
    total: int
    limit: int
    offset: int
    model_name: str
    model_version: str
