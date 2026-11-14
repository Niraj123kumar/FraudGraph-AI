import type {
  BatchResponse,
  EvaluationReportResponse,
  HealthResponse,
  ModelInfoResponse,
  PaymentActivityQuery,
  PaymentActivityResponse,
  PredictionResponse,
  TransactionSummaryResponse,
  TransactionsQuery,
  TransactionsResponse,
  TransactionFeatures,
  DatasetTransaction,
  InvestigationCase,
  InvestigationCaseList,
  TransactionGraphResponse,
  UploadedDataset,
  UploadedDatasetList,
  DatasetPreview,
  UploadedPrediction,
  UploadedAnalysisRun,
  UploadedDatasetPredictionsResponse,
} from '../types/api'

const REQUEST_TIMEOUT_MS = 10_000

function getApiUrl(path: string): string {
  const baseUrl = import.meta.env.VITE_API_BASE_URL?.trim() || '/api'

  return `${baseUrl.replace(/\/+$/, '')}${path}`
}

async function requestJson<T>(
  path: string,
  method: 'GET' | 'POST' | 'PATCH' | 'DELETE' = 'GET',
  body?: unknown,
): Promise<T> {
  const url = getApiUrl(path)
  const controller = new AbortController()
  const isFormData = typeof FormData !== 'undefined' && body instanceof FormData
  const timeoutMs = isFormData ? 1_800_000 : REQUEST_TIMEOUT_MS
  const timeout = setTimeout(() => controller.abort(), timeoutMs)

  let response: Response
  try {
    response = await fetch(url, {
      method,
      headers: body === undefined || isFormData ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : isFormData ? body : JSON.stringify(body),
      signal: controller.signal,
    })
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error(
        `Request to ${path} timed out after ${timeoutMs} ms.`,
        { cause: error },
      )
    }

    if (error instanceof TypeError) {
      throw new Error(
        `Network error while requesting ${path}. Check the backend URL, network connection, and CORS configuration.`,
        { cause: error },
      )
    }

    throw error
  } finally {
    clearTimeout(timeout)
  }

  if (!response.ok) {
    const statusText = response.statusText ? ` ${response.statusText}` : ''
    throw new Error(
      `Request to ${path} failed with HTTP ${response.status}${statusText}.`,
    )
  }

  if (response.status === 204) return undefined as T

  try {
    return (await response.json()) as T
  } catch (error) {
    throw new Error(`The backend returned invalid JSON for ${path}.`, {
      cause: error,
    })
  }
}

export function checkHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>('/health')
}

export function getModelInfo(): Promise<ModelInfoResponse> {
  return requestJson<ModelInfoResponse>('/model/info')
}

export function predictTransaction(
  features: TransactionFeatures,
): Promise<PredictionResponse> {
  return requestJson<PredictionResponse>('/predict', 'POST', features)
}

export function predictDatasetTransaction(transactionId: string): Promise<PredictionResponse> {
  return requestJson<PredictionResponse>(`/transactions/${encodeURIComponent(transactionId)}/predict`, 'POST')
}

export function predictBatch(
  transactions: TransactionFeatures[],
): Promise<BatchResponse> {
  return requestJson<BatchResponse>('/predict/batch', 'POST', { transactions })
}

function appendQuery<T extends object>(path: string, params: T): string {
  const query = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== '') {
      query.set(key, String(value))
    }
  }
  const encoded = query.toString()
  return encoded ? `${path}?${encoded}` : path
}

export function getTransactions(
  params: TransactionsQuery = {},
): Promise<TransactionsResponse> {
  return requestJson<TransactionsResponse>(appendQuery('/transactions', params))
}

export function getTransaction(
  transactionId: string,
): Promise<DatasetTransaction> {
  return requestJson<DatasetTransaction>(
    `/transactions/${encodeURIComponent(transactionId)}`,
  )
}

export function getTransactionSummary(): Promise<TransactionSummaryResponse> {
  return requestJson<TransactionSummaryResponse>('/transactions/summary')
}

export function getPaymentActivity(
  params: PaymentActivityQuery = {},
): Promise<PaymentActivityResponse> {
  return requestJson<PaymentActivityResponse>(
    appendQuery('/transactions/payment-activity', {
      interval_seconds: params.interval_seconds ?? 3600,
      limit: params.limit ?? 500,
      offset: params.offset ?? 0,
    }),
  )
}

export function getEvaluationReport(): Promise<EvaluationReportResponse> {
  return requestJson<EvaluationReportResponse>('/evaluation/report')
}

export function getInvestigationCases(): Promise<InvestigationCaseList> {
  return requestJson<InvestigationCaseList>('/investigations')
}

export function createInvestigationCase(input: {
  transaction_id: string; severity: InvestigationCase['severity']; notes: string
}): Promise<InvestigationCase> {
  return requestJson<InvestigationCase>('/investigations', 'POST', input)
}

export function updateInvestigationCase(id: string, input: Partial<Pick<InvestigationCase, 'severity' | 'status' | 'notes'>>): Promise<InvestigationCase> {
  return requestJson<InvestigationCase>(`/investigations/${encodeURIComponent(id)}`, 'PATCH', input)
}

export function getPaysimGraph(params: { limit?: number; max_rows?: number; dataset_id?: string; account_id?: string; neighbor_limit?: number } = {}): Promise<TransactionGraphResponse> {
  return requestJson<TransactionGraphResponse>(appendQuery('/graph/paysim', params))
}

export function uploadDataset(file: File): Promise<UploadedDataset> {
  const body = new FormData()
  body.set('file', file)
  return requestJson<UploadedDataset>('/datasets/upload', 'POST', body)
}

export function getUploadedDatasets(): Promise<UploadedDatasetList> {
  return requestJson<UploadedDatasetList>('/datasets')
}

export function deleteUploadedDataset(datasetId: string): Promise<void> {
  return requestJson<void>(`/datasets/${encodeURIComponent(datasetId)}`, 'DELETE')
}

export function getUploadedDataset(datasetId: string): Promise<UploadedDataset> {
  return requestJson<UploadedDataset>(`/datasets/${encodeURIComponent(datasetId)}`)
}

export function getUploadedDatasetPreview(datasetId: string, limit = 10, offset = 0): Promise<DatasetPreview> {
  return requestJson<DatasetPreview>(appendQuery(`/datasets/${encodeURIComponent(datasetId)}/preview`, { limit, offset }))
}

export function predictUploadedTransaction(datasetId: string, sourceRow: number): Promise<UploadedPrediction> {
  return requestJson<UploadedPrediction>(`/datasets/${encodeURIComponent(datasetId)}/transactions/${sourceRow}/risk`, 'POST')
}

export function startUploadedDatasetAnalysis(datasetId: string): Promise<UploadedAnalysisRun> {
  return requestJson<UploadedAnalysisRun>(`/datasets/${encodeURIComponent(datasetId)}/analysis`, 'POST')
}

export function getUploadedDatasetAnalysis(datasetId: string): Promise<UploadedAnalysisRun> {
  return requestJson<UploadedAnalysisRun>(`/datasets/${encodeURIComponent(datasetId)}/analysis`)
}

export function getUploadedDatasetPredictions(datasetId: string, limit = 50, offset = 0): Promise<UploadedDatasetPredictionsResponse> {
  return requestJson<UploadedDatasetPredictionsResponse>(appendQuery(`/datasets/${encodeURIComponent(datasetId)}/predictions`, { limit, offset }))
}
