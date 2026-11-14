import { useEffect, useMemo, useRef, useState, type ChangeEvent, type FormEvent, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react'
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  CreditCard,
  Download,
  Fingerprint,
  LayoutDashboard,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  SlidersHorizontal,
  Trash2,
  UsersRound,
  WalletCards,
  type LucideIcon,
} from 'lucide-react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  checkHealth,
  getEvaluationReport,
  getInvestigationCases,
  createInvestigationCase,
  updateInvestigationCase,
  getPaysimGraph,
  uploadDataset,
  getUploadedDatasets,
  deleteUploadedDataset,
  getUploadedDatasetPreview,
  getUploadedDatasetAnalysis,
  getUploadedDatasetPredictions,
  predictUploadedTransaction,
  startUploadedDatasetAnalysis,
  getModelInfo,
  getPaymentActivity,
  getTransaction,
  getTransactionSummary,
  getTransactions,
  predictDatasetTransaction,
} from './services/api'
import type {
  DatasetTransaction,
  EvaluationReportResponse,
  ModelInfoResponse,
  PaymentActivityResponse,
  PredictionResponse,
  TransactionSortField,
  TransactionSummaryResponse,
  TransactionsResponse,
  InvestigationCase as ApiInvestigationCase,
  TransactionGraphResponse,
  UploadedDataset,
  UploadedDatasetList,
  DatasetPreview,
  UploadedPrediction,
  UploadedAnalysisRun,
  UploadedDatasetPredictionsResponse,
} from './types/api'
import { isSelectedGraphEdge, isSelectedTransferAccount } from './graph-selection.mjs'
import './App.css'

type LocalInvestigation = {
  id: string
  transaction: DatasetTransaction
  severity: 'Low' | 'Medium' | 'High'
  status: 'Open' | 'In Review' | 'Resolved'
  notes: string
  createdAt: string
  updatedAt: string
}

function toLocalInvestigation(item: ApiInvestigationCase): LocalInvestigation {
  return {
    id: item.id,
    transaction: { transaction_id: item.transaction_id, dataset_version: item.dataset_version,
      source_row: item.source_row, Time: item.Time, Amount: item.Amount, Class: item.Class },
    severity: item.severity, status: item.status, notes: item.notes,
    createdAt: item.created_at, updatedAt: item.updated_at,
  }
}

function displayObservedLabel(value: number): string {
  return value === 1 ? 'Labeled fraud (Class 1)' : 'Labeled legitimate (Class 0)'
}

const navigation: { label: string; icon: LucideIcon }[] = [
  { label: 'Overview', icon: LayoutDashboard },
  { label: 'Transactions', icon: CreditCard },
  { label: 'Investigations', icon: Fingerprint },
  { label: 'Model Performance', icon: Activity },
  { label: 'Fraud Network', icon: UsersRound },
  { label: 'Data Lab', icon: Download },
]

const tooltipStyle = {
  background: '#151e2b',
  border: '1px solid #293648',
  borderRadius: '8px',
  color: '#e9eff7',
  fontSize: '12px',
}

type BackendStatusResult = {
  status: 'connected' | 'offline'
  modelInfo: ModelInfoResponse | null
  error: string
}

async function fetchBackendStatus(): Promise<BackendStatusResult> {
  try {
    const health = await checkHealth()
    if (health.status.toLowerCase() !== 'ok' || health.model !== 'ready') {
      return { status: 'offline', modelInfo: null, error: '' }
    }

    try {
      return {
        status: 'connected',
        modelInfo: await getModelInfo(),
        error: '',
      }
    } catch (error) {
      return {
        status: 'connected',
        modelInfo: null,
        error: error instanceof Error ? error.message : 'Model metadata is unavailable.',
      }
    }
  } catch {
    return { status: 'offline', modelInfo: null, error: '' }
  }
}

function App() {
  const [activePage, setActivePage] = useState('Overview')
  const [graphDatasetId, setGraphDatasetId] = useState<string | null>(null)
  const [analysisDatasetId, setAnalysisDatasetId] = useState<string | null>(null)
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'offline'>('checking')
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null)
  const [modelInfoError, setModelInfoError] = useState('')

  useEffect(() => {
    let active = true
    void fetchBackendStatus().then((result) => {
      if (!active) return
      setBackendStatus(result.status)
      setModelInfo(result.modelInfo)
      setModelInfoError(result.error)
    })
    return () => {
      active = false
    }
  }, [])

  const retryBackendStatus = () => {
    setBackendStatus('checking')
    setModelInfoError('')
    void fetchBackendStatus().then((result) => {
      setBackendStatus(result.status)
      setModelInfo(result.modelInfo)
      setModelInfoError(result.error)
    })
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" onClick={() => setActivePage('Overview')} type="button">
          <span className="brand-mark">
            <Fingerprint size={19} strokeWidth={2.2} />
          </span>
          <span className="brand-name">FraudGraph <b>AI</b></span>
        </button>

        <div className="workspace-picker">
          <span className="workspace-avatar">FG</span>
          <span className="workspace-copy">
            <strong>FraudGraph</strong>
            <small>CSV analysis</small>
          </span>
          <ChevronDown size={15} />
        </div>

        <div className="sidebar-label">WORKSPACE</div>
        <nav className="primary-nav" aria-label="Main navigation">
          {navigation.map(({ label, icon: Icon }) => (
            <button
              className={`nav-item ${activePage === label ? 'active' : ''}`}
              key={label}
              onClick={() => setActivePage(label)}
              type="button"
            >
              <Icon size={17} strokeWidth={1.8} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-label sources-label">DATA SOURCES</div>
        <button
          className={`nav-item source-item ${activePage === 'Payment Activity' ? 'active' : ''}`}
          onClick={() => setActivePage('Payment Activity')}
          type="button"
        >
          <WalletCards size={17} strokeWidth={1.8} />
          <span>Payment activity</span>
          <span className="source-dot" />
        </button>
        <div className="sidebar-bottom">
          <button className="nav-item help-item" type="button">
            <CircleHelp size={17} strokeWidth={1.8} />
            <span>Help & support</span>
          </button>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>Workspace</span>
            <ChevronRight size={14} />
            <strong>{activePage}</strong>
          </div>
          <div className="topbar-actions">
            <div
              className={`model-status ${backendStatus}`}
              aria-live="polite"
              title={backendStatus === 'connected' ? 'Backend health check succeeded' : 'Backend health status'}
            >
              <span className="status-indicator" />
              <span>
                {backendStatus === 'connected'
                  ? 'Model connected'
                  : backendStatus === 'offline'
                    ? 'Backend offline'
                    : 'Checking backend'}
              </span>
              <span className="status-divider" />
              <span className="status-no-model">
                {backendStatus === 'connected'
                  ? modelInfo
                    ? `${modelInfo.model_name} · ${modelInfo.model_version}`
                    : 'Model details unavailable'
                  : backendStatus === 'offline'
                    ? 'Data unavailable'
                    : 'Checking model status'}
              </span>
              {backendStatus === 'offline' && (
                <button className="status-retry" onClick={retryBackendStatus} type="button">
                  Retry
                </button>
              )}
            </div>
          </div>
        </header>

        <div className="dashboard-body">
          <div className="page-heading">
            <div>
              <div className="eyebrow"><span className="eyebrow-dot" /> FRAUD INTELLIGENCE</div>
              <h1>{activePage}</h1>
              <p className="page-description">
                {activePage === 'Overview' && 'A summary of the available credit-card CSV and recorded activity.'}
                {activePage === 'Transactions' && 'Search records and check risk for an actual CSV transaction.'}
                {activePage === 'Investigations' && 'Create and update cases linked to real transaction records.'}
                {activePage === 'Model Performance' && 'See what the saved model was measured on and its reported results.'}
                {activePage === 'Fraud Network' && 'Inspect account connections from the PaySim CSV. Patterns need investigation; they do not prove fraud.'}
                {activePage === 'Payment Activity' && 'View recorded transaction counts from the credit-card CSV over elapsed time.'}
                {activePage === 'Data Lab' && 'Upload a CSV to inspect its quality and available analyses.'}
              </p>
            </div>
          </div>

          {activePage === 'Overview' ? (
            <OverviewData onViewAll={() => setActivePage('Transactions')} />
          ) : activePage === 'Transactions' ? (
            <TransactionsPage initialDatasetId={analysisDatasetId} onDatasetSelected={setAnalysisDatasetId} key={analysisDatasetId ?? 'preloaded-creditcard'} />
          ) : activePage === 'Investigations' ? (
            <InvestigationsPage />
          ) : activePage === 'Model Performance' ? (
            <ModelPerformancePage
              backendStatus={backendStatus}
              modelInfo={modelInfo}
              modelInfoError={modelInfoError}
              onRetry={retryBackendStatus}
            />
          ) : activePage === 'Payment Activity' ? (
            <PaymentActivityPage />
          ) : activePage === 'Data Lab' ? (
            <DataLabPage onAnalyzeGraph={(datasetId) => { setGraphDatasetId(datasetId); setActivePage('Fraud Network') }} onAnalysisComplete={(datasetId) => { setAnalysisDatasetId(datasetId); setActivePage('Transactions') }} />
          ) : (
            <FraudNetworkPage key={graphDatasetId ?? 'bundled-paysim'} datasetId={graphDatasetId} onUseBundled={() => setGraphDatasetId(null)} />
          )}
          <footer className="page-footer">
            <span>FraudGraph AI <span className="footer-separator">/</span> Local CSV analysis</span>
            <span>
              {activePage === 'Investigations' ? 'Persisted cases only' :
                activePage === 'Fraud Network' ? 'PaySim CSV records' : 'Current data source is shown on each page'}
            </span>
          </footer>
        </div>
      </main>
    </div>
  )
}

function MetricCard({
  icon: Icon,
  iconClass,
  label,
  value,
  comparison,
  change,
  trend,
  footnote,
}: {
  icon: LucideIcon
  iconClass: string
  label: string
  value: string
  comparison?: string
  change?: string
  trend?: 'up' | 'down'
  footnote: string
}) {
  const TrendIcon = trend === 'down' ? ArrowDownRight : ArrowUpRight
  return (
    <article className="panel metric-card">
      <div className="metric-top">
        <span className={`metric-icon ${iconClass}`}><Icon size={17} strokeWidth={1.9} /></span>
        <span className="metric-demo">{footnote}</span>
      </div>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {change && trend && (
        <div className="metric-bottom">
          <span className={`metric-change ${trend}`}><TrendIcon size={13} />{change}</span>
          <span>{comparison}</span>
        </div>
      )}
    </article>
  )
}

function OverviewData({ onViewAll }: { onViewAll: () => void }) {
  const [summary, setSummary] = useState<TransactionSummaryResponse | null>(null)
  const [activity, setActivity] = useState<PaymentActivityResponse | null>(null)
  const [recent, setRecent] = useState<TransactionsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [refreshKey, setRefreshKey] = useState(0)

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      setError('')
      setSummary(null)
      setActivity(null)
      setRecent(null)
      try {
        const [s, a, r] = await Promise.all([
          getTransactionSummary(),
          getPaymentActivity({ interval_seconds: 3600, limit: 500, offset: 0 }),
          getTransactions({ limit: 6, sort_by: 'source_row', sort_order: 'desc' }),
        ])
        if (!active) return
        setSummary(s)
        setActivity(a)
        setRecent(r)
      } catch (requestError) {
        if (active) {
          setError(requestError instanceof Error ? requestError.message : 'Unable to load overview data.')
        }
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => {
      active = false
    }
  }, [refreshKey])

  const dash = '—'
  const buckets = activity?.buckets ?? []
  return (
    <>
      {error && (
        <div className="dataset-error" role="alert">
          <span>{error}</span>
          <button className="text-button" onClick={() => setRefreshKey((current) => current + 1)} type="button">Retry</button>
        </div>
      )}

      <section className="metrics-grid" aria-label="Dataset summary">
        <MetricCard icon={CreditCard} iconClass="cyan" label="Dataset records" value={summary ? summary.record_count.toLocaleString() : dash} footnote="Backend data" />
        <MetricCard icon={ShieldAlert} iconClass="amber" label="Previously labeled fraud" value={summary ? summary.positive_count.toLocaleString() : dash} footnote="CSV labels, not new findings" />
        <MetricCard icon={Activity} iconClass="rose" label="Share previously labeled fraud" value={summary ? `${(summary.positive_rate * 100).toFixed(3)}%` : dash} footnote="CSV labels, not new findings" />
        <MetricCard icon={ShieldCheck} iconClass="green" label="Mean amount" value={summary ? summary.mean_amount.toLocaleString(undefined, { maximumFractionDigits: 2 }) : dash} footnote="Dataset units" />
      </section>

      <section className="charts-grid">
        <article className="panel activity-panel">
          <div className="panel-heading">
            <div>
              <h2>Transaction activity</h2>
              <p>Transactions and rows previously labeled as fraud, by elapsed hour</p>
            </div>
            {activity && <span className="demo-tag">{buckets.length.toLocaleString()} BUCKETS</span>}
          </div>
          {loading && !activity ? (
            <p className="dataset-message">Loading activity…</p>
          ) : buckets.length === 0 ? (
            <p className="dataset-message">No activity data available.</p>
          ) : (
            <div className="activity-chart">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={buckets} margin={{ top: 8, right: 4, left: -18, bottom: 0 }}>
                  <CartesianGrid stroke="#202c3a" strokeDasharray="3 5" vertical={false} />
                  <XAxis dataKey="period_start_seconds" axisLine={false} tickLine={false} tick={{ fill: '#94a4b6', fontSize: 12 }} tickFormatter={(seconds: number) => `${(seconds / 3600).toFixed(0)}h`} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fill: '#94a4b6', fontSize: 12 }} />
                  <Tooltip contentStyle={tooltipStyle} labelFormatter={(seconds) => `${Number(seconds).toLocaleString()} s from first transaction`} />
                  <Area type="monotone" dataKey="transaction_count" name="Transactions" stroke="#52c8d8" strokeWidth={2} fill="#52c8d822" />
                  <Area type="monotone" dataKey="fraud_count" name="Previously labeled fraud" stroke="#f0b660" strokeWidth={2} fill="#f0b66022" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </article>

      </section>

      <section className="lower-grid details-collapsed">
        <article className="panel transactions-panel">
          <div className="panel-heading transaction-heading">
            <div>
              <div className="heading-title-line"><h2>Recent dataset records</h2></div>
              <p>Highest source rows from the backend dataset</p>
            </div>
            <div className="table-actions">
              <button className="text-button" type="button" onClick={onViewAll}>View all <ChevronRight size={14} /></button>
            </div>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr><th>TRANSACTION</th><th>AMOUNT</th><th>TIME · SECONDS</th><th>CSV LABEL</th></tr>
              </thead>
              <tbody>
                {(recent?.items ?? []).map((item) => (
                  <tr key={item.transaction_id}>
                    <td><strong>{item.transaction_id}</strong><small> Row {item.source_row}</small></td>
                    <td className="amount-cell">{item.Amount.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                    <td>{item.Time.toLocaleString()} s</td>
                    <td><span className={`class-label class-${item.Class}`}>{displayObservedLabel(item.Class)}</span></td>
                  </tr>
                ))}
                {loading && !recent && <tr><td className="dataset-message" colSpan={4}>Loading records…</td></tr>}
              </tbody>
            </table>
          </div>
        </article>
      </section>
    </>
  )
}

function TransactionsPage({ initialDatasetId, onDatasetSelected }: { initialDatasetId: string | null; onDatasetSelected: (datasetId: string | null) => void }) {
  const [datasets, setDatasets] = useState<UploadedDataset[]>([])
  const [datasetId, setDatasetId] = useState(initialDatasetId ?? '')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    void getUploadedDatasets().then((result) => {
      if (active) { setDatasets(result.items); setError('') }
    }).catch((requestError: unknown) => {
      if (active) setError(requestError instanceof Error ? requestError.message : 'Uploaded dataset list is unavailable.')
    }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [])
  const selected = datasets.find((dataset) => dataset.dataset_id === datasetId) ?? null
  return <>
    <div className="dataset-source-switcher"><label><span>Transaction source</span><select aria-label="Transaction source" value={datasetId} onChange={(event) => { setDatasetId(event.target.value); onDatasetSelected(event.target.value || null) }}><option value="">Preloaded credit-card dataset</option>{datasets.map((dataset) => <option key={dataset.dataset_id} value={dataset.dataset_id}>{dataset.filename}</option>)}</select></label><small>{datasetId ? selected?.supports.transaction_risk ? 'Predictions shown only for this selected upload.' : 'This upload does not match the saved credit-card model schema.' : 'Showing the preloaded credit-card dataset and its recorded CSV labels.'}</small></div>
    {error && !loading && <p className="dataset-message error" role="alert">Uploaded dataset list is unavailable: {error}</p>}
    {datasetId && selected ? <UploadedDatasetTransactionsPage key={datasetId} dataset={selected} /> : datasetId && loading ? <p className="dataset-message">Loading selected uploaded dataset…</p> : datasetId ? <p className="dataset-message error">The selected upload is no longer available. Choose another transaction source.</p> : <PreloadedTransactionsPage />}
  </>
}

function UploadedDatasetTransactionsPage({ dataset }: { dataset: UploadedDataset }) {
  const [run, setRun] = useState<UploadedAnalysisRun | null>(null)
  const [response, setResponse] = useState<UploadedDatasetPredictionsResponse | null>(null)
  const [selectedRow, setSelectedRow] = useState<number | null>(null)
  const [offset, setOffset] = useState(0)
  const [refresh, setRefresh] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [starting, setStarting] = useState(false)
  const limit = 50
  useEffect(() => {
    let active = true
    let timer: number | undefined
    const poll = async () => {
      try {
        const status = await getUploadedDatasetAnalysis(dataset.dataset_id)
        if (!active) return
        setRun(status)
        if (status.status === 'completed') {
          const predictions = await getUploadedDatasetPredictions(dataset.dataset_id, limit, offset)
          if (!active) return
          setResponse(predictions)
          setSelectedRow((current) => predictions.items.some((item) => item.source_row === current) ? current : predictions.items[0]?.source_row ?? null)
          setError('')
          setLoading(false)
        } else if (status.status === 'queued' || status.status === 'running') {
          timer = window.setTimeout(() => void poll(), 800)
        } else {
          setResponse(null)
          setError(status.error || 'Analysis failed. Retry the analysis to try again.')
          setLoading(false)
        }
      } catch (requestError) {
        if (!active) return
        if (requestError instanceof Error && requestError.message.includes('HTTP 404')) {
          setRun(null); setResponse(null); setError(''); setLoading(false)
        } else {
          setError(requestError instanceof Error ? requestError.message : 'Could not load uploaded dataset analysis.')
          setLoading(false)
        }
      }
    }
    void poll()
    return () => { active = false; if (timer !== undefined) window.clearTimeout(timer) }
  }, [dataset.dataset_id, offset, refresh])

  async function analyze() {
    setStarting(true)
    setLoading(true)
    setError('')
    try {
      const result = await startUploadedDatasetAnalysis(dataset.dataset_id)
      setRun(result)
      setRefresh((value) => value + 1)
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Could not start model analysis.')
      setLoading(false)
    } finally { setStarting(false) }
  }

  const selected = response?.items.find((item) => item.source_row === selectedRow) ?? null
  const progress = run && run.rows_total > 0 ? Math.min(100, Math.round(run.rows_scanned / run.rows_total * 100)) : 0
  return <section className="workspace-page uploaded-transactions-page">
    <div className="page-section-heading"><div><h2>{dataset.filename}</h2><p>Model results and CSV labels from this upload only.</p></div><button className="button-secondary" disabled={!dataset.supports.transaction_risk || starting || run?.status === 'running' || run?.status === 'queued' || run?.status === 'completed'} onClick={() => void analyze()} type="button">{starting || run?.status === 'queued' || run?.status === 'running' ? 'Analyzing…' : run?.status === 'completed' ? 'Analysis complete' : 'Analyze for Fraud'}</button></div>
    {!dataset.supports.transaction_risk && <p className="dataset-message error" role="alert">{dataset.supports.reasons.transaction_risk ?? 'This CSV is not compatible with the saved credit-card model.'}</p>}
    {run && (run.status === 'queued' || run.status === 'running') && <div className="upload-analysis-progress" role="status"><div><strong>{run.status === 'queued' ? 'Analysis queued' : 'Analyzing uploaded rows'}</strong><span>{run.rows_scanned.toLocaleString()} of {run.rows_total.toLocaleString()} rows checked · {run.rows_processed.toLocaleString()} predictions ready</span></div><progress max="100" value={progress} /></div>}
    {run?.status === 'completed' && response && <>
      <div className="dataset-stat-grid transaction-summary-grid"><DatasetStat label="Rows processed" value={run.rows_processed.toLocaleString()} detail="Real model inferences" /><DatasetStat label="Rows rejected" value={run.rows_rejected.toLocaleString()} detail="Missing or non-numeric features" /><DatasetStat label="Rows skipped" value={run.rows_skipped.toLocaleString()} detail="Blank feature rows" /><DatasetStat label="Predictions" value={response.total.toLocaleString()} detail={`${response.model_name} · ${response.model_version}`} /></div>
      <div className="transaction-workspace"><div className="panel workspace-table-panel"><div className="panel-heading"><div><h3>Uploaded transaction predictions</h3><p>{dataset.filename} · results are isolated to dataset {dataset.dataset_id.slice(0, 8)}</p></div></div><div className="table-scroll"><table className="workspace-table dataset-table"><thead><tr><th>CSV ROW</th><th>AMOUNT</th><th>TIME · SECONDS</th><th>MODEL RESULT</th><th>RECORDED CSV LABEL</th></tr></thead><tbody>{response.items.map((item) => <tr aria-selected={selectedRow === item.source_row} className={selectedRow === item.source_row ? 'selected-row' : ''} key={`${dataset.dataset_id}-${item.source_row}`} onClick={() => setSelectedRow(item.source_row)}><td><strong>Row {item.source_row}</strong></td><td className="amount-cell">{item.Amount.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td><td>{item.Time.toLocaleString()}</td><td><span className={`class-label ${item.is_fraud ? 'class-1' : 'class-0'}`}>{item.risk_level} · {(item.probability * 100).toFixed(2)}%</span></td><td>{item.recorded_csv_label ?? 'Not present'}</td></tr>)}</tbody></table></div><div className="table-footer"><span>Showing {response.items.length ? `${response.offset + 1}–${response.offset + response.items.length} of ${response.total.toLocaleString()}` : '0 rows'}</span><div className="pagination"><button disabled={loading || offset === 0} onClick={() => { setLoading(true); setOffset((value) => Math.max(0, value - limit)) }} type="button" aria-label="Previous prediction page">‹</button><span>{Math.floor(offset / limit) + 1}</span><button disabled={loading || offset + response.items.length >= response.total} onClick={() => { setLoading(true); setOffset((value) => value + limit) }} type="button" aria-label="Next prediction page">›</button></div></div></div>
        {selected && <aside className="panel detail-panel workspace-detail"><h2>Transaction details</h2><div className="dataset-detail-id">{dataset.dataset_id}:row:{String(selected.source_row).padStart(9, '0')}</div><div className="detail-amount compact"><strong>{selected.Amount.toLocaleString(undefined, { maximumFractionDigits: 2 })}</strong><span>Amount · dataset units</span></div><div className="detail-section-title">TRANSACTION INFORMATION</div><DetailRow label="Source row" value={String(selected.source_row)} /><DetailRow label="Time" value={`${selected.Time.toLocaleString()} seconds`} /><DetailRow label="Recorded CSV label" value={selected.recorded_csv_label ?? 'Not present'} /><div className="detail-section-title">MODEL PREDICTION</div><DetailRow label="Result" value={selected.is_fraud ? 'Flagged at configured threshold' : 'Below configured threshold'} /><DetailRow label="Estimated risk" value={`${(selected.probability * 100).toFixed(2)}% · ${selected.risk_level}`} /><DetailRow label="Model" value={`${response.model_name} ${response.model_version}`} /></aside>}
      </div>
    </>}
    {run?.status === 'completed' && response?.items.length === 0 && <p className="dataset-message">No rows had valid model features. The uploaded file’s original values remain unchanged.</p>}
    {error && <p className="dataset-message error" role="alert">{error}</p>}
    {!run && !loading && dataset.supports.transaction_risk && !error && <p className="dataset-message">Ready to analyze all {dataset.row_count.toLocaleString()} rows using the saved credit-card model. Missing or non-numeric feature rows will be reported and excluded from inference.</p>}
  </section>
}

function PreloadedTransactionsPage() {
  const [search, setSearch] = useState('')
  const [minAmount, setMinAmount] = useState('')
  const [maxAmount, setMaxAmount] = useState('')
  const [classFilter, setClassFilter] = useState('')
  const [sortBy, setSortBy] = useState<TransactionSortField>('source_row')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')
  const [offset, setOffset] = useState(0)
  const [response, setResponse] = useState<TransactionsResponse | null>(null)
  const [summary, setSummary] = useState<TransactionSummaryResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [summaryError, setSummaryError] = useState('')
  const [selectedId, setSelectedId] = useState('')
  const [selectedDetail, setSelectedDetail] = useState<DatasetTransaction | null>(null)
  const [detailError, setDetailError] = useState<{ transactionId: string; message: string } | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [summaryRefresh, setSummaryRefresh] = useState(0)
  const [detailRefresh, setDetailRefresh] = useState(0)
  const [rowPrediction, setRowPrediction] = useState<{ transactionId: string; result: PredictionResponse } | null>(null)
  const [predictionLoading, setPredictionLoading] = useState(false)
  const [predictionError, setPredictionError] = useState('')
  const limit = 50
  const items = response?.items ?? []
  const currentDetail = selectedDetail?.transaction_id === selectedId ? selectedDetail : null
  const currentDetailError = detailError?.transactionId === selectedId ? detailError.message : ''
  const detailLoading = Boolean(selectedId) && !currentDetail && !currentDetailError

  useEffect(() => {
    let active = true
    const loadTransactions = async () => {
      setLoading(true)
      setError('')
      try {
        const result = await getTransactions({
          limit,
          offset,
          search: search.trim() || undefined,
          min_amount: minAmount === '' ? undefined : Number(minAmount),
          max_amount: maxAmount === '' ? undefined : Number(maxAmount),
          class_label: classFilter === '0' ? 0 : classFilter === '1' ? 1 : undefined,
          sort_by: sortBy,
          sort_order: sortOrder,
        })
        if (!active) return
        setResponse(result)
        setSelectedId((current) =>
          result.items.some((item) => item.transaction_id === current)
            ? current
            : result.items[0]?.transaction_id ?? '',
        )
      } catch (requestError) {
        if (active) {
          setResponse(null)
          setSelectedId('')
          setError(requestError instanceof Error ? requestError.message : 'Unable to load transactions.')
        }
      } finally {
        if (active) setLoading(false)
      }
    }
    void loadTransactions()
    return () => {
      active = false
    }
  }, [classFilter, maxAmount, minAmount, offset, refreshKey, search, sortBy, sortOrder])

  useEffect(() => {
    let active = true
    void getTransactionSummary().then((result) => {
      if (active) {
        setSummary(result)
        setSummaryError('')
      }
    }).catch((requestError: unknown) => {
      if (active) {
        setSummary(null)
        setSummaryError(requestError instanceof Error ? requestError.message : 'Unable to load dataset summary.')
      }
    })
    return () => {
      active = false
    }
  }, [summaryRefresh])

  useEffect(() => {
    if (!selectedId) {
      return
    }
    let active = true
    void getTransaction(selectedId).then((detail) => {
      if (active) setSelectedDetail(detail)
    }).catch((requestError: unknown) => {
      if (active) {
        setDetailError({
          transactionId: selectedId,
          message: requestError instanceof Error ? requestError.message : 'Unable to load transaction details.',
        })
      }
    })
    return () => {
      active = false
    }
  }, [detailRefresh, selectedId])

  function changeSort(field: TransactionSortField) {
    if (sortBy === field) {
      setSortOrder((current) => current === 'asc' ? 'desc' : 'asc')
    } else {
      setSortBy(field)
      setSortOrder('asc')
    }
    setOffset(0)
  }

  async function runSelectedPrediction() {
    if (!selectedId) return
    setPredictionLoading(true)
    setPredictionError('')
    try {
      const result = await predictDatasetTransaction(selectedId)
      setRowPrediction({ transactionId: selectedId, result })
    } catch (requestError) {
      setPredictionError(requestError instanceof Error ? requestError.message : 'Unable to run model prediction.')
    } finally { setPredictionLoading(false) }
  }

  function sortHeading(field: TransactionSortField, label: string) {
    return (
      <th aria-sort={sortBy === field ? (sortOrder === 'asc' ? 'ascending' : 'descending') : 'none'}>
        <button className="dataset-sort-button" onClick={() => changeSort(field)} type="button">
          {label}{sortBy === field && <span>{sortOrder === 'asc' ? '↑' : '↓'}</span>}
        </button>
      </th>
    )
  }

  return (
    <section className="workspace-page">
      <div className="dataset-stat-grid transaction-summary-grid">
        <DatasetStat label="Dataset records" value={summary ? summary.record_count.toLocaleString() : '—'} detail={summary ? `${summary.dataset_version} · ${summary.data_source}` : summaryError ? 'Summary unavailable' : 'Loading backend summary'} />
        <DatasetStat label="Rows labeled fraud" value={summary ? summary.positive_count.toLocaleString() : '—'} detail={summary ? `Share: ${(summary.positive_rate * 100).toFixed(3)}%` : summaryError ? 'Summary unavailable' : 'From the CSV'} />
        <DatasetStat label="Rows not labeled fraud" value={summary ? summary.negative_count.toLocaleString() : '—'} detail={summary ? 'Recorded CSV labels' : summaryError ? 'Summary unavailable' : 'From the CSV'} />
        <DatasetStat label="Total Amount" value={summary ? summary.total_amount.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '—'} detail="Dataset units; currency not specified" />
      </div>
      {summaryError && (
        <p className="inline-error" role="alert">
          Dataset summary unavailable: {summaryError}{' '}
          <button className="text-button" onClick={() => setSummaryRefresh((current) => current + 1)} type="button">Retry</button>
        </p>
      )}

      <div className={`transaction-workspace ${selectedId ? '' : 'no-selection'}`}>
        <div className="panel workspace-table-panel">
          <div className="dataset-toolbar">
            <label className="workspace-search dataset-search">
              <Search size={15} />
              <input aria-label="Search transaction ID substring" onChange={(event) => { setSearch(event.target.value); setOffset(0) }} placeholder="Search transaction ID…" value={search} />
            </label>
            <label className="dataset-filter">
              <span>Min amount</span>
              <input aria-label="Minimum transaction amount" min="0" onChange={(event) => { setMinAmount(event.target.value); setOffset(0) }} type="number" value={minAmount} />
            </label>
            <label className="dataset-filter">
              <span>Max amount</span>
              <input aria-label="Maximum transaction amount" min="0" onChange={(event) => { setMaxAmount(event.target.value); setOffset(0) }} type="number" value={maxAmount} />
            </label>
            <label className="risk-filter">
              <SlidersHorizontal size={14} />
              <select aria-label="Filter by recorded CSV label" onChange={(event) => { setClassFilter(event.target.value); setOffset(0) }} value={classFilter}>
                <option value="">All Class labels</option>
                <option value="0">Labeled legitimate</option>
                <option value="1">Labeled fraud</option>
              </select>
            </label>
          </div>
          <div className="table-scroll">
            <table className="workspace-table dataset-table">
              <thead>
                <tr>
                  {sortHeading('source_row', 'TRANSACTION ID / ROW')}
                  {sortHeading('Amount', 'AMOUNT')}
                  {sortHeading('Time', 'TIME · SECONDS')}
                  {sortHeading('Class', 'CSV LABEL')}
                </tr>
              </thead>
              <tbody>
                {!error && items.map((item) => (
                  <tr
                    aria-selected={selectedId === item.transaction_id}
                    className={selectedId === item.transaction_id ? 'selected-row' : ''}
                    key={item.transaction_id}
                    onClick={() => { setSelectedId(item.transaction_id); setRowPrediction(null); setPredictionError('') }}
                  >
                    <td><strong>{item.transaction_id}</strong><small>Row {item.source_row}</small></td>
                    <td className="amount-cell">{item.Amount.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                    <td>{item.Time.toLocaleString()} s</td>
                    <td><span className={`class-label class-${item.Class}`}>{displayObservedLabel(item.Class)}</span></td>
                  </tr>
                ))}
                {!loading && !error && items.length === 0 && (
                  <tr><td className="empty-table" colSpan={4}>No backend records match these filters.</td></tr>
                )}
                {loading && <tr><td className="dataset-message" colSpan={4}>Loading backend transactions…</td></tr>}
                {error && (
                  <tr><td className="dataset-message error" colSpan={4} role="alert">
                    {error}{' '}
                    <button className="text-button" onClick={() => setRefreshKey((current) => current + 1)} type="button">Retry</button>
                  </td></tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="table-footer">
            <span>
              {response ? (
                items.length > 0
                  ? <>Showing <strong>{response.offset + 1}–{response.offset + items.length}</strong> of {response.total.toLocaleString()} records</>
                  : <>Showing <strong>0</strong> of {response.total.toLocaleString()} records</>
              ) : 'Awaiting backend dataset'}
            </span>
            <div className="pagination">
              <button disabled={loading || offset === 0} onClick={() => setOffset((current) => Math.max(0, current - limit))} type="button" aria-label="Previous transaction page">‹</button>
              <span>{response ? `${Math.floor(response.offset / Math.max(response.limit, 1)) + 1}` : '—'}</span>
              <button disabled={loading || !response || response.offset + response.items.length >= response.total} onClick={() => setOffset((current) => current + limit)} type="button" aria-label="Next transaction page">›</button>
            </div>
          </div>
        </div>
        {selectedId && (
          <aside className="panel detail-panel workspace-detail" aria-label="Selected backend transaction details">
            <h2>Transaction details</h2>
            <div className="dataset-detail-id">{selectedId}</div>
            {detailLoading && <p className="dataset-message">Loading transaction details…</p>}
            {currentDetailError && (
              <p className="dataset-message error" role="alert">
                {currentDetailError}{' '}
                <button
                  className="text-button"
                  onClick={() => {
                    setDetailError(null)
                    setDetailRefresh((current) => current + 1)
                  }}
                  type="button"
                >Retry</button>
              </p>
            )}
            {currentDetail && (
              <>
                <div className="detail-amount"><strong>{currentDetail.Amount.toLocaleString(undefined, { maximumFractionDigits: 2 })}</strong><span>Amount · dataset units (currency unspecified)</span></div>
                <div className="detail-section-title">TRANSACTION INFORMATION</div>
                <DetailRow label="Dataset version" value={currentDetail.dataset_version} />
                <DetailRow label="Source row" value={String(currentDetail.source_row)} />
                <DetailRow label="Time" value={`${currentDetail.Time.toLocaleString()} seconds`} />
                <DetailRow label="Recorded CSV label" value={displayObservedLabel(currentDetail.Class)} />
                <button className="button-secondary model-retry" disabled={predictionLoading} onClick={() => void runSelectedPrediction()} type="button">
                  {predictionLoading ? 'Running model…' : 'Run model on this row'}
                </button>
                {predictionError && <p className="dataset-message error" role="alert">{predictionError}</p>}
                {rowPrediction?.transactionId === selectedId && <div className="uploaded-model-result"><div className="detail-section-title">MODEL PREDICTION</div><DetailRow label="Result" value={rowPrediction.result.is_fraud ? 'Flagged at configured threshold' : 'Below configured threshold'} /><DetailRow label="Estimated risk" value={`${(rowPrediction.result.probability * 100).toFixed(2)}% · ${rowPrediction.result.risk_level}`} /></div>}
              </>
            )}
          </aside>
        )}
      </div>
    </section>
  )
}

function DatasetStat({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <article className="panel dataset-stat">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  )
}

function InvestigationsPage() {
  const [query, setQuery] = useState('')
  const [localCases, setLocalCases] = useState<LocalInvestigation[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [availableTransactions, setAvailableTransactions] = useState<DatasetTransaction[]>([])
  const [transactionsLoading, setTransactionsLoading] = useState(true)
  const [transactionsError, setTransactionsError] = useState('')
  const [transactionRefresh, setTransactionRefresh] = useState(0)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [selectedTransactionId, setSelectedTransactionId] = useState('')
  const [newSeverity, setNewSeverity] = useState<LocalInvestigation['severity']>('Medium')
  const [newNotes, setNewNotes] = useState('')
  const [caseError, setCaseError] = useState('')
  const [caseLoading, setCaseLoading] = useState(true)

  useEffect(() => {
    let active = true
    void getInvestigationCases().then((result) => {
      if (active) setLocalCases(result.items.map(toLocalInvestigation))
    }).catch((error: unknown) => {
      if (active) setCaseError(error instanceof Error ? error.message : 'Unable to load saved cases.')
    }).finally(() => { if (active) setCaseLoading(false) })
    return () => { active = false }
  }, [])

  useEffect(() => {
    let active = true
    const loadTransactions = async () => {
      setTransactionsLoading(true)
      setTransactionsError('')
      try {
        const result = await getTransactions({
          limit: 50,
          offset: 0,
          sort_by: 'source_row',
          sort_order: 'desc',
        })
        if (!active) return
        setAvailableTransactions(result.items)
        setSelectedTransactionId((current) =>
          result.items.some((item) => item.transaction_id === current)
            ? current
            : result.items[0]?.transaction_id ?? '',
        )
      } catch (requestError) {
        if (active) {
          setTransactionsError(requestError instanceof Error ? requestError.message : 'Unable to load backend transactions.')
        }
      } finally {
        if (active) setTransactionsLoading(false)
      }
    }
    void loadTransactions()
    return () => {
      active = false
    }
  }, [transactionRefresh])

  const filteredLocalCases = localCases.filter((caseItem) =>
    [
      caseItem.id,
      caseItem.transaction.transaction_id,
      caseItem.severity,
      caseItem.status,
      caseItem.notes,
      String(caseItem.transaction.Class),
    ].join(' ')
      .toLowerCase()
      .includes(query.trim().toLowerCase()),
  )
  const selected = localCases.find((caseItem) => caseItem.id === selectedId)
  const selectedTransaction = availableTransactions.find(
    (transaction) => transaction.transaction_id === selectedTransactionId,
  )
  const classSemantics = 'This is the label recorded in the source CSV, not a model prediction.'
  const openCount = localCases.filter((caseItem) => caseItem.status === 'Open').length
  const reviewCount = localCases.filter((caseItem) => caseItem.status === 'In Review').length
  const resolvedCount = localCases.filter((caseItem) => caseItem.status === 'Resolved').length

  async function createInvestigation(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!selectedTransaction) return
    setCaseError('')
    try {
      const saved = await createInvestigationCase({ transaction_id: selectedTransaction.transaction_id, severity: newSeverity, notes: newNotes.trim() })
      const newCase = toLocalInvestigation(saved)
      setLocalCases((current) => [newCase, ...current])
      setSelectedId(newCase.id)
      setNewNotes('')
      setShowCreateForm(false)
    } catch (error) { setCaseError(error instanceof Error ? error.message : 'Unable to save case.') }
  }

  async function updateSelectedCase(update: Partial<Pick<LocalInvestigation, 'status' | 'notes'>>) {
    if (!selected) return
    setCaseError('')
    try {
      const saved = await updateInvestigationCase(selected.id, update)
      const updated = toLocalInvestigation(saved)
      setLocalCases((current) => current.map((caseItem) => caseItem.id === selected.id ? updated : caseItem))
    } catch (error) { setCaseError(error instanceof Error ? error.message : 'Unable to update case.') }
  }

  return (
    <section className="workspace-page">
      <div className="page-section-heading">
        <div>
          <h2>Investigation workspace</h2>
          <p>Create analyst cases linked to real backend dataset rows. Cases are stored in the backend database.</p>
        </div>
        <button className="button-secondary" onClick={() => setShowCreateForm((current) => !current)} type="button">
          {showCreateForm ? 'Cancel' : 'Create case'}
        </button>
      </div>
      <div className="case-summary-grid">
        <SummaryCard label="Open cases" value={String(openCount)} detail="Persisted backend cases" icon={Fingerprint} />
        <SummaryCard label="In review" value={String(reviewCount)} detail="Persisted backend cases" icon={Activity} />
        <SummaryCard label="Resolved" value={String(resolvedCount)} detail="Persisted backend cases" icon={ShieldCheck} />
      </div>
      {caseError && <div className="dataset-error" role="alert">{caseError}</div>}

      {showCreateForm && (
        <form className="panel case-create-form" onSubmit={createInvestigation}>
          <div className="case-form-heading">
            <div><h3>Create investigation case</h3><p>Choose an actual backend dataset record and add analyst-provided triage notes.</p></div>
          <span className="demo-tag">SAVED TO BACKEND</span>
          </div>
          {transactionsLoading ? (
            <p className="dataset-message">Loading recent backend transaction records…</p>
          ) : transactionsError ? (
            <p className="dataset-message error" role="alert">
              {transactionsError}{' '}
              <button className="text-button" onClick={() => setTransactionRefresh((current) => current + 1)} type="button">Retry transaction load</button>
            </p>
          ) : availableTransactions.length === 0 ? (
            <p className="dataset-message">No backend transaction rows are available to link.</p>
          ) : (
            <>
              <label className="case-form-field">
                <span>Backend transaction</span>
                <select onChange={(event) => setSelectedTransactionId(event.target.value)} value={selectedTransactionId}>
                  {availableTransactions.map((transaction) => (
                    <option key={transaction.transaction_id} value={transaction.transaction_id}>
                      {transaction.transaction_id} · row {transaction.source_row} · amount {transaction.Amount} · {displayObservedLabel(transaction.Class)}
                    </option>
                  ))}
                </select>
              </label>
              {selectedTransaction && (
                <div className="case-evidence-preview">
                  <strong>Selected transaction evidence</strong>
                  <span>Row {selectedTransaction.source_row}</span>
                  <span>Time {selectedTransaction.Time.toLocaleString()} seconds</span>
                  <span>Amount {selectedTransaction.Amount.toLocaleString(undefined, { maximumFractionDigits: 2 })} (dataset units)</span>
                  <span>Observed CSV label: {displayObservedLabel(selectedTransaction.Class)}</span>
                  <small>{classSemantics} No PCA features, model probability, or account links are supplied by this endpoint.</small>
                </div>
              )}
              <div className="case-form-row">
                <label className="case-form-field">
                  <span>Analyst severity</span>
                  <select onChange={(event) => setNewSeverity(event.target.value as LocalInvestigation['severity'])} value={newSeverity}>
                    <option>Low</option>
                    <option>Medium</option>
                    <option>High</option>
                  </select>
                </label>
                <label className="case-form-field case-notes-field">
                  <span>Initial notes</span>
                  <textarea onChange={(event) => setNewNotes(event.target.value)} placeholder="Add notes about what should be reviewed." value={newNotes} />
                </label>
              </div>
              <button className="analysis-submit case-create-submit" disabled={!selectedTransaction} type="submit">Create case</button>
            </>
          )}
        </form>
      )}

      <div className="page-section-heading subsection-heading">
        <div><h2>Saved investigation cases</h2><p>Linked to actual backend transaction rows; edits persist in SQLite.</p></div>
      </div>
      <div className={`case-workspace ${selected ? '' : 'no-selection'}`}>
        <div className="panel workspace-table-panel">
          <div className="workspace-toolbar">
            <label className="workspace-search">
              <Search size={15} />
                  <input aria-label="Search persisted investigation cases" onChange={(event) => setQuery(event.target.value)} placeholder="Search cases, transactions, notes…" value={query} />
            </label>
          </div>
          <div className="table-scroll">
            <table className="workspace-table case-table local-case-table">
              <thead><tr><th>CASE ID</th><th>TRANSACTION</th><th>CLASS LABEL</th><th>SEVERITY</th><th>STATUS</th><th>UPDATED</th></tr></thead>
              <tbody>
                {filteredLocalCases.map((caseItem) => (
                  <tr
                    aria-selected={selectedId === caseItem.id}
                    className={selectedId === caseItem.id ? 'selected-row' : ''}
                    key={caseItem.id}
                    onClick={() => setSelectedId(caseItem.id)}
                  >
                    <td>{caseItem.id}</td>
                    <td>{caseItem.transaction.transaction_id}</td>
                    <td><span className={`class-label class-${caseItem.transaction.Class}`}>{displayObservedLabel(caseItem.transaction.Class)}</span></td>
                    <td><span className={`risk-badge ${caseItem.severity.toLowerCase()}`}><i />{caseItem.severity}</span></td>
                    <td><span className={`case-status ${caseItem.status.toLowerCase().replace(' ', '-')}`}>{caseItem.status}</span></td>
                    <td>{caseItem.updatedAt}</td>
                  </tr>
                ))}
                {!transactionsLoading && !transactionsError && filteredLocalCases.length === 0 && (
                  <tr><td className="empty-table" colSpan={6}>{query ? 'No saved cases match your search.' : 'No saved cases yet.'}</td></tr>
                )}
                {(transactionsLoading || caseLoading) && localCases.length === 0 && <tr><td className="dataset-message" colSpan={6}>Loading persisted cases and backend dataset…</td></tr>}
              </tbody>
            </table>
          </div>
          <div className="table-footer"><span><strong>{filteredLocalCases.length}</strong> persisted cases</span></div>
        </div>
        {selected && (
          <aside className="panel detail-panel workspace-detail local-case-detail" aria-label="Temporary investigation case details">
            <span className="detail-overline">PERSISTED CASE</span>
            <div className="detail-id">{selected.id}</div>
            <div className="case-detail-flags">
              <span className={`risk-badge ${selected.severity.toLowerCase()}`}><i />{selected.severity} severity</span>
              <span className={`case-status ${selected.status.toLowerCase().replace(' ', '-')}`}>{selected.status}</span>
            </div>
            <div className="detail-section-title">LINKED BACKEND RECORD</div>
            <DetailRow label="Transaction" value={selected.transaction.transaction_id} />
            <DetailRow label="Dataset version" value={selected.transaction.dataset_version} />
            <DetailRow label="Source row" value={String(selected.transaction.source_row)} />
            <DetailRow label="Elapsed time" value={`${selected.transaction.Time.toLocaleString()} seconds`} />
            <DetailRow label="Amount" value={`${selected.transaction.Amount.toLocaleString(undefined, { maximumFractionDigits: 2 })} dataset units`} />
            <DetailRow label="Observed CSV label" value={displayObservedLabel(selected.transaction.Class)} />
            <DetailRow label="Model prediction" value="Unavailable · no feature vector supplied" />
            <DetailRow label="Related accounts" value="Not available in the dataset" />
            <p className="workspace-disclaimer">{classSemantics} The transaction API does not provide PCA feature values or a prediction probability, so no prediction was requested.</p>
            <label className="case-form-field case-detail-status">
              <span>Case status</span>
              <select onChange={(event) => updateSelectedCase({ status: event.target.value as LocalInvestigation['status'] })} value={selected.status}>
                <option>Open</option>
                <option>In Review</option>
                <option>Resolved</option>
              </select>
            </label>
            <label className="case-form-field case-detail-notes">
              <span>Analyst notes</span>
              <textarea onChange={(event) => updateSelectedCase({ notes: event.target.value })} value={selected.notes} />
            </label>
            <DetailRow label="Created (UTC)" value={selected.createdAt} />
            <DetailRow label="Last updated (UTC)" value={selected.updatedAt} />
            <p className="workspace-disclaimer">Case status and notes are persisted by the backend and remain available after reload.</p>
          </aside>
        )}
      </div>


    </section>
  )
}

function SummaryCard({
  label,
  value,
  detail,
  icon: Icon,
}: {
  label: string
  value: string
  detail: string
  icon: LucideIcon
}) {
  return (
    <article className="panel case-summary-card">
      <span className="summary-icon"><Icon size={16} /></span>
      <span className="summary-label">{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  )
}

function DataLabPage({ onAnalyzeGraph, onAnalysisComplete }: { onAnalyzeGraph: (datasetId: string) => void; onAnalysisComplete: (datasetId: string) => void }) {
  const onAnalysisCompleteRef = useRef(onAnalysisComplete)
  useEffect(() => { onAnalysisCompleteRef.current = onAnalysisComplete }, [onAnalysisComplete])
  const [history, setHistory] = useState<UploadedDataset[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [preview, setPreview] = useState<DatasetPreview | null>(null)
  const [loadingHistory, setLoadingHistory] = useState(true)
  const [historyRefresh, setHistoryRefresh] = useState(0)
  const [uploading, setUploading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [previewError, setPreviewError] = useState<{ datasetId: string; message: string } | null>(null)
  const [uploadError, setUploadError] = useState('')
  const [prediction, setPrediction] = useState<{ row: number; result: UploadedPrediction } | null>(null)
  const [predictingRow, setPredictingRow] = useState<number | null>(null)
  const [predictionError, setPredictionError] = useState('')
  const [analysisRun, setAnalysisRun] = useState<UploadedAnalysisRun | null>(null)
  const [analysisError, setAnalysisError] = useState('')
  const [startingAnalysis, setStartingAnalysis] = useState(false)
  const analysisStatus = analysisRun?.status
  const selected = history.find((item) => item.dataset_id === selectedId) ?? null
  const currentPreview = preview?.dataset_id === selectedId ? preview : null
  const currentPreviewError = previewError?.datasetId === selectedId ? previewError.message : ''
  const loadingPreview = Boolean(selectedId) && !currentPreview && !currentPreviewError

  useEffect(() => {
    let active = true
    void getUploadedDatasets().then((result: UploadedDatasetList) => {
      if (!active) return
      setHistory(result.items)
      setSelectedId((current) => result.items.some((item) => item.dataset_id === current) ? current : '')
      setHistoryError('')
    }).catch((error: unknown) => {
      if (!active) return
      setHistory([])
      setSelectedId('')
      setHistoryError(error instanceof Error ? error.message : 'Uploaded dataset history is unavailable.')
    }).finally(() => { if (active) setLoadingHistory(false) })
    return () => { active = false }
  }, [historyRefresh])

  useEffect(() => {
    if (!selectedId) return
    let active = true
    void getUploadedDatasetPreview(selectedId, 10, 0).then((result) => {
      if (active) setPreview(result)
    }).catch((error: unknown) => {
      if (active) {
        setPreviewError({ datasetId: selectedId, message: error instanceof Error ? error.message : 'CSV preview is unavailable.' })
      }
    })
    return () => { active = false }
  }, [selectedId])

  useEffect(() => {
    if (!selectedId) return
    let active = true
    void getUploadedDatasetAnalysis(selectedId).then((run) => { if (active) setAnalysisRun(run) }).catch(() => { if (active) setAnalysisRun(null) })
    return () => { active = false }
  }, [selectedId])

  useEffect(() => {
    if (!selectedId || !analysisStatus || !['queued', 'running'].includes(analysisStatus)) return
    let active = true
    let timer: number | undefined
    const poll = async () => {
      try {
        const run = await getUploadedDatasetAnalysis(selectedId)
        if (!active) return
        setAnalysisRun(run)
        if (run.status === 'completed') onAnalysisCompleteRef.current(selectedId)
        else if (run.status === 'queued' || run.status === 'running') timer = window.setTimeout(() => void poll(), 800)
      } catch (error) {
        if (active) setAnalysisError(error instanceof Error ? error.message : 'Analysis status is unavailable.')
      }
    }
    timer = window.setTimeout(() => void poll(), 800)
    return () => { active = false; if (timer !== undefined) window.clearTimeout(timer) }
  }, [analysisStatus, selectedId])

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setUploadError('')
    setPrediction(null)
    setAnalysisRun(null)
    setAnalysisError('')
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setUploadError('Choose a CSV file. Other file types are not supported.')
      return
    }
    if (file.size > 1_000_000_000) {
      setUploadError('This file is larger than 1 GB. Choose a smaller CSV file.')
      return
    }
    setUploading(true)
    try {
      const uploaded = await uploadDataset(file)
      setHistory((current) => [uploaded, ...current.filter((item) => item.dataset_id !== uploaded.dataset_id)])
      setSelectedId(uploaded.dataset_id)
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'The CSV could not be uploaded or checked.')
    } finally { setUploading(false) }
  }

  async function analyzeSelectedDataset() {
    if (!selected) return
    if (!selected.supports.transaction_risk) {
      setAnalysisError(selected.supports.reasons.transaction_risk ?? 'This CSV does not match the saved model schema.')
      return
    }
    setStartingAnalysis(true)
    setAnalysisError('')
    try {
      const run = await startUploadedDatasetAnalysis(selected.dataset_id)
      setAnalysisRun(run)
      if (run.status === 'completed') onAnalysisComplete(selected.dataset_id)
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : 'Could not start model analysis.')
    } finally { setStartingAnalysis(false) }
  }

  async function checkRowRisk(sourceRow: number) {
    if (!selected) return
    setPredictingRow(sourceRow)
    setPrediction(null)
    setPredictionError('')
    try {
      const result = await predictUploadedTransaction(selected.dataset_id, sourceRow)
      setPrediction({ row: sourceRow, result })
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : 'Risk could not be checked for this row.')
    } finally { setPredictingRow(null) }
  }

  async function removeSelectedDataset() {
    if (!selected || !window.confirm(`Delete ${selected.filename} and its saved upload record? This cannot be undone.`)) return
    setUploadError('')
    try {
      await deleteUploadedDataset(selected.dataset_id)
      setHistory((current) => current.filter((item) => item.dataset_id !== selected.dataset_id))
      setSelectedId('')
      setPreview(null)
      setPrediction(null)
      setHistoryRefresh((value) => value + 1)
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'The uploaded CSV could not be deleted.')
    }
  }

  function exportDatasetReport() {
    if (!selected) return
    const report = {
      dataset_id: selected.dataset_id,
      filename: selected.filename,
      uploaded_at: selected.uploaded_at,
      rows_read: selected.rows_read,
      rows_processed: selected.rows_processed,
      rows_rejected: selected.rows_rejected,
      rows_skipped: selected.rows_skipped,
      columns: selected.columns,
      dtypes: selected.dtypes,
      missing_values: selected.missing_values,
      duplicate_rows: selected.duplicate_rows,
    }
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a')
    link.href = url
    link.download = `${selected.filename.replace(/\.csv$/i, '')}-quality-report.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <section className="workspace-page data-lab-page">
      <div className="page-section-heading">
        <div><h2>Work with a CSV file</h2><p>Upload is stored locally by the backend. Your original file is kept unchanged and each upload receives its own ID.</p></div>
        <div className="data-lab-actions">
          <button className="button-secondary" onClick={exportDatasetReport} disabled={!selected} type="button"><Download size={15} /> Export quality report</button>
          <button className="button-secondary danger-button" onClick={() => void removeSelectedDataset()} disabled={!selected} type="button"><Trash2 size={15} /> Delete upload</button>
        </div>
      </div>

      <article className="panel data-step-panel">
        <div className="data-step-heading"><span className="data-step-number">1</span><div><h3>Upload your data</h3><p>Choose a CSV up to 1 GB. We’ll show a preview and which checks are available for its columns.</p></div></div>
        <label className="data-upload-control"><span>{uploading ? 'Uploading and checking file…' : 'Choose CSV file'}</span><input accept=".csv,text/csv" disabled={uploading} onChange={(event) => void handleUpload(event)} type="file" /></label>
        {uploading && <p className="dataset-message" role="status">Reading the file in chunks and checking its rows. Large files can take a little time.</p>}
        {uploadError && <p className="dataset-message error" role="alert">{uploadError}</p>}
      </article>

      <div className="data-step-heading data-history-heading"><span className="data-step-number">2</span><div><h3>Understand your data</h3><p>History below comes only from uploaded CSVs recorded by this backend.</p></div>
        {historyError && <button className="text-button" onClick={() => { setHistory([]); setSelectedId(''); setLoadingHistory(true); setHistoryRefresh((value) => value + 1) }} type="button">Retry</button>}
      </div>
      {historyError && <p className="dataset-message error" role="alert">{historyError}</p>}
      {loadingHistory ? <article className="panel"><p className="dataset-message">Loading saved upload history…</p></article> : history.length === 0 ? (
        <article className="panel dataset-empty-state"><h3>No CSV files uploaded</h3><p>Choose a file above to see its actual rows, columns, missing values, repeated rows, and compatible checks.</p></article>
      ) : (
        <div className="data-lab-layout">
          <aside className="panel data-history-panel"><h3>Uploaded files</h3><div className="data-history-list">{history.map((item) => <button className={`data-history-item ${selectedId === item.dataset_id ? 'active' : ''}`} key={item.dataset_id} onClick={() => { setPrediction(null); setSelectedId(item.dataset_id) }} type="button"><strong>{item.filename}</strong><small>{item.row_count.toLocaleString()} rows · {item.dataset_id.slice(0, 8)}</small></button>)}</div></aside>
          <div className="data-lab-detail">
            {selected && <>
              <article className="panel data-quality-panel">
                <div className="panel-heading"><div><h3>{selected.filename}</h3><p>Uploaded {new Date(selected.uploaded_at).toLocaleString()} · ID {selected.dataset_id}</p></div><span className="demo-tag">CSV SOURCE</span></div>
                <div className="data-quality-grid">
                  <DatasetStat label="Rows read" value={selected.rows_read.toLocaleString()} detail={`${selected.rows_processed.toLocaleString()} processed`} />
                  <DatasetStat label="Rows rejected" value={selected.rows_rejected.toLocaleString()} detail={`${selected.rows_skipped.toLocaleString()} skipped`} />
                  <DatasetStat label="Columns" value={String(selected.columns.length)} detail="Original CSV headers" />
                  <DatasetStat label="Missing values" value={selected.total_missing_values.toLocaleString()} detail={`${selected.duplicate_rows.toLocaleString()} repeated rows`} />
                </div>
                <p className="workspace-disclaimer">All readable rows were processed for this quality summary. Missing values are counted by column below. A repeated row is an exact duplicate across all CSV fields.</p>
                <div className="data-column-list">{selected.columns.map((column) => <div className="data-column-item" key={column}><strong>{column}</strong><span>{selected.dtypes[column] ?? 'unknown type'}</span><small>{(selected.missing_values[column] ?? 0).toLocaleString()} missing</small></div>)}</div>
              </article>

              <article className="panel data-preview-panel">
                <div className="panel-heading"><div><h3>Actual file preview</h3><p>First 10 data rows from this upload.</p></div></div>
                {loadingPreview ? <p className="dataset-message">Loading rows from the uploaded CSV…</p> : currentPreviewError ? <p className="dataset-message error" role="alert">{currentPreviewError}</p> : currentPreview ? <div className="table-scroll"><table className="workspace-table"><thead><tr><th>CSV ROW</th>{currentPreview.columns.map((column) => <th key={column}>{column}</th>)}<th>CHECK RISK</th></tr></thead><tbody>{currentPreview.items.map((row, index) => {
                  const sourceRow = currentPreview.offset + index + 1
                  return <tr key={`${selected.dataset_id}-${sourceRow}`}><td className="data-preview-row-number">{String(sourceRow).padStart(6, '0')}</td>{currentPreview.columns.map((column) => <td key={column}>{row[column] === null || row[column] === undefined ? 'Missing' : String(row[column])}</td>)}<td><button className="text-button" disabled={!selected.supports.transaction_risk || predictingRow === sourceRow} onClick={() => void checkRowRisk(sourceRow)} type="button">{predictingRow === sourceRow ? 'Checking…' : 'Check risk'}</button></td></tr>
                })}</tbody></table></div> : <p className="dataset-message">Preview unavailable.</p>}
                {predictionError && <p className="dataset-message error" role="alert">{predictionError}</p>}
                {prediction && prediction.result.dataset_id === selected.dataset_id && <div className="case-reason-card"><strong>Estimated risk for CSV row {prediction.row}</strong><p>{(prediction.result.probability * 100).toFixed(2)}% · {prediction.result.risk_level} · model {prediction.result.model_name} {prediction.result.model_version}</p><small>This is a model estimate, not proof of fraud. The CSV's label, if present, is separate.</small></div>}
              </article>

              <article className="panel data-analysis-panel">
                <div className="data-step-heading"><span className="data-step-number">3</span><div><h3>Choose a check</h3><p>Only checks this upload can currently support are available.</p></div></div>
                <AnalysisOption title="Data quality" available={selected.supports.data_quality} detail="Available. Row counts, column types, missing cells, duplicates, and a preview were read from this CSV." />
                <AnalysisOption title="Transaction risk" available={selected.supports.transaction_risk} detail={selected.supports.transaction_risk ? `Available for this upload. The saved model will process ${selected.row_count.toLocaleString()} rows in bounded batches; invalid feature rows will be reported separately.` : selected.supports.reasons.transaction_risk ?? 'Not available for this CSV.'} action={selected.supports.transaction_risk ? <button className="button-secondary" disabled={startingAnalysis || analysisRun?.status === 'running' || analysisRun?.status === 'queued'} onClick={() => void analyzeSelectedDataset()} type="button">{startingAnalysis || analysisRun?.status === 'running' || analysisRun?.status === 'queued' ? 'Analyzing…' : analysisRun?.status === 'completed' ? 'View predictions' : 'Analyze for Fraud'}</button> : undefined} />
                <AnalysisOption title="Account connections" available={selected.supports.account_relationships} detail={selected.supports.reasons.account_relationships ?? 'Available. Explore connections and row evidence from this upload.'} action={selected.supports.account_relationships ? <button className="button-secondary" onClick={() => onAnalyzeGraph(selected.dataset_id)} type="button">Open account graph</button> : undefined} />
                <AnalysisOption title="Evaluation against labels" available={selected.supports.evaluation} detail={selected.supports.reasons.evaluation ?? 'Not available for uploaded files. The saved model report describes its separate holdout set.'} />
                {analysisRun && (analysisRun.status === 'queued' || analysisRun.status === 'running') && <div className="upload-analysis-progress" role="status"><div><strong>{analysisRun.status === 'queued' ? 'Analysis queued' : 'Analyzing uploaded rows'}</strong><span>{analysisRun.rows_scanned.toLocaleString()} of {analysisRun.rows_total.toLocaleString()} rows · {analysisRun.rows_processed.toLocaleString()} predictions ready</span></div><progress max="100" value={analysisRun.rows_total ? Math.round(analysisRun.rows_scanned / analysisRun.rows_total * 100) : 0} /></div>}
                {analysisRun?.status === 'completed' && <p className="dataset-message">Completed: {analysisRun.rows_processed.toLocaleString()} predictions, {analysisRun.rows_rejected.toLocaleString()} rejected rows, {analysisRun.rows_skipped.toLocaleString()} skipped rows.</p>}
                {(analysisError || (analysisRun?.status === 'failed' && analysisRun.error)) && <p className="dataset-message error" role="alert">{analysisError || analysisRun?.error}</p>}
              </article>
            </>}
          </div>
        </div>
      )}
    </section>
  )
}

function AnalysisOption({ title, available, detail, action }: { title: string; available: boolean; detail: string; action?: ReactNode }) {
  return <div className={`analysis-option ${available ? 'available' : 'unavailable'}`}><span className="analysis-option-status" aria-label={available ? 'Available' : 'Not available'}>{available ? 'Available' : 'Not available'}</span><div><strong>{title}</strong><p>{detail}</p></div>{action}</div>
}

function evaluationF1(precision: number | null, recall: number | null): number | null {
  return precision === null || recall === null || precision + recall === 0 ? null : 2 * precision * recall / (precision + recall)
}

function evaluationRatio(numerator: number, denominator: number): number | null {
  return denominator === 0 ? null : numerator / denominator
}

function formatEvaluationValue(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value) ? '—' : value.toFixed(4)
}

function EvaluationChart({ title, xLabel, yLabel, xValue, yValue, curveX, curveY, metricLabel, metricValue, baseline = false }: {
  title: string; xLabel: string; yLabel: string; xValue: number | null; yValue: number | null
  curveX?: number[]; curveY?: number[]; metricLabel: string; metricValue: number | null | undefined; baseline?: boolean
}) {
  const available = xValue !== null && yValue !== null
  const pointX = available ? 52 + xValue * 390 : 52
  const pointY = available ? 206 - yValue * 170 : 206
  const curvePoints = curveX && curveY ? curveX.map((value, index) => ({ x: value, y: curveY[index] })).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y) && point.x >= 0 && point.x <= 1 && point.y >= 0 && point.y <= 1) : []
  const curvePath = curvePoints.length > 1 ? curvePoints.map((point, index) => (index ? 'L' : 'M') + ' ' + (52 + point.x * 390).toFixed(2) + ' ' + (206 - point.y * 170).toFixed(2)).join(' ') : ''
  return <article className="panel performance-chart-panel">
    <div className="performance-card-heading"><h3>{title}</h3><span className="chart-metric-tag">{metricLabel} {formatEvaluationValue(metricValue)}</span></div>
    <svg className="performance-chart" viewBox="0 0 500 260" role="img" aria-label={title + (curvePath ? ', curve plotted from ' + curvePoints.length + ' saved-model score thresholds' : available ? ', one operating point from backend threshold counts' : ', threshold point unavailable')}>
      {[0, 1, 2, 3, 4].map((tick) => {
        const x = 52 + tick * 97.5
        const y = 206 - tick * 42.5
        return <g key={tick}><line x1={x} y1="36" x2={x} y2="206" className="performance-chart-grid" /><line x1="52" y1={y} x2="442" y2={y} className="performance-chart-grid" /><text x={x} y="226" textAnchor="middle">{(tick / 4).toFixed(1)}</text><text x="38" y={y + 4} textAnchor="end">{(tick / 4).toFixed(1)}</text></g>
      })}
      <line x1="52" y1="206" x2="442" y2="206" className="performance-chart-axis" /><line x1="52" y1="36" x2="52" y2="206" className="performance-chart-axis" />
      {baseline && <line x1="52" y1="206" x2="442" y2="36" className="performance-chart-baseline" />}
      {curvePath && <path d={curvePath} pathLength="1" className="performance-chart-curve" />}
      {available ? <><circle cx={pointX} cy={pointY} r="7" className="performance-chart-point" /><circle cx={pointX} cy={pointY} r="14" className="performance-chart-point-halo" /></> : !curvePath && <text x="247" y="126" textAnchor="middle" className="performance-chart-empty">Threshold point unavailable</text>}
      <text x="247" y="250" textAnchor="middle" className="performance-chart-label">{xLabel}</text><text x="14" y="122" textAnchor="middle" className="performance-chart-label" transform="rotate(-90 14 122)">{yLabel}</text>
    </svg>
    <p className="performance-chart-note">{curvePath ? 'Curve uses the saved model’s scores on the reported held-out data.' : available ? 'One operating point calculated from backend confusion counts; full curve values are unavailable.' : 'Curve points were not included in the backend report.'}</p>
  </article>
}

function ModelPerformancePage({ backendStatus, modelInfo, modelInfoError, onRetry }: {
  backendStatus: 'checking' | 'connected' | 'offline'; modelInfo: ModelInfoResponse | null; modelInfoError: string; onRetry: () => void
}) {
  const [evaluation, setEvaluation] = useState<EvaluationReportResponse | null>(null)
  const [evaluationLoading, setEvaluationLoading] = useState(true)
  const [evaluationError, setEvaluationError] = useState('')
  const [evaluationRefresh, setEvaluationRefresh] = useState(0)

  useEffect(() => {
    let active = true
    const loadEvaluation = async () => {
      setEvaluationLoading(true)
      setEvaluationError('')
      try {
        const report = await getEvaluationReport()
        if (active) setEvaluation(report)
      } catch (requestError) {
        if (active) setEvaluationError(requestError instanceof Error ? requestError.message : 'Unable to load evaluation report.')
      } finally {
        if (active) setEvaluationLoading(false)
      }
    }
    void loadEvaluation()
    return () => { active = false }
  }, [evaluationRefresh])

  const matrix = evaluation?.available ? evaluation.confusion_matrix : null
  const zeroPrecision = matrix ? evaluationRatio(matrix.tn, matrix.tn + matrix.fn) : null
  const zeroRecall = matrix ? evaluationRatio(matrix.tn, matrix.tn + matrix.fp) : null
  const onePrecision = matrix ? evaluationRatio(matrix.tp, matrix.tp + matrix.fp) : null
  const oneRecall = matrix ? evaluationRatio(matrix.tp, matrix.tp + matrix.fn) : null
  const classRows = matrix ? [
    { label: 'Class 0 · Normal', precision: zeroPrecision, recall: zeroRecall, f1: evaluationF1(zeroPrecision, zeroRecall), support: matrix.tn + matrix.fp },
    { label: 'Class 1 · Fraud', precision: onePrecision, recall: oneRecall, f1: evaluationF1(onePrecision, oneRecall), support: matrix.fn + matrix.tp },
  ] : []
  const totalSupport = classRows.reduce((sum, row) => sum + row.support, 0)
  const averageMetric = (key: 'precision' | 'recall' | 'f1', weighted: boolean): number | null => {
    const rows = classRows.filter((row) => row[key] !== null)
    if (!rows.length) return null
    if (!weighted) return rows.reduce((sum, row) => sum + (row[key] ?? 0), 0) / rows.length
    const support = rows.reduce((sum, row) => sum + row.support, 0)
    return support ? rows.reduce((sum, row) => sum + (row[key] ?? 0) * row.support, 0) / support : null
  }
  const point = matrix ? {
    fpr: evaluationRatio(matrix.fp, matrix.fp + matrix.tn),
    tpr: evaluationRatio(matrix.tp, matrix.tp + matrix.fn),
    precision: evaluationRatio(matrix.tp, matrix.tp + matrix.fp),
    recall: evaluationRatio(matrix.tp, matrix.tp + matrix.fn),
  } : null
  const metricCards = [
    { label: 'Precision', value: evaluation?.available ? evaluation.metrics?.precision : null, tone: 'precision' },
    { label: 'Recall', value: evaluation?.available ? evaluation.metrics?.recall : null, tone: 'recall' },
    { label: 'F1 score', value: evaluation?.available ? evaluation.metrics?.f1 : null, tone: 'f1' },
    { label: 'ROC-AUC', value: evaluation?.available ? evaluation.metrics?.roc_auc : null, tone: 'auc' },
  ]
  const refreshReport = () => setEvaluationRefresh((current) => current + 1)

  return <section className="workspace-page performance-page">
    <div className="performance-page-heading"><div className="performance-page-actions">
      <button className="button-secondary" disabled={backendStatus === 'checking'} onClick={onRetry} type="button"><RefreshCw size={14} />{backendStatus === 'checking' ? 'Checking…' : 'Refresh model status'}</button>
      <button className="button-primary" disabled={evaluationLoading} onClick={refreshReport} type="button"><Activity size={14} />{evaluationLoading ? 'Loading report…' : 'Refresh evaluation'}</button>
    </div></div>

    {(evaluationError || (evaluation && !evaluation.available)) && <div className="performance-report-alert" role="status"><div><strong>{evaluationError ? 'Evaluation report unavailable' : 'Evaluation has not been generated'}</strong><p>{evaluationError || evaluation?.reason || 'The backend has not supplied an evaluation report.'}</p></div><button className="text-button" onClick={refreshReport} type="button">Retry</button></div>}

    <div className="performance-overview-grid">
      <article className="panel performance-model-card"><div className="performance-card-heading"><div><span className={'model-state-dot ' + backendStatus} /><h3>Model information</h3></div><span className={'performance-live-status ' + backendStatus}>{backendStatus === 'connected' ? 'LIVE' : backendStatus === 'checking' ? 'CHECKING' : 'OFFLINE'}</span></div>
        {modelInfo ? <><div className="performance-model-identity"><span className="performance-model-icon"><Activity size={22} /></span><div><span>Model name</span><strong>{modelInfo.model_name}</strong></div><div><span>Version</span><strong>{modelInfo.model_version}</strong></div></div><div className="performance-model-facts"><div><span>Algorithm</span><strong>{modelInfo.model_name.toLowerCase().includes('xgb') ? 'XGBoost model' : 'Reported model'}</strong></div><div><span>Feature count</span><strong>{modelInfo.feature_count.toLocaleString()}</strong></div></div>
          {modelInfoError && <p className="performance-inline-error">{modelInfoError}</p>}
          <details className="performance-advanced-details"><summary>Advanced model details</summary><div className="performance-advanced-grid"><MetadataItem label="Decision threshold" value={String(modelInfo.threshold)} /><MetadataItem label="Feature count" value={String(modelInfo.feature_count)} /></div><div className="performance-feature-list"><span>Feature names</span><div className="feature-chip-list">{modelInfo.feature_names.map((feature) => <span className="feature-chip" key={feature}>{feature}</span>)}</div></div></details>
        </> : <p className={'performance-model-placeholder ' + (backendStatus === 'offline' ? 'error' : '')}>{backendStatus === 'checking' ? 'Loading model information…' : modelInfoError || 'Model information is unavailable.'}</p>}
      </article>
      {metricCards.map(({ label, value, tone }) => <article className={'panel performance-metric-card ' + tone} key={label}><span>{label}</span><strong>{evaluationLoading ? '…' : formatEvaluationValue(value)}</strong><small>{evaluationError || (evaluation && !evaluation.available) ? 'Not available' : value === null || value === undefined ? 'Not reported' : 'Backend-reported'}</small></article>)}
    </div>

    <div className="performance-charts-grid">
      <EvaluationChart title="ROC curve" xLabel="False positive rate" yLabel="True positive rate" xValue={point?.fpr ?? null} yValue={point?.tpr ?? null} curveX={evaluation?.available ? evaluation.curves?.roc.false_positive_rate : undefined} curveY={evaluation?.available ? evaluation.curves?.roc.true_positive_rate : undefined} metricLabel="ROC-AUC" metricValue={evaluation?.available ? evaluation.metrics?.roc_auc : null} baseline />
      <EvaluationChart title="Precision-recall" xLabel="Recall" yLabel="Precision" xValue={point?.recall ?? null} yValue={point?.precision ?? null} curveX={evaluation?.available ? evaluation.curves?.precision_recall.recall : undefined} curveY={evaluation?.available ? evaluation.curves?.precision_recall.precision : undefined} metricLabel="Average precision" metricValue={evaluation?.available ? evaluation.metrics?.average_precision : null} />
      <article className="panel performance-matrix-panel"><div className="performance-card-heading"><h3>Confusion matrix</h3><span className="chart-metric-tag">Actual vs predicted</span></div>{matrix ? <div className="performance-matrix-wrap"><table className="performance-matrix"><thead><tr><th>Actual</th><th>Predicted normal</th><th>Predicted fraud</th></tr></thead><tbody><tr><th>Normal</th><td className="correct">{matrix.tn.toLocaleString()}</td><td>{matrix.fp.toLocaleString()}</td></tr><tr><th>Fraud</th><td>{matrix.fn.toLocaleString()}</td><td className="correct">{matrix.tp.toLocaleString()}</td></tr></tbody></table></div> : <div className="performance-matrix-empty">{evaluationLoading ? 'Loading backend counts…' : 'Confusion counts are not available in the report.'}</div>}<p className="performance-chart-note">Counts come from the backend evaluation at its reported threshold.</p></article>
    </div>

    <div className="performance-details-grid">
      <article className="panel performance-report-panel"><div className="performance-card-heading"><div><h3>Evaluation report</h3><p>Per-class scores calculated from the backend confusion counts.</p></div>{evaluation?.available && <span className="demo-tag">BACKEND REPORT</span>}</div>
        {matrix ? <div className="performance-table-wrap"><table className="performance-report-table"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1 score</th><th>Support</th></tr></thead><tbody>{classRows.map((row) => <tr key={row.label}><th>{row.label}</th><td>{formatEvaluationValue(row.precision)}</td><td>{formatEvaluationValue(row.recall)}</td><td>{formatEvaluationValue(row.f1)}</td><td>{row.support.toLocaleString()}</td></tr>)}<tr><th>Macro average</th><td>{formatEvaluationValue(averageMetric('precision', false))}</td><td>{formatEvaluationValue(averageMetric('recall', false))}</td><td>{formatEvaluationValue(averageMetric('f1', false))}</td><td>{totalSupport.toLocaleString()}</td></tr><tr><th>Weighted average</th><td>{formatEvaluationValue(averageMetric('precision', true))}</td><td>{formatEvaluationValue(averageMetric('recall', true))}</td><td>{formatEvaluationValue(averageMetric('f1', true))}</td><td>{totalSupport.toLocaleString()}</td></tr></tbody></table></div> : <p className="performance-model-placeholder">{evaluationLoading ? 'Loading evaluation report…' : 'Per-class results are unavailable without backend confusion counts.'}</p>}
      </article>
      <article className="panel performance-features-panel"><div className="performance-card-heading"><div><h3>Model parameters &amp; features</h3><p>Details supplied by the model service.</p></div></div>{modelInfo ? <><div className="performance-parameter-list"><div><span>Model name</span><strong>{modelInfo.model_name}</strong></div><div><span>Version</span><strong>{modelInfo.model_version}</strong></div><div><span>Decision threshold</span><strong>{modelInfo.threshold}</strong></div><div><span>Feature count</span><strong>{modelInfo.feature_count}</strong></div></div><details className="performance-feature-details"><summary>View feature names ({modelInfo.feature_names.length})</summary><div className="feature-chip-list">{modelInfo.feature_names.map((feature) => <span className="feature-chip" key={feature}>{feature}</span>)}</div></details><p className="performance-chart-note">Feature importance and training parameters were not included in the model information response.</p></> : <p className="performance-model-placeholder">Model parameters and feature names are unavailable.</p>}</article>
    </div>

    <article className="panel performance-scope-panel"><div className="performance-scope-summary"><div><span className="performance-scope-icon"><Activity size={19} /></span><div><h3>Evaluation scope &amp; data source</h3><p>{evaluation?.available ? evaluation.evaluation_scope || 'Backend-reported evaluation' : 'Evaluation provenance from the backend report.'}</p></div></div><div className="performance-scope-stats"><div><span>Data source</span><strong>{evaluation?.data_source || 'Not reported'}</strong></div><div><span>Evaluation type</span><strong>{evaluation?.available ? 'Backend evaluation' : 'Unavailable'}</strong></div><div><span>Samples evaluated</span><strong>{evaluation?.sample_count?.toLocaleString() ?? 'Not reported'}</strong></div></div></div>
      <details className="performance-advanced-details"><summary>Evaluation provenance and technical details</summary><div className="performance-advanced-grid"><MetadataItem label="Dataset version" value={evaluation?.dataset_version ?? 'Not reported'} /><MetadataItem label="Model" value={[evaluation?.model_name, evaluation?.model_version].filter(Boolean).join(' · ') || 'Not reported'} /><MetadataItem label="Decision threshold" value={evaluation?.threshold === null || evaluation?.threshold === undefined ? 'Not reported' : String(evaluation.threshold)} /><MetadataItem label="Evaluation date (UTC)" value={evaluation?.evaluation_date_utc ?? 'Not reported'} /><MetadataItem label="Positive / negative samples" value={(evaluation?.positive_count?.toLocaleString() ?? '—') + ' / ' + (evaluation?.negative_count?.toLocaleString() ?? '—')} /><MetadataItem label="Test split" value={evaluation?.split?.test_fraction === null || evaluation?.split?.test_fraction === undefined ? 'Not reported' : (evaluation.split.test_fraction * 100).toFixed(2) + '%'} /></div>{(evaluation?.dataset_sha256 || evaluation?.model_sha256) && <div className="evaluation-hashes">{evaluation.dataset_sha256 && <span>Dataset SHA-256 <code>{evaluation.dataset_sha256}</code></span>}{evaluation.model_sha256 && <span>Model SHA-256 <code>{evaluation.model_sha256}</code></span>}</div>}</details>
    </article>
  </section>
}

function MetadataItem({ label, value }: { label: string; value: string }) {
  return <div className="metadata-item"><span>{label}</span><strong>{value}</strong></div>
}

function PaymentActivityPage() {
  const [activity, setActivity] = useState<PaymentActivityResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [refreshKey, setRefreshKey] = useState(0)

  useEffect(() => {
    let active = true
    const loadActivity = async () => {
      setLoading(true)
      setError('')
      try {
        const result = await getPaymentActivity({
          interval_seconds: 3600,
          limit: 500,
          offset: 0,
        })
        if (active) setActivity(result)
      } catch (requestError) {
        if (active) {
          setActivity(null)
          setError(requestError instanceof Error ? requestError.message : 'Unable to load payment activity.')
        }
      } finally {
        if (active) setLoading(false)
      }
    }
    void loadActivity()
    return () => {
      active = false
    }
  }, [refreshKey])

  const buckets = activity?.buckets ?? []
  const transactionCount = buckets.reduce((total, bucket) => total + bucket.transaction_count, 0)
  const classOneCount = buckets.reduce((total, bucket) => total + bucket.fraud_count, 0)
  const totalAmount = buckets.reduce((total, bucket) => total + bucket.total_amount, 0)

  return (
    <section className="workspace-page">
      <div className="page-section-heading">
        <div>
          <h2>Payment activity</h2>
          <p>Historical activity from the backend dataset, grouped into one-hour elapsed-time intervals.</p>
        </div>
        <button className="button-secondary model-retry" disabled={loading} onClick={() => setRefreshKey((current) => current + 1)} type="button">
          <RefreshCw size={14} /> {loading ? 'Loading…' : 'Refresh activity'}
        </button>
      </div>
      {activity && (
        <div className="payment-context">
          <span><strong>{activity.dataset_version}</strong> · {activity.data_source}</span>
          <span>Elapsed time from first transaction · {activity.interval_seconds.toLocaleString()} seconds per bucket</span>
          <span>Recorded CSV labels, not model predictions</span>
        </div>
      )}
      <div className="dataset-stat-grid payment-stat-grid">
        <DatasetStat label="Transactions in returned buckets" value={loading && !activity ? '—' : transactionCount.toLocaleString()} detail="Historical backend records" />
          <DatasetStat label="CSV rows labeled fraud" value={loading && !activity ? '—' : classOneCount.toLocaleString()} detail="Recorded labels, not model predictions" />
        <DatasetStat label="Total amount" value={loading && !activity ? '—' : totalAmount.toLocaleString(undefined, { maximumFractionDigits: 2 })} detail="Dataset units; currency unspecified" />
      </div>
      {error && (
        <div className="dataset-error" role="alert">
          <span>{error}</span>
          <button className="text-button" onClick={() => setRefreshKey((current) => current + 1)} type="button">Retry</button>
        </div>
      )}
      <article className="panel payment-chart-panel">
        <div className="panel-heading">
          <div>
            <h2>Historical transaction activity</h2>
            <p>Transaction volume and rows labeled fraud in the CSV, by elapsed hour</p>
          </div>
          {activity && <span className="demo-tag">{activity.buckets.length.toLocaleString()} BUCKETS</span>}
        </div>
        {loading && !activity ? (
          <p className="dataset-message">Loading historical activity…</p>
        ) : !error && buckets.length === 0 ? (
          <p className="dataset-message">The backend returned no activity buckets.</p>
        ) : buckets.length > 0 ? (
          <div className="payment-activity-chart">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={buckets} margin={{ top: 8, right: 12, left: -12, bottom: 4 }}>
                <CartesianGrid stroke="#202c3a" strokeDasharray="3 5" vertical={false} />
                <XAxis
                  dataKey="period_start_seconds"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#94a4b6', fontSize: 11 }}
                  tickFormatter={(seconds: number) => `${(seconds / 3600).toFixed(0)}h`}
                />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#94a4b6', fontSize: 11 }} />
                <Tooltip
                  contentStyle={tooltipStyle}
                  labelFormatter={(seconds) => `${Number(seconds).toLocaleString()} seconds from first transaction`}
                />
                <Area type="monotone" dataKey="transaction_count" name="Transactions" stroke="#52c8d8" strokeWidth={2} fill="#52c8d822" />
                <Area type="monotone" dataKey="fraud_count" name="CSV rows labeled fraud" stroke="#f0b660" strokeWidth={2} fill="#f0b66022" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ) : null}
      </article>
    </section>
  )
}

function FraudNetworkPage({ datasetId, onUseBundled }: { datasetId: string | null; onUseBundled: () => void }) {
  const [graph, setGraph] = useState<TransactionGraphResponse | null>(null)
  const [focusedGraph, setFocusedGraph] = useState<TransactionGraphResponse | null>(null)
  const [selectedAccountId, setSelectedAccountId] = useState('')
  const [neighborLimit, setNeighborLimit] = useState(24)
  const [direction, setDirection] = useState<'all' | 'incoming' | 'outgoing'>('all')
  const [transferType, setTransferType] = useState('all')
  const [transactionSearch, setTransactionSearch] = useState('')
  const [selectedTransfer, setSelectedTransfer] = useState<{ source: string; target: string; evidence: TransactionGraphResponse['edges'][number]['evidence'][number] } | null>(null)
  const [hoveredEdge, setHoveredEdge] = useState('')
  const [hoveredNode, setHoveredNode] = useState('')
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [panning, setPanning] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)
  const panStart = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null)
  const chooseAccount = (accountId: string) => {
    setSelectedTransfer(null)
    setFocusedGraph(null)
    setDirection('all')
    setTransferType('all')
    setTransactionSearch('')
    setHoveredEdge('')
    setZoom(1)
    setPan({ x: 0, y: 0 })
    setSelectedAccountId(accountId)
  }
  const focusTransferAccount = (accountId: string) => {
    setFocusedGraph(null)
    setZoom(1)
    setPan({ x: 0, y: 0 })
    setSelectedAccountId(accountId)
  }
  const inspectTransfer = (source: string, target: string, evidence: TransactionGraphResponse['edges'][number]['evidence'][number]) => {
    setSelectedTransfer({ source, target, evidence })
    setFocusedGraph(null)
    setDirection('all')
    setTransferType('all')
    setSelectedAccountId(source)
  }
  useEffect(() => {
    let active = true
    setSelectedAccountId('')
    setSelectedTransfer(null)
    setFocusedGraph(null)
    void getPaysimGraph({ limit: 30, max_rows: 100000, dataset_id: datasetId ?? undefined }).then((data) => {
      if (active) {
        setGraph(data)
        setError('')
      }
    }).catch((requestError: unknown) => {
      if (active) setError(requestError instanceof Error ? requestError.message : 'Unable to load account graph.')
    }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [refresh, datasetId])
  useEffect(() => {
    let active = true
    if (!selectedAccountId || !graph) return () => { active = false }
    void getPaysimGraph({ limit: 30, max_rows: 100000, dataset_id: datasetId ?? undefined,
      account_id: selectedAccountId, neighbor_limit: neighborLimit }).then((data) => {
      if (active) { setFocusedGraph(data); setError('') }
    }).catch((requestError: unknown) => {
      if (active) setError(requestError instanceof Error ? requestError.message : 'Unable to load account connections.')
    })
    return () => { active = false }
  }, [graph, selectedAccountId, datasetId, neighborLimit])

  const graphNodes = useMemo(() => [...new Map([...(graph?.nodes ?? []), ...(focusedGraph?.nodes ?? [])].map((node) => [node.id, node])).values()], [graph, focusedGraph])
  const graphEdges = useMemo(() => [...new Map([...(graph?.edges ?? []), ...(focusedGraph?.edges ?? [])].map((edge) => [`${edge.source}\u0000${edge.target}`, edge])).values()], [graph, focusedGraph])
  const selectedAccount = graphNodes.find((node) => node.id === selectedAccountId)
  const selectableAccounts = [...graphNodes].sort((left, right) => left.id.localeCompare(right.id))
  const allAccountEdges = graphEdges.filter((edge) => edge.source === selectedAccountId || edge.target === selectedAccountId)
  const incomingEdges = allAccountEdges.filter((edge) => edge.target === selectedAccountId)
  const outgoingEdges = allAccountEdges.filter((edge) => edge.source === selectedAccountId)
  const matchesDirection = (edge: TransactionGraphResponse['edges'][number]) => direction === 'all' || (direction === 'incoming' ? edge.target === selectedAccountId : edge.source === selectedAccountId)
  const matchesType = (edge: TransactionGraphResponse['edges'][number]) => transferType === 'all' || edge.evidence.some((item) => item.type === transferType)
  const displayedIncoming = incomingEdges.filter((edge) => matchesDirection(edge) && matchesType(edge))
  const displayedOutgoing = outgoingEdges.filter((edge) => matchesDirection(edge) && matchesType(edge))
  const focusId = selectedAccountId || selectedTransfer?.source || ''
  const viewHeight = 620
  const centerY = viewHeight / 2
  const coords = useMemo(() => {
    const nodes = [...graphNodes].sort((a, b) => a.id.localeCompare(b.id))
    const xy = new Map<string, { x: number; y: number }>()
    const direct = new Set(graphEdges.filter((edge) => edge.source === focusId || edge.target === focusId).flatMap((edge) => [edge.source, edge.target]).filter((id) => id !== focusId))
    const others = nodes.filter((node) => node.id !== focusId && !direct.has(node.id))
    const focused = nodes.filter((node) => node.id !== focusId && direct.has(node.id))
    if (focusId) xy.set(focusId, { x: 500, y: centerY })
    focused.forEach((node, i) => {
      const angle = (Math.PI * 2 * i) / Math.max(focused.length, 1) - Math.PI / 2
      const radius = 205
      xy.set(node.id, { x: 500 + Math.cos(angle) * radius, y: centerY + Math.sin(angle) * radius })
    })
    const rest = focusId ? others : nodes
    rest.forEach((node, i) => {
      const angle = i * 2.399963
      const radius = Math.sqrt((i + 1) / Math.max(rest.length, 1)) * 375
      xy.set(node.id, { x: 500 + Math.cos(angle) * radius, y: centerY + Math.sin(angle) * radius * .68 })
    })
    const springs = graphEdges.flatMap((edge) => {
      const a = xy.get(edge.source), b = xy.get(edge.target)
      return a && b ? [{ a: edge.source, b: edge.target }] : []
    })
    for (let step = 0; step < 44; step++) {
      const forces = new Map(nodes.map((node) => [node.id, { x: 0, y: 0 }]))
      const movable = nodes.filter((node) => node.id !== focusId)
      for (let i = 0; i < movable.length; i++) for (let j = i + 1; j < movable.length; j++) {
        const a = xy.get(movable[i].id)!, b = xy.get(movable[j].id)!
        let dx = a.x - b.x, dy = a.y - b.y
        const d2 = Math.max(dx * dx + dy * dy, 100)
        const force = 850 / d2
        dx *= force; dy *= force
        forces.get(movable[i].id)!.x += dx; forces.get(movable[i].id)!.y += dy
        forces.get(movable[j].id)!.x -= dx; forces.get(movable[j].id)!.y -= dy
      }
      for (const spring of springs) {
        const a = xy.get(spring.a)!, b = xy.get(spring.b)!
        const dx = b.x - a.x, dy = b.y - a.y, distance = Math.max(Math.hypot(dx, dy), 1)
        const force = (distance - (spring.a === focusId || spring.b === focusId ? 205 : 120)) * .002
        if (spring.a !== focusId) { forces.get(spring.a)!.x += dx * force; forces.get(spring.a)!.y += dy * force }
        if (spring.b !== focusId) { forces.get(spring.b)!.x -= dx * force; forces.get(spring.b)!.y -= dy * force }
      }
      const rate = .22 * (1 - step / 55)
      for (const node of movable) {
        const pos = xy.get(node.id)!, force = forces.get(node.id)!
        pos.x = Math.max(35, Math.min(965, pos.x + Math.max(-12, Math.min(12, force.x)) * rate))
        pos.y = Math.max(35, Math.min(viewHeight - 35, pos.y + Math.max(-12, Math.min(12, force.y)) * rate))
      }
    }
    return xy
  }, [graphNodes, graphEdges, focusId, centerY])
  const positions = graphNodes.filter((node) => node.id !== focusId).map((node) => ({ node, ...(coords.get(node.id) ?? { x: 0, y: 0 }) }))
  const focus = coords.get(focusId) ?? { x: 500, y: centerY }
  const centerSelectedTransfer = () => {
    if (!selectedTransfer) return
    const source = coords.get(selectedTransfer.source)
    const target = coords.get(selectedTransfer.target)
    if (!source || !target) return
    setZoom(1.2)
    setPan({ x: 500 - (source.x + target.x) / 2, y: centerY - (source.y + target.y) / 2 })
  }
  const evidenceEdges = selectedAccountId ? allAccountEdges : graphEdges
  const transactionRows = evidenceEdges.flatMap((edge) => edge.evidence.map((evidence) => ({ source: edge.source, target: edge.target, evidence })))
    .filter(({ source, target, evidence }) => {
      const isFocusTransfer = source === selectedAccountId || target === selectedAccountId
      const isSelected = selectedTransfer?.evidence.source_row === evidence.source_row
      const directionMatches = !isFocusTransfer || direction === 'all' || (direction === 'incoming' ? target === selectedAccountId : source === selectedAccountId)
      const typeMatches = transferType === 'all' || evidence.type === transferType
      const query = transactionSearch.trim().toLowerCase()
      const searchMatches = !query || `${source} ${target} ${evidence.source_row} ${evidence.type}`.toLowerCase().includes(query)
      return isSelected || (directionMatches && typeMatches && searchMatches)
    })
    .sort((left, right) => left.evidence.source_row - right.evidence.source_row)
  const transactions = selectedTransfer && !transactionRows.some((row) => row.evidence.source_row === selectedTransfer.evidence.source_row)
    ? [...transactionRows, selectedTransfer].sort((left, right) => left.evidence.source_row - right.evidence.source_row)
    : transactionRows
  const incomingAmount = incomingEdges.reduce((total, edge) => total + edge.total_amount, 0)
  const outgoingAmount = outgoingEdges.reduce((total, edge) => total + edge.total_amount, 0)
  const totalIncomingTransactions = incomingEdges.reduce((total, edge) => total + edge.transaction_count, 0)
  const totalOutgoingTransactions = outgoingEdges.reduce((total, edge) => total + edge.transaction_count, 0)
  const resetView = () => {
    setSelectedTransfer(null)
    setDirection('all')
    setTransferType('all')
    setTransactionSearch('')
    setNeighborLimit(24)
    setZoom(1)
    setPan({ x: 0, y: 0 })
    setPanning(false)
    setSelectedAccountId('')
    setFocusedGraph(null)
    setLoading(true)
    setRefresh((value) => value + 1)
  }
  const startPan = (event: ReactPointerEvent<SVGRectElement>) => {
    panStart.current = { x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y }
    setPanning(true)
    event.currentTarget.setPointerCapture(event.pointerId)
  }
  const movePan = (event: ReactPointerEvent<SVGSVGElement>) => {
    const start = panStart.current
    if (!start) return
    const bounds = event.currentTarget.getBoundingClientRect()
    const scale = 1000 / Math.max(bounds.width, 1)
    setPan({ x: start.panX + (event.clientX - start.x) * scale, y: start.panY + (event.clientY - start.y) * scale })
  }
  return (
    <section className="workspace-page fraud-network-page">
      <div className="fraud-network-heading"><div><p>{graph ? `${graph.data_source} · ${graph.sampled ? 'seeded sample, up to 100,000 eligible rows' : 'all eligible rows'} · ${graph.dataset_version}` : 'Loading sampled PaySim account relationships…'}</p></div><div className="network-actions">{datasetId && <button className="button-secondary" onClick={onUseBundled} type="button">Use bundled PaySim</button>}<button className="button-secondary" disabled={loading} onClick={resetView} type="button">Reset view</button></div></div>
      {error && <div className="dataset-error" role="alert">{error}<button className="text-button" onClick={() => { setError(''); setLoading(true); setRefresh((value) => value + 1) }} type="button">Retry</button></div>}
      {loading && !graph ? <article className="panel network-loading"><p className="dataset-message">Loading the sampled PaySim network…</p></article> : graph ? <>
        <div className="fraud-network-workspace">
          <aside className="panel selected-account-panel" aria-label="Selected account details">{selectedAccount ? <>
            <div className="fraud-panel-heading"><div><span className="fraud-panel-kicker">SELECTED ACCOUNT</span><h3 className="account-id-heading">{selectedAccount.id}</h3></div><span className={`account-pattern-badge ${selectedAccount.reasons.length ? 'flagged' : ''}`}>{selectedAccount.reasons.length ? 'Pattern alert' : 'No listed pattern'}</span></div>
            <div className="pattern-score"><strong>{(selectedAccount.risk_score * 100).toFixed(1)}<small>%</small></strong><span>Structural pattern score</span><p>Summarizes activity, connections, flow amount, and timing in the sampled rows. It is not a probability or proof of fraud.</p></div>
            <div className="account-flow-summary"><div><span>Outgoing</span><strong>{totalOutgoingTransactions.toLocaleString()}</strong><small>{outgoingAmount.toLocaleString(undefined, { maximumFractionDigits: 2 })} total value</small></div><div><span>Incoming</span><strong>{totalIncomingTransactions.toLocaleString()}</strong><small>{incomingAmount.toLocaleString(undefined, { maximumFractionDigits: 2 })} total value</small></div></div>
            <div className="account-filter-heading"><h4>Direct connections</h4><span>{incomingEdges.length + outgoingEdges.length}</span></div>
            <div className="account-direction-filters" role="group" aria-label="Filter direct connections"><button className={direction === 'all' ? 'active' : ''} onClick={() => setDirection('all')} type="button">All</button><button className={direction === 'outgoing' ? 'outgoing active' : 'outgoing'} onClick={() => setDirection('outgoing')} type="button">Outgoing {outgoingEdges.length}</button><button className={direction === 'incoming' ? 'incoming active' : 'incoming'} onClick={() => setDirection('incoming')} type="button">Incoming {incomingEdges.length}</button></div>
            <div className="direct-connection-list">{(direction === 'incoming' ? displayedIncoming : direction === 'outgoing' ? displayedOutgoing : [...displayedIncoming, ...displayedOutgoing]).map((edge) => { const connectedId = edge.source === selectedAccount.id ? edge.target : edge.source; const isOutgoing = edge.source === selectedAccount.id; return <button className="direct-connection" key={`${edge.source}-${edge.target}`} onClick={() => chooseAccount(connectedId)} type="button"><i className={isOutgoing ? 'outgoing' : 'incoming'}>{isOutgoing ? '→' : '←'}</i><span><strong>{connectedId}</strong><small>{edge.transaction_count} transfers</small></span><b>{edge.total_amount.toLocaleString(undefined, { maximumFractionDigits: 0 })}</b></button> })}</div>
            {selectedAccount.reasons.length > 0 && <div className="account-pattern-reasons"><h4>Why this pattern was flagged</h4><ul>{selectedAccount.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul></div>}
          </> : <div className="network-instruction"><h3>Select an account or transaction</h3><p>Select an account in the graph, choose one from the account list, or select a transaction below to inspect its connections.</p></div>}</aside>

          <article className="panel network-graph-panel" aria-label="Account relationship graph">
            <div className="graph-panel-heading"><div><h3>Account neighborhood</h3><p>Choose an account or select a connected node to investigate.</p></div><div className="graph-heading-actions"><label className="graph-account-picker"><span>Choose account</span><select aria-label="Choose an account to investigate" value={selectedAccountId} onChange={(event) => { if (event.target.value) chooseAccount(event.target.value) }}><option value="" disabled>Select an account</option>{selectableAccounts.map((node) => <option key={node.id} value={node.id}>{node.id}{node.reasons.length ? ' · Pattern flagged' : ''}</option>)}</select></label><label className="graph-type-filter"><span>Transfer type</span><select value={transferType} onChange={(event) => setTransferType(event.target.value)}><option value="all">All types</option>{['TRANSFER', 'CASH_OUT'].map((type) => <option key={type}>{type}</option>)}</select></label><div className="graph-view-controls"><button aria-label="Zoom out" onClick={() => setZoom((value) => Math.max(0.65, value - 0.15))} type="button">−</button><span>{Math.round(zoom * 100)}%</span><button aria-label="Zoom in" onClick={() => setZoom((value) => Math.min(2.5, value + 0.15))} type="button">+</button><button onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }) }} type="button">Fit</button></div></div></div>
            <div className="graph-legend"><span><i className="legend-node selected" />Selected account</span><span><i className="legend-node flagged" />Pattern flagged</span><span><i className="legend-line outgoing" />Outgoing</span><span><i className="legend-line incoming" />Incoming</span><span><i className="legend-line contextual" />Other transfers</span></div>
            <div className={`network-canvas ${panning ? 'panning' : ''}`} aria-label="Drag empty space to pan. Use plus and minus to zoom."><svg className="network-graph" viewBox={`0 0 1000 ${viewHeight}`} role="img" aria-label={selectedAccount ? `PaySim transfer network focused on ${selectedAccount.id}` : 'Full sampled PaySim account transfer network'} onPointerMove={movePan} onPointerUp={() => { panStart.current = null; setPanning(false) }} onPointerCancel={() => { panStart.current = null; setPanning(false) }} onWheel={(event) => { event.preventDefault(); setZoom((value) => Math.min(2.5, Math.max(0.65, value * (event.deltaY < 0 ? 1.1 : 0.9)))) }}>
              <defs><marker id="network-arrow-outgoing" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#ee777c" /></marker><marker id="network-arrow-incoming" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#64b6f6" /></marker><marker id="network-arrow-selected" markerWidth="9" markerHeight="9" refX="8" refY="3.5" orient="auto"><path d="M0,0 L0,7 L8,3.5 z" fill="#f4c66e" /></marker><marker id="network-arrow-muted" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#526170" /></marker></defs>
              <g transform={`translate(${pan.x} ${pan.y}) scale(${zoom})`}>
                <rect className="network-pan-surface" x="0" y="0" width="1000" height={viewHeight} onPointerDown={startPan} />
                <g className="network-links">{graphEdges.map((edge) => { const start = coords.get(edge.source); const end = coords.get(edge.target); const highlighted = isSelectedGraphEdge(edge, selectedTransfer); const isFocusEdge = Boolean(selectedAccountId && (edge.source === selectedAccountId || edge.target === selectedAccountId)); const filteredFocusEdge = isFocusEdge && (!matchesDirection(edge) || !matchesType(edge)); const relevant = selectedTransfer ? highlighted : !selectedAccountId || (isFocusEdge && !filteredFocusEdge); const key = `${edge.source}\u0000${edge.target}`; const isOutgoing = edge.source === selectedAccountId; const arrow = highlighted ? 'url(#network-arrow-selected)' : !relevant ? 'url(#network-arrow-muted)' : isOutgoing ? 'url(#network-arrow-outgoing)' : 'url(#network-arrow-incoming)'; const midpointX = start && end ? (start.x + end.x) / 2 : 0; const midpointY = start && end ? (start.y + end.y) / 2 : 0; const evidence = selectedTransfer && highlighted ? selectedTransfer.evidence : null; return start && end ? <g className={`graph-edge-group ${highlighted ? 'highlighted' : !relevant ? 'contextual' : isOutgoing ? 'outgoing' : 'incoming'}`} key={`${edge.source}-${edge.target}`} role="button" tabIndex={0} aria-label={`Select a transfer from ${edge.source} to ${edge.target}; ${edge.transaction_count} rows, CSV row ${edge.evidence[0]?.source_row ?? 'not supplied'}`} onMouseEnter={() => setHoveredEdge(key)} onMouseLeave={() => setHoveredEdge('')} onClick={() => edge.evidence[0] && inspectTransfer(edge.source, edge.target, edge.evidence[0])} onKeyDown={(event) => { if ((event.key === 'Enter' || event.key === ' ') && edge.evidence[0]) { event.preventDefault(); inspectTransfer(edge.source, edge.target, edge.evidence[0]) } }}><line x1={start.x} y1={start.y} x2={end.x} y2={end.y} markerEnd={arrow} className={highlighted ? 'network-link-highlighted' : !relevant ? 'network-edge-context' : isOutgoing ? 'network-edge-outgoing' : 'network-edge-incoming'} /><title>{`${edge.source} to ${edge.target}: ${edge.transaction_count} transfers, ${edge.total_amount.toLocaleString(undefined, { maximumFractionDigits: 2 })} total`}</title>{(evidence || hoveredEdge === key) && <text className="graph-edge-amount" x={midpointX} y={midpointY - 8}>{(evidence?.amount ?? edge.total_amount).toLocaleString(undefined, { maximumFractionDigits: 0 })}</text>}</g> : null })}</g>
                {selectedAccount && <g className="account-node selected" role="button" tabIndex={0} aria-label={`Selected account ${selectedAccount.id}`} onClick={() => chooseAccount(selectedAccount.id)}><circle cx={focus.x} cy={focus.y} r={31} /><text x={focus.x} y={focus.y + 52} textAnchor="middle">{selectedAccount.id}</text></g>}
                {positions.map(({ node, x, y }) => { const flagged = node.reasons.length > 0; const connected = isSelectedTransferAccount(node.id, selectedTransfer); const direct = Boolean(selectedAccountId && graphEdges.some((edge) => (edge.source === selectedAccountId && edge.target === node.id) || (edge.target === selectedAccountId && edge.source === node.id))); const overview = !selectedAccountId && !selectedTransfer; const prominent = overview || direct || connected; const focused = node.id === focusId; const showLabel = (!overview && (direct || connected || focused)) || hoveredNode === node.id; return <g className={`account-node ${overview ? 'overview' : prominent ? 'prominent' : 'dimmed'} ${flagged ? 'flagged' : 'normal'} ${connected ? 'transfer-connected' : ''}`} key={node.id} role="button" tabIndex={0} aria-label={`Inspect account ${node.id}${flagged ? ', pattern flagged' : ''}`} onMouseEnter={() => setHoveredNode(node.id)} onMouseLeave={() => setHoveredNode('')} onClick={() => chooseAccount(node.id)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); chooseAccount(node.id) } }}><circle cx={focused ? focus.x : x} cy={focused ? focus.y : y} r={connected ? 21 : focused ? 31 : 16} /><text className={showLabel ? '' : 'node-label-hidden'} x={focused ? focus.x : x} y={(focused ? focus.y : y) + (focused ? 52 : 34)} textAnchor="middle">{showLabel ? node.id : ''}</text><title>{node.reasons.join('; ') || 'No listed structural alert pattern'}</title></g> })}
              </g>
            </svg></div>
            {focusedGraph?.neighborhood_truncated && <div className="graph-expand-row"><span>Some direct connections are outside the current neighborhood sample.</span><button className="text-button" onClick={() => setNeighborLimit((value) => Math.min(value + 24, 500))} type="button">Show more connections</button></div>}
            <div className="network-disclaimer"><ShieldAlert size={14} />PaySim is simulated data. Pattern alerts describe observed structure, not confirmed fraud; simulator labels do not affect the score.</div>
          </article>

          <aside className="panel transfer-details-panel" aria-label="Selected transaction details"><div className="fraud-panel-heading"><div><span className="fraud-panel-kicker">TRANSACTION DETAILS</span><h3>{selectedTransfer ? 'Selected transfer' : 'No transfer selected'}</h3></div>{selectedTransfer && <button className="icon-button" aria-label="Clear selected transfer" onClick={() => setSelectedTransfer(null)} type="button">×</button>}</div>
            {selectedTransfer ? <><strong className="transfer-amount">{selectedTransfer.evidence.amount.toLocaleString(undefined, { maximumFractionDigits: 2 })}</strong><span className="transfer-amount-caption">Amount · PaySim units</span><div className="transfer-detail-list"><DetailRow label="Transaction type" value={selectedTransfer.evidence.type} /><DetailRow label="PaySim step" value={String(selectedTransfer.evidence.step)} /><div className="transfer-account-row"><span>Source account</span><button onClick={() => focusTransferAccount(selectedTransfer.source)} type="button">{selectedTransfer.source}</button></div><div className="transfer-account-row"><span>Destination account</span><button onClick={() => focusTransferAccount(selectedTransfer.target)} type="button">{selectedTransfer.target}</button></div><DetailRow label="Original CSV row" value={String(selectedTransfer.evidence.source_row)} /></div><div className="transfer-detail-actions"><button className="button-secondary highlight-transfer-button" onClick={centerSelectedTransfer} type="button">Center highlighted transfer</button><button className="button-secondary" onClick={() => focusTransferAccount(selectedTransfer.target)} type="button">Inspect destination account</button></div></> : selectedAccount ? <p>Select a transaction row or graph edge to see its actual CSV row, type, step, and amount.</p> : <p>Select an account or transaction to inspect its connections and evidence.</p>}
          </aside>

            <section className="panel network-transactions-panel" aria-label="Transactions in the graph"><div className="graph-panel-heading"><div><h3>Transactions</h3><p>{transactions.length.toLocaleString()} sampled evidence rows {selectedAccount ? `for ${selectedAccount.id}` : 'in the full graph'}.</p></div><label className="transaction-search"><Search size={14} /><input aria-label="Search graph transactions" placeholder="Search account or CSV row…" value={transactionSearch} onChange={(event) => setTransactionSearch(event.target.value)} /></label></div><div className="table-scroll network-transactions-scroll"><table className="workspace-table network-transactions-table"><thead><tr><th>CSV ROW</th><th>SOURCE ACCOUNT</th><th>DESTINATION ACCOUNT</th><th>AMOUNT</th><th>TYPE</th><th>STEP</th></tr></thead><tbody>{transactions.map(({ source, target, evidence }) => <tr key={`${source}-${target}-${evidence.source_row}`} aria-selected={selectedTransfer?.evidence.source_row === evidence.source_row} className={selectedTransfer?.evidence.source_row === evidence.source_row ? 'selected-row' : ''} onClick={() => inspectTransfer(source, target, evidence)}><td><button className="csv-row-link" onClick={(event) => { event.stopPropagation(); inspectTransfer(source, target, evidence) }} type="button">{evidence.source_row}</button></td><td>{source}</td><td>{target}</td><td>{evidence.amount.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td><td>{evidence.type}</td><td>{evidence.step}</td></tr>)}</tbody></table>{transactions.length === 0 && <p className="dataset-message">No sampled transfer evidence matches these filters.</p>}</div></section>
        </div>
      </> : !error && <article className="panel network-empty-state"><h3>No account neighborhood returned</h3><p>The selected account has no direct transfer evidence in the configured sample.</p><button className="button-secondary" onClick={resetView} type="button">Reset view</button></article>}
    </section>
  )
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return <div className="detail-row"><span>{label}</span><strong>{value}</strong></div>
}

export default App
