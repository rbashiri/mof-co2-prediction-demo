import { useEffect, useMemo, useState } from 'react'
import TeamContact from './TeamContact'

interface PressurePoint {
  pressure: number
  uptake: number
}

interface ShapData {
  baseline: number
  feature_names: string[]
  feature_values: number[]
  contributions: number[]
}

interface PredictResponse {
  pressures: PressurePoint[]
  features: Record<string, number>
  shap: Record<string, ShapData>
  summary: Record<string, string>
  gemini_summary: string | null
  gemini_error: string | null
  explanation_token: string
  predictions_csv: string
  geometry: Record<string, number>
  geometry_source: 'zeopp' | 'manual'
  geometry_warning: string | null
  geometry_settings: {
    surface_probe_radius_angstrom: number
    volume_probe_radius_angstrom: number
  } | null
}

const GEOMETRY_FIELDS = [
  { name: 'lcd', label: 'Largest cavity diameter (angstrom)' },
  { name: 'pld', label: 'Pore limiting diameter (angstrom)' },
  { name: 'void_fraction', label: 'Void fraction (0-1)' },
  { name: 'surface_area_m2g', label: 'Surface area (m2/g)' },
] as const

const PRESSURES = [0.01, 0.05, 0.1, 0.5, 2.5]

function PredictionProgress() {
  const [elapsedSeconds, setElapsedSeconds] = useState(0)

  useEffect(() => {
    const startedAt = performance.now()
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((performance.now() - startedAt) / 1000))
    }, 1000)
    return () => window.clearInterval(timer)
  }, [])

  return (
    <div className="card prediction-progress">
      <svg className="molecule-loader" viewBox="0 0 80 80" aria-hidden="true">
        <g className="molecule-orbit">
          <path d="M40 12 L64 26 L64 54 L40 68 L16 54 L16 26 Z M16 26 L64 54 M64 26 L16 54"
            fill="none" stroke="currentColor" strokeWidth="2" />
          <circle cx="40" cy="12" r="5" />
          <circle cx="64" cy="26" r="5" />
          <circle cx="64" cy="54" r="5" />
          <circle cx="40" cy="68" r="5" />
          <circle cx="16" cy="54" r="5" />
          <circle cx="16" cy="26" r="5" />
        </g>
        <circle cx="40" cy="40" r="7" className="molecule-center" />
      </svg>
      <div>
        <div role="status">
          <strong>Examining your MOF...</strong>
          <p>
            Processing your structure and predictions.
          </p>
        </div>
        <p className="elapsed-time" role="timer" aria-label="Elapsed processing time">
          Elapsed: {elapsedSeconds} {elapsedSeconds === 1 ? 'second' : 'seconds'}
        </p>
      </div>
    </div>
  )
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [showFeatures, setShowFeatures] = useState(false)
  const [selectedPressure, setSelectedPressure] = useState<string>('0.1')
  const [summaryLoading, setSummaryLoading] = useState(false)

  const selectFile = (nextFile: File) => {
    setFile(nextFile)
    setResult(null)
    setError(null)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    if (!loading && !summaryLoading && e.dataTransfer.files?.[0]) selectFile(e.dataTransfer.files[0])
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) selectFile(e.target.files[0])
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    const form = new FormData()
    form.append('cif', file)

    try {
      const res = await fetch('/api/predict', { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail?.error || 'Prediction failed')
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const explainResults = async () => {
    if (!result || summaryLoading) return
    const token = result.explanation_token
    setSummaryLoading(true)
    setResult((current) => current && { ...current, gemini_error: null })
    try {
      const response = await fetch(`/api/explanations/${encodeURIComponent(token)}`, { method: 'POST' })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail?.error || 'Explanation failed')
      setResult((current) => current?.explanation_token === token
        ? { ...current, gemini_summary: data.gemini_summary } : current)
    } catch (err) {
      setResult((current) => current?.explanation_token === token
        ? { ...current, gemini_error: err instanceof Error ? err.message : 'Explanation failed' } : current)
    } finally {
      setSummaryLoading(false)
    }
  }

  const downloadCsv = () => {
    if (!result) return
    const blob = new Blob([atob(result.predictions_csv)], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'predictions.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="container">
      <header>
        <h1>MOF CO₂ Uptake Predictor</h1>
        <p className="header-subtitle"><strong>Upload a CIF to extract all descriptors automatically and predict uptake</strong></p>
        <p className="header-tagline"><strong>From crystal structure to carbon-capture insights</strong></p>
      </header>

      <section className="card prediction-guide" aria-labelledby="prediction-guide-title">
        <div className="prediction-guide-heading">
          <h2 id="prediction-guide-title">About these predictions</h2>
          <span className="model-estimates-badge">Model estimates · MVP</span>
        </div>
        <p className="prediction-guide-description">
          Predict CO₂ uptake using models trained on simulated hMOF data.
          Results are estimates, not experimental measurements.
        </p>
        <h3>Scientific tools behind the predictions</h3>
        <div className="science-tools">
          <div className="science-tool tool-geometry">
            <svg viewBox="0 0 48 48" aria-hidden="true">
              <path d="m24 5 16 9v20l-16 9-16-9V14Zm0 0v19m16-10-16 10L8 14m16 10v19" />
            </svg>
            <strong>Zeo++</strong><span>Pore geometry</span>
          </div>
          <div className="science-tool tool-linkers">
            <svg viewBox="0 0 48 48" aria-hidden="true">
              <path d="m24 13-12 20m12-20 12 20M12 33h24" />
              <circle cx="24" cy="9" r="5" /><circle cx="10" cy="36" r="5" /><circle cx="38" cy="36" r="5" />
            </svg>
            <strong>MOFid</strong><span>Linker extraction</span>
          </div>
          <div className="science-tool tool-chemistry">
            <svg viewBox="0 0 48 48" aria-hidden="true">
              <ellipse cx="24" cy="24" rx="20" ry="8" transform="rotate(45 24 24)" />
              <ellipse cx="24" cy="24" rx="20" ry="8" transform="rotate(-45 24 24)" />
              <circle cx="24" cy="24" r="3" />
            </svg>
            <strong>RDKit</strong><span>Linker chemistry</span>
          </div>
          <div className="science-tool tool-materials">
            <svg viewBox="0 0 48 48" aria-hidden="true">
              <path d="m24 5 10 6v12l-10 6-10-6V11Zm-10 18-10 6v12l10 5 10-5V29m10-18 10 6v12l-10 6-10-6" />
            </svg>
            <strong>pymatgen</strong><span>Crystal descriptors</span>
          </div>
        </div>
      </section>

      <form onSubmit={handleSubmit}>
        <div className="card upload-card">
          <label
            className={`dropzone ${dragOver ? 'dragover' : ''}`}
            onDragOver={(e) => {
              e.preventDefault()
              setDragOver(true)
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
          >
            <input type="file" accept=".cif" onChange={handleFileChange} disabled={loading || summaryLoading} hidden />
            <div>Drag & drop a CIF file here, or click to browse</div>
            {file && <div className="file-name">{file.name}</div>}
          </label>

          <div className="actions">
            <button type="submit" className="btn" disabled={!file || loading || summaryLoading}>
              {loading ? 'Extracting & predicting…' : 'Extract descriptors and predict'}
            </button>
          </div>
        </div>
      </form>

      {loading && <PredictionProgress />}

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <div className="card">
            <h3>Geometry extracted by Zeo++</h3>
            <table className="feature-table">
              <thead><tr><th>Descriptor</th><th>Value</th></tr></thead>
              <tbody>
                {GEOMETRY_FIELDS.map(({ name, label }) => (
                  <tr key={name}><th scope="row">{label}</th>
                    <td>{Number(result.geometry[name].toPrecision(6)).toString()}</td></tr>
                ))}
              </tbody>
            </table>
            {result.geometry_settings && (
              <p className="about-body">
                Settings chosen by comparing Zeo++ geometry with reference values for
                10 sample MOFs: {result.geometry_settings.surface_probe_radius_angstrom} angstrom
                probe for surface area and {result.geometry_settings.volume_probe_radius_angstrom} angstrom
                probe for void fraction.
              </p>
            )}
          </div>
          <div className="card">
            <h3>Predicted CO₂ uptake</h3>
            {result.pressures.some((point) => point.uptake < 0) && (
              <p role="alert" className="about-body">
                The model returned negative uptake, which is not physically meaningful.
                Values are shown unchanged; do not treat these predictions as validated.
              </p>
            )}
            <div className="metric-grid">
              {result.pressures.map((p) => (
                <div className="metric" key={p.pressure}>
                  <div className="metric-value">{p.uptake.toFixed(3)}</div>
                  <div className="metric-label">{p.pressure} bar</div>
                </div>
              ))}
            </div>
            <UptakeTable data={result.pressures} />
            <PressureCurve data={result.pressures} />
            <div className="actions">
              <button className="btn btn-secondary" onClick={downloadCsv}>
                Download predictions CSV
              </button>
            </div>
          </div>

          <div className="card">
            <h3>SHAP explanation</h3>
            <div className="shap-controls">
              <label htmlFor="pressure">Pressure:</label>
              <select
                id="pressure"
                value={selectedPressure}
                onChange={(e) => setSelectedPressure(e.target.value)}
              >
                {PRESSURES.map((p) => (
                  <option key={p} value={String(p)}>
                    {p} bar
                  </option>
                ))}
              </select>
            </div>
            <ShapChart data={result.shap[selectedPressure]} />
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              SHAP describes model contributions, not proven chemical mechanisms.
            </p>
          </div>

          <div className="card">
            <div className="explanation-action">
              <button className="btn" aria-expanded={showFeatures} aria-controls="model-input-features"
                onClick={() => setShowFeatures((s) => !s)}>
                {showFeatures ? 'Hide' : 'Inspect'} the 48 model input features
              </button>
            </div>
            <div id="model-input-features" hidden={!showFeatures}>
              {showFeatures && <FeatureTable features={result.features} />}
            </div>
          </div>

          <div className="card">
            {!result.gemini_summary && (
              <div className="explanation-action">
                <button className="btn" onClick={explainResults} disabled={summaryLoading}>
                  {summaryLoading ? 'Preparing your explanation...' : 'Click here to learn more about the results'}
                </button>
              </div>
            )}
            {summaryLoading && (
              <p className="insights-status" role="status">
                <span aria-hidden="true">⌛</span>
                Preparing scientific insights from adsorption trends and SHAP results…
              </p>
            )}
            {result.gemini_summary && (
              <ScientificInterpretation text={result.gemini_summary} />
            )}
            {result.gemini_error && (
              <div className="error" role="alert">
                Gemini summary unavailable: {result.gemini_error}
                {' '}Predictions and SHAP are unchanged.
              </div>
            )}
          </div>
        </>
      )}
      <TeamContact />
    </div>
  )
}

function ScientificInterpretation({ text }: { text: string }) {
  const headings: Record<string, string> = {
    'ADSORPTION BEHAVIOR': 'Predicted adsorption behavior',
    'STRUCTURE-PROPERTY INSIGHTS': 'Structure–property relationships',
    'RESEARCH IMPLICATIONS': 'Suggested next investigation',
  }
  const parts = text.split(/(ADSORPTION BEHAVIOR|STRUCTURE-PROPERTY INSIGHTS|RESEARCH IMPLICATIONS)/)

  return (
    <div className="scientific-interpretation">
      <p className="insights-status insights-ready" role="status">
        <span aria-hidden="true">✓</span> Scientific insights ready
      </p>
      <h3>Scientific interpretation</h3>
      {parts[0].trim() && <p>{parts[0].trim()}</p>}
      {parts.map((part, index) => headings[part] && (
        <section className="interpretation-section" key={`${part}-${index}`}>
          <h4>{headings[part]}</h4>
          <p>{parts[index + 1]?.trim()}</p>
        </section>
      ))}
      <section className="interpretation-section">
        <h4>Scientific limitations</h4>
        <p>
          AI interpretation of simulated hMOF predictions, not experimental results.
          SHAP explains model contributions, not chemical causes; contribution sizes
          are not directly comparable across pressure models. Experimental validation is needed.
        </p>
      </section>
    </div>
  )
}

function UptakeTable({ data }: { data: PressurePoint[] }) {
  const lowest = Math.min(...data.map((point) => point.uptake))
  const highest = Math.max(...data.map((point) => point.uptake))
  return (
    <table className="feature-table uptake-table">
      <caption>Five-pressure comparison for this structure</caption>
      <thead>
        <tr><th scope="col">Pressure (bar)</th><th scope="col">Predicted CO₂ uptake (mol/kg)</th><th scope="col">Comparison</th></tr>
      </thead>
      <tbody>
        {[...data].sort((a, b) => a.pressure - b.pressure).map((point) => (
          <tr key={point.pressure} className={point.uptake === highest ? 'uptake-highest' : point.uptake === lowest ? 'uptake-lowest' : ''}>
            <th scope="row">{point.pressure}</th>
            <td>{Number(point.uptake.toPrecision(6)).toString()}</td>
            <td>
              {point.uptake === highest && point.uptake === lowest ? 'Highest / lowest (equal)' :
                point.uptake === highest ? 'Highest' : point.uptake === lowest ? 'Lowest' : '—'}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function PressureCurve({ data }: { data: PressurePoint[] }) {
  const width = 700
  const height = 260
  const padding = { top: 20, right: 35, bottom: 55, left: 75 }
  const chartWidth = width - padding.left - padding.right
  const chartHeight = height - padding.top - padding.bottom

  const xMin = Math.log10(Math.min(...data.map((d) => d.pressure)))
  const xMax = Math.log10(Math.max(...data.map((d) => d.pressure)))
  const yMax = Math.max(0, ...data.map((d) => d.uptake)) * 1.1 || 1
  const yMin = Math.min(0, ...data.map((d) => d.uptake)) * 1.1

  const xScale = (p: number) =>
    padding.left + ((Math.log10(p) - xMin) / (xMax - xMin)) * chartWidth
  const yScale = (v: number) =>
    padding.top + chartHeight - ((v - yMin) / (yMax - yMin)) * chartHeight

  const path = data
    .map((d, i) => `${i === 0 ? 'M' : 'L'} ${xScale(d.pressure)} ${yScale(d.uptake)}`)
    .join(' ')

  const yTicks = Array.from({ length: 5 }, (_, i) => yMin + (i * (yMax - yMin)) / 4)

  return (
    <div className="chart-scroll" role="region" aria-label="Pressure uptake chart" tabIndex={0}>
    <svg className="pressure-curve" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
      <line
        x1={padding.left}
        y1={padding.top + chartHeight}
        x2={padding.left + chartWidth}
        y2={padding.top + chartHeight}
        stroke="var(--surface-2)"
      />
      <line
        x1={padding.left}
        y1={padding.top}
        x2={padding.left}
        y2={padding.top + chartHeight}
        stroke="var(--surface-2)"
      />
      {yTicks.map((t, i) => (
        <g key={i}>
          <line
            x1={padding.left}
            y1={yScale(t)}
            x2={padding.left + chartWidth}
            y2={yScale(t)}
            stroke="var(--surface-2)"
            strokeDasharray="4"
          />
          <text x={padding.left - 8} y={yScale(t)} fill="var(--text-muted)" fontSize="11" textAnchor="end" dy="4">
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      {data.map((d) => (
        <g key={d.pressure}>
          <circle cx={xScale(d.pressure)} cy={yScale(d.uptake)} r="5" fill="var(--accent)" />
          <text
            x={xScale(d.pressure)}
            y={padding.top + chartHeight + 18}
            fill="var(--text-muted)"
            fontSize="11"
            textAnchor="middle"
          >
            {d.pressure} bar
          </text>
        </g>
      ))}
      <path d={path} fill="none" stroke="var(--accent)" strokeWidth="2.5" />
      <text x={padding.left} y={height - 5} fill="var(--text-muted)" fontSize="12">
        Pressure (bar, logarithmic scale)
      </text>
      <text
        x={15}
        y={padding.top + chartHeight / 2}
        fill="var(--text-muted)"
        fontSize="12"
        transform={`rotate(-90, 15, ${padding.top + chartHeight / 2})`}
        textAnchor="middle"
      >
        CO₂ uptake (mol/kg)
      </text>
    </svg>
    </div>
  )
}

function ShapChart({ data }: { data: ShapData | undefined }) {
  const rows = useMemo(() => {
    if (!data) return []
    const combined = data.feature_names.map((name, i) => ({
      name,
      value: data.feature_values[i],
      contribution: data.contributions[i],
    }))
    combined.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    return combined.slice(0, 12)
  }, [data])

  if (!data) return null

  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.contribution)), 1e-9)
  const width = 900
  const height = rows.length * 32 + 65
  const barMax = 170
  const labelEnd = 420
  const zeroX = 610

  return (
    <div className="chart-scroll" role="region" aria-label="Selected pressure SHAP chart" tabIndex={0}>
    <svg className="shap-chart" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
      <text x={20} y={20} fill="var(--text-muted)" fontSize="12">
        Baseline: {data.baseline.toFixed(4)} mol/kg
      </text>
      <text x={labelEnd} y={45} fill="var(--text-muted)" fontSize="12" textAnchor="end">Feature</text>
      <text x={zeroX} y={45} fill="var(--text-muted)" fontSize="12" textAnchor="middle">Contribution (mol/kg)</text>
      <text x={width - 10} y={45} fill="var(--text-muted)" fontSize="12" textAnchor="end">Value</text>
      <line x1={zeroX} x2={zeroX} y1={52} y2={height - 5} stroke="var(--text-muted)" />
      {rows.map((row, i) => {
        const y = 55 + i * 32
        const barWidth = (row.contribution / maxAbs) * barMax
        const barX = row.contribution >= 0 ? zeroX : zeroX + barWidth
        const absWidth = Math.abs(barWidth)
        return (
          <g key={row.name}>
            <text x={labelEnd} y={y + 16} fill="var(--text)" fontSize="12" textAnchor="end">
              {row.name}
            </text>
            <rect
              x={barX}
              y={y + 4}
              width={absWidth}
              height={20}
              fill={row.contribution >= 0 ? 'var(--success)' : 'var(--danger)'}
              rx="4"
            />
            <text
              x={width - 10}
              y={y + 18}
              fill="var(--text-muted)"
              fontSize="11"
              textAnchor="end"
            >
              {row.contribution > 0 ? '+' : ''}
              {row.contribution.toExponential(2)}
            </text>
          </g>
        )
      })}
    </svg>
    </div>
  )
}

function FeatureTable({ features }: { features: Record<string, number> }) {
  const rows = Object.entries(features)
  return (
    <div style={{ maxHeight: '320px', overflow: 'auto', marginTop: '1rem' }}>
      <table className="feature-table">
        <thead>
          <tr>
            <th>Feature</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, value]) => (
            <tr key={name}>
              <td>{name}</td>
              <td>{typeof value === 'number' ? value.toExponential(4) : value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
