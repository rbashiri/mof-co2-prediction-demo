import { useMemo, useState } from 'react'

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
  predictions_csv: string
}

const GEOMETRY_FIELDS = [
  { name: 'lcd', label: 'Largest cavity diameter (angstrom)' },
  { name: 'pld', label: 'Pore limiting diameter (angstrom)' },
  { name: 'void_fraction', label: 'Void fraction (0-1)' },
  { name: 'surface_area_m2g', label: 'Surface area (m2/g)' },
] as const

const PRESSURES = [0.01, 0.05, 0.1, 0.5, 2.5]

export default function App() {
  const [geometry, setGeometry] = useState<Record<string, string>>({
    lcd: '',
    pld: '',
    void_fraction: '',
    surface_area_m2g: '',
  })
  const [file, setFile] = useState<File | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [showAbout, setShowAbout] = useState(false)
  const [showFeatures, setShowFeatures] = useState(false)
  const [selectedPressure, setSelectedPressure] = useState<string>('0.1')

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0])
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setFile(e.target.files[0])
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    const form = new FormData()
    GEOMETRY_FIELDS.forEach(({ name }) => form.append(name, geometry[name]))
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
        <p>Upload a CIF and enter its four matching geometry values</p>
      </header>

      <div className="card">
        <button className="about-toggle" onClick={() => setShowAbout((s) => !s)}>
          {showAbout ? 'Hide' : 'Show'} about these predictions
        </button>
        {showAbout && (
          <div className="about-body">
            Geometry values must describe the uploaded structure and are supplied by you,
            not calculated by this app. The models predict simulated hMOF uptake; predictions
            are not experimental measurements and accuracy on other MOF families has not been
            established.
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit}>
        <div className="card">
          <div className="grid">
            {GEOMETRY_FIELDS.map(({ name, label }) => (
              <div className="field" key={name}>
                <label htmlFor={name}>{label}</label>
                <input
                  id={name}
                  type="number"
                  step="any"
                  min={name === 'void_fraction' ? 0 : undefined}
                  max={name === 'void_fraction' ? 1 : undefined}
                  required
                  value={geometry[name]}
                  onChange={(e) => setGeometry((g) => ({ ...g, [name]: e.target.value }))}
                />
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <label
            className={`dropzone ${dragOver ? 'dragover' : ''}`}
            onDragOver={(e) => {
              e.preventDefault()
              setDragOver(true)
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
          >
            <input type="file" accept=".cif" onChange={handleFileChange} hidden />
            <div>Drag & drop a CIF file here, or click to browse</div>
            {file && <div className="file-name">{file.name}</div>}
          </label>

          <div className="actions">
            <button type="submit" className="btn" disabled={!file || loading}>
              {loading ? 'Extracting & predicting…' : 'Extract descriptors and predict'}
            </button>
          </div>
        </div>
      </form>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <div className="card">
            <h3>Predicted CO₂ uptake</h3>
            <div className="metric-grid">
              {result.pressures.map((p) => (
                <div className="metric" key={p.pressure}>
                  <div className="metric-value">{p.uptake.toFixed(3)}</div>
                  <div className="metric-label">{p.pressure} bar</div>
                </div>
              ))}
            </div>
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
            <h3>Scientific summary</h3>
            <div className="summary-section">
              {Object.entries(result.summary).map(([title, text]) => (
                <div key={title}>
                  <h4>{title}</h4>
                  <p>{text}</p>
                </div>
              ))}
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '1rem' }}>
              Rule-based summary — not AI-generated. Draft for scientific review.
            </p>
          </div>

          <div className="card">
            <button className="about-toggle" onClick={() => setShowFeatures((s) => !s)}>
              {showFeatures ? 'Hide' : 'Inspect'} the 48 model input features
            </button>
            {showFeatures && <FeatureTable features={result.features} />}
          </div>
        </>
      )}
    </div>
  )
}

function PressureCurve({ data }: { data: PressurePoint[] }) {
  const width = 700
  const height = 260
  const padding = { top: 20, right: 30, bottom: 40, left: 50 }
  const chartWidth = width - padding.left - padding.right
  const chartHeight = height - padding.top - padding.bottom

  const xMin = data[0].pressure
  const xMax = data[data.length - 1].pressure
  const yMax = Math.max(...data.map((d) => d.uptake)) * 1.1
  const yMin = 0

  const xScale = (p: number) =>
    padding.left + ((p - xMin) / (xMax - xMin)) * chartWidth
  const yScale = (v: number) =>
    padding.top + chartHeight - ((v - yMin) / (yMax - yMin)) * chartHeight

  const path = data
    .map((d, i) => `${i === 0 ? 'M' : 'L'} ${xScale(d.pressure)} ${yScale(d.uptake)}`)
    .join(' ')

  const yTicks = Array.from({ length: 5 }, (_, i) => yMin + (i * (yMax - yMin)) / 4)

  return (
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
        Pressure (bar)
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
  const width = 600
  const height = rows.length * 32 + 40
  const barMax = 240
  const leftMargin = 220

  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" style={{ width: '100%', height: 'auto' }}>
      <text x={20} y={20} fill="var(--text-muted)" fontSize="12">
        Baseline: {data.baseline.toFixed(4)} mol/kg
      </text>
      {rows.map((row, i) => {
        const y = 35 + i * 32
        const barWidth = (row.contribution / maxAbs) * barMax
        const barX = row.contribution >= 0 ? leftMargin : leftMargin + barWidth
        const absWidth = Math.abs(barWidth)
        return (
          <g key={row.name}>
            <text x={leftMargin - 10} y={y + 16} fill="var(--text)" fontSize="12" textAnchor="end">
              {row.name.length > 32 ? row.name.slice(0, 29) + '…' : row.name}
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
              x={row.contribution >= 0 ? barX + absWidth + 6 : barX - 6}
              y={y + 18}
              fill="var(--text-muted)"
              fontSize="11"
              textAnchor={row.contribution >= 0 ? 'start' : 'end'}
            >
              {row.contribution > 0 ? '+' : ''}
              {row.contribution.toExponential(2)}
            </text>
          </g>
        )
      })}
    </svg>
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
