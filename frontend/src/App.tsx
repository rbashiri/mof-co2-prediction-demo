import { useEffect, useMemo, useState } from 'react'
import CrystalViewer from './CrystalViewer'
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
  { name: 'lcd', label: 'Largest cavity diameter (angstrom)', short: 'LCD', unit: 'Å', desc: 'Maximum included sphere' },
  { name: 'pld', label: 'Pore limiting diameter (angstrom)', short: 'PLD', unit: 'Å', desc: 'Maximum free sphere' },
  { name: 'void_fraction', label: 'Void fraction (0-1)', short: 'Void Fraction', unit: 'φ (0–1)', desc: 'Total probe-occupiable fraction' },
  { name: 'surface_area_m2g', label: 'Surface area (m2/g)', short: 'Surface Area', unit: 'm²/g', desc: 'Accessible + pocket surface' },
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
          <path
            d="M40 12 L64 26 L64 54 L40 68 L16 54 L16 26 Z M16 26 L64 54 M64 26 L16 54"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          />
          <circle cx="40" cy="12" r="5" />
          <circle cx="64" cy="26" r="5" />
          <circle cx="64" cy="54" r="5" />
          <circle cx="40" cy="68" r="5" />
          <circle cx="16" cy="54" r="5" />
          <circle cx="16" cy="26" r="5" />
        </g>
        <circle cx="40" cy="40" r="7" className="molecule-center" />
      </svg>
      <div className="progress-copy">
        <div role="status">
          <strong>Examining your MOF crystal lattice…</strong>
          <p>
            Running Zeo++ Voronoi pore analysis, MOFid linker fragmentation, RDKit chemistry, and 5-pressure XGBoost + SHAP models.
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
  const [showPipelineInfo, setShowPipelineInfo] = useState(false)
  const [selectedPressure, setSelectedPressure] = useState<string>('0.1')
  const [summaryLoading, setSummaryLoading] = useState(false)
  const [featureSearch, setFeatureSearch] = useState('')

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
      setResult((current) =>
        current?.explanation_token === token ? { ...current, gemini_summary: data.gemini_summary } : current
      )
    } catch (err) {
      setResult((current) =>
        current?.explanation_token === token
          ? { ...current, gemini_error: err instanceof Error ? err.message : 'Explanation failed' }
          : current
      )
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

  const maxUptake = result ? Math.max(...result.pressures.map((p) => p.uptake), 0.001) : 1

  return (
    <div className="app-shell">
      <div className="ambient-glow ambient-glow-left" aria-hidden="true" />
      <div className="ambient-glow ambient-glow-right" aria-hidden="true" />

      <div className="container">
        <header className="hero-header">
          <h1>PRISM</h1>
          <p className="header-expansion">Pore Regression &amp; Interpretability for Simulated MOFs</p>
          <p className="header-subtitle">
            <strong>Upload a CIF to extract all descriptors automatically and predict CO₂ uptake</strong>
          </p>
          <p className="header-tagline">
            <strong>From crystal structure to carbon-capture insights</strong>
          </p>
          <div className="hero-badge-row hero-badge-row-bottom">
            <span className="hero-pill">
              <span className="hero-pill-dot" aria-hidden="true" />
              PHASE 3 MODEL B · 48 DESCRIPTORS
            </span>
            <span className="hero-pill hero-pill-subtle">ZEO++ · MOFID · RDKIT · PYMATGEN</span>
            <span className="model-estimates-badge">Model estimates · MVP</span>
          </div>
        </header>

        {/* Fixed 2-Column Top Workspace: Upload on Left, 3D Visualizer Always on Right */}
        <div className="workspace-fixed-grid">
          <form onSubmit={handleSubmit} className="workspace-upload-col">
            <div className="card upload-card">
              <div className="card-header-inline">
                <div>
                  <span className="section-eyebrow">STRUCTURE INPUT</span>
                  <h3>Upload Crystallographic Information File (.cif)</h3>
                </div>
                {file && (
                  <span className="file-ready-pill">
                    Ready for analysis · {(file.size / 1024).toFixed(1)} KB
                  </span>
                )}
              </div>

              <label
                className={`dropzone ${dragOver ? 'dragover' : ''} ${file ? 'has-file' : ''}`}
                onDragOver={(e) => {
                  e.preventDefault()
                  setDragOver(true)
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
              >
                <input
                  type="file"
                  accept=".cif"
                  onChange={handleFileChange}
                  disabled={loading || summaryLoading}
                  hidden
                />
                <div className="dropzone-icon" aria-hidden="true">
                  <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.8">
                    <path d="M24 6 39 14.5v19L24 42 9 33.5v-19L24 6Z" />
                    <path d="M24 16v14M18 21l6-6 6 6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
                <div className="dropzone-title">Drag & drop a CIF file here, or click to browse</div>
                <div className="dropzone-subtitle">
                  Instant 3D unit-cell preview + automatic Zeo++ & MOFid/RDKit/pymatgen extraction
                </div>
                {file && (
                  <div className="file-name">
                    <span className="file-name-icon" aria-hidden="true">◆</span>
                    {file.name}
                  </div>
                )}
              </label>

              <div className="actions">
                <button type="submit" className="btn" disabled={!file || loading || summaryLoading}>
                  {loading ? 'Extracting & predicting…' : 'Extract descriptors and predict'}
                </button>
              </div>
            </div>
          </form>

          <div className="workspace-viewer-col">
            <CrystalViewer file={file} />
          </div>
        </div>

        {loading && <PredictionProgress />}

        {error && (
          <div className="error" role="alert">
            <strong>Processing error:</strong> {error}
          </div>
        )}

        {/* Main Prediction Summary, AI Interpretation, Uptake, SHAP, and Geometry right below */}
        {result && (
          <div className="results-stack">
            <div className="card gemini-card">
              {!result.gemini_summary && (
                <div className="gemini-cta-panel">
                  <div>
                    <span className="section-eyebrow">AI SCIENTIFIC NARRATION · OPTIONAL</span>
                    <h3>Synthesize Structure–Property Insights</h3>
                    <p className="about-body">
                      Generate a concise scientific interpretation of the 5-pressure isotherm and endpoint SHAP
                      attributions using Gemini. Only validated model outputs are shared—your CIF structure never leaves
                      the server.
                    </p>
                  </div>
                  <div className="explanation-action">
                    <button className="btn" onClick={explainResults} disabled={summaryLoading}>
                      {summaryLoading ? 'Preparing your explanation...' : 'Click here to learn more about the results'}
                    </button>
                  </div>
                </div>
              )}
              {summaryLoading && (
                <p className="insights-status" role="status">
                  <span aria-hidden="true">⌛</span>
                  Preparing scientific insights from adsorption trends and SHAP results…
                </p>
              )}
              {result.gemini_summary && <ScientificInterpretation text={result.gemini_summary} />}
              {result.gemini_error && (
                <div className="error" role="alert" style={{ marginTop: '1rem', marginBottom: 0 }}>
                  Gemini summary unavailable: {result.gemini_error} Predictions and SHAP are unchanged.
                </div>
              )}
            </div>

            <div className="results-two-col">
              <div className="card">
                <div className="card-header-inline">
                  <div>
                    <span className="section-eyebrow">ADSORPTION ISOTHERM</span>
                    <h3>Predicted CO₂ uptake</h3>
                  </div>
                  <button className="btn btn-secondary btn-compact" onClick={downloadCsv}>
                    ↓ Download predictions CSV
                  </button>
                </div>

                {result.pressures.some((point) => point.uptake < 0) && (
                  <p role="alert" className="about-body warning-callout">
                    The model returned negative uptake, which is not physically meaningful. Values are shown unchanged; do
                    not treat these predictions as validated.
                  </p>
                )}

                <div className="metric-grid">
                  {result.pressures.map((p) => {
                    const ratio = Math.max(0, Math.min(100, (p.uptake / maxUptake) * 100))
                    return (
                      <div className="metric" key={p.pressure}>
                        <div className="metric-label">{p.pressure} bar</div>
                        <div className="metric-value">{p.uptake.toFixed(3)}</div>
                        <div className="metric-unit">mol / kg</div>
                        <div className="metric-bar-track" aria-hidden="true">
                          <div className="metric-bar-fill" style={{ width: `${ratio}%` }} />
                        </div>
                      </div>
                    )
                  })}
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
                <div className="card-header-inline">
                  <div>
                    <span className="section-eyebrow">MODEL INTERPRETABILITY</span>
                    <h3>SHAP explanation</h3>
                  </div>
                  <div className="shap-controls">
                    <label htmlFor="pressure">Pressure:</label>
                    <div className="pressure-pill-group" role="group" aria-label="Quick select pressure">
                      {PRESSURES.map((p) => {
                        const val = String(p)
                        return (
                          <button
                            key={p}
                            type="button"
                            className={`pressure-pill ${selectedPressure === val ? 'active' : ''}`}
                            onClick={() => setSelectedPressure(val)}
                          >
                            {p} bar
                          </button>
                        )
                      })}
                    </div>
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
                </div>

                <ShapChart data={result.shap[selectedPressure]} />
                <p className="shap-disclaimer">
                  SHAP describes model contributions relative to the pressure model’s baseline, not proven chemical mechanisms.
                </p>
              </div>
            </div>

            <div className="card">
              <div className="card-header-inline">
                <div>
                  <span className="section-eyebrow">PORE ARCHITECTURE</span>
                  <h3>Geometry extracted by Zeo++</h3>
                </div>
                <span className="source-badge">
                  {result.geometry_source === 'zeopp' ? 'Zeo++ 0.3 High-Accuracy' : 'Manual Geometry'}
                </span>
              </div>

              <div className="geometry-bento">
                {GEOMETRY_FIELDS.map(({ name, short, unit, desc }) => (
                  <div className="geometry-stat-card" key={name}>
                    <div className="geometry-stat-top">
                      <span className="geometry-stat-short">{short}</span>
                      <span className="geometry-stat-unit">{unit}</span>
                    </div>
                    <div className="geometry-stat-value">
                      {Number(result.geometry[name].toPrecision(6)).toString()}
                    </div>
                    <div className="geometry-stat-desc">{desc}</div>
                  </div>
                ))}
              </div>

              <table className="feature-table">
                <thead>
                  <tr>
                    <th>Descriptor</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {GEOMETRY_FIELDS.map(({ name, label }) => (
                    <tr key={name}>
                      <th scope="row">{label}</th>
                      <td className="mono-cell">{Number(result.geometry[name].toPrecision(6)).toString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {result.geometry_settings && (
                <p className="about-body">
                  Settings chosen by comparing Zeo++ geometry with reference values for 10 sample MOFs:{' '}
                  <strong>{result.geometry_settings.surface_probe_radius_angstrom} Å</strong> probe for surface area and{' '}
                  <strong>{result.geometry_settings.volume_probe_radius_angstrom} Å</strong> probe for void fraction.
                </p>
              )}
            </div>

            <div className="card">
              <div className="card-header-inline" style={{ marginBottom: showFeatures ? '1rem' : 0 }}>
                <div>
                  <span className="section-eyebrow">COMPLETE FEATURE VECTOR</span>
                  <h3>48 Phase 3 Model B Descriptors</h3>
                </div>
                <button
                  className="btn btn-secondary btn-compact"
                  aria-expanded={showFeatures}
                  aria-controls="model-input-features"
                  onClick={() => setShowFeatures((s) => !s)}
                >
                  {showFeatures ? 'Hide' : 'Inspect'} the 48 model input features
                </button>
              </div>
              <div id="model-input-features" hidden={!showFeatures}>
                {showFeatures && (
                  <FeatureTable
                    features={result.features}
                    search={featureSearch}
                    onSearchChange={setFeatureSearch}
                  />
                )}
              </div>
            </div>
          </div>
        )}

        {/* Collapsible Model & Scientific Tools Card (below main workspace & results) */}
        <section className="card prediction-guide collapsible-guide" aria-labelledby="prediction-guide-title">
          <button
            type="button"
            className="guide-collapse-trigger"
            aria-expanded={showPipelineInfo}
            aria-controls="prediction-guide-content"
            onClick={() => setShowPipelineInfo((v) => !v)}
          >
            <div className="prediction-guide-heading">
              <div>
                <span className="section-eyebrow">COMPUTATIONAL PIPELINE & PACKAGES</span>
                <h2 id="prediction-guide-title">About these predictions & scientific tools</h2>
              </div>
              <div className="guide-trigger-right">
                <span className="guide-chevron">
                  {showPipelineInfo ? 'Hide details ▲' : 'Click to view packages & model info ▼'}
                </span>
              </div>
            </div>
          </button>

          <div id="prediction-guide-content" hidden={!showPipelineInfo} className="guide-collapsible-body">
            <p className="prediction-guide-description">
              Predict CO₂ uptake across five pressures (0.01 to 2.5 bar) using frozen XGBoost models trained on simulated
              hMOF structures. Results are computational estimates, not experimental measurements.
            </p>
            <h3>Scientific tools behind the predictions</h3>
            <div className="science-tools">
              <div className="science-tool tool-geometry">
                <span className="tool-step">01</span>
                <div className="tool-icon-wrap">
                  <svg viewBox="0 0 48 48" aria-hidden="true">
                    <path d="m24 5 16 9v20l-16 9-16-9V14Zm0 0v19m16-10-16 10L8 14m16 10v19" />
                  </svg>
                </div>
                <strong>Zeo++</strong>
                <span>Pore geometry (LCD, PLD, ASA, POAV)</span>
              </div>
              <div className="science-tool tool-linkers">
                <span className="tool-step">02</span>
                <div className="tool-icon-wrap">
                  <svg viewBox="0 0 48 48" aria-hidden="true">
                    <path d="m24 13-12 20m12-20 12 20M12 33h24" />
                    <circle cx="24" cy="9" r="5" />
                    <circle cx="10" cy="36" r="5" />
                    <circle cx="38" cy="36" r="5" />
                  </svg>
                </div>
                <strong>MOFid</strong>
                <span>Linker & node fragmentation</span>
              </div>
              <div className="science-tool tool-chemistry">
                <span className="tool-step">03</span>
                <div className="tool-icon-wrap">
                  <svg viewBox="0 0 48 48" aria-hidden="true">
                    <ellipse cx="24" cy="24" rx="20" ry="8" transform="rotate(45 24 24)" />
                    <ellipse cx="24" cy="24" rx="20" ry="8" transform="rotate(-45 24 24)" />
                    <circle cx="24" cy="24" r="3" />
                  </svg>
                </div>
                <strong>RDKit</strong>
                <span>29 organic linker descriptors</span>
              </div>
              <div className="science-tool tool-materials">
                <span className="tool-step">04</span>
                <div className="tool-icon-wrap">
                  <svg viewBox="0 0 48 48" aria-hidden="true">
                    <path d="m24 5 10 6v12l-10 6-10-6V11Zm-10 18-10 6v12l10 5 10-5V29m10-18 10 6v12l-10 6-10-6" />
                  </svg>
                </div>
                <strong>pymatgen</strong>
                <span>15 metal & unit-cell features</span>
              </div>
            </div>
          </div>
        </section>

        <TeamContact />
      </div>
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
      <div className="interpretation-header">
        <div>
          <span className="section-eyebrow">AI-ASSISTED LITERATURE SYNTHESIS</span>
          <h3>Scientific interpretation</h3>
        </div>
        <p className="insights-status insights-ready" role="status">
          <span aria-hidden="true">✓</span> Scientific insights ready
        </p>
      </div>
      {parts[0].trim() && <p>{parts[0].trim()}</p>}
      <div className="interpretation-grid">
        {parts.map(
          (part, index) =>
            headings[part] && (
              <section className="interpretation-section" key={`${part}-${index}`}>
                <h4>{headings[part]}</h4>
                <p>{parts[index + 1]?.trim()}</p>
              </section>
            )
        )}
      </div>
      <section className="interpretation-section interpretation-limitations">
        <h4>Scientific limitations</h4>
        <p>
          AI interpretation of simulated hMOF predictions, not experimental results. SHAP explains model contributions,
          not chemical causes; contribution sizes are not directly comparable across pressure models. Experimental
          validation is needed.
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
        <tr>
          <th scope="col">Pressure (bar)</th>
          <th scope="col">Predicted CO₂ uptake (mol/kg)</th>
          <th scope="col">Comparison</th>
        </tr>
      </thead>
      <tbody>
        {[...data]
          .sort((a, b) => a.pressure - b.pressure)
          .map((point) => (
            <tr
              key={point.pressure}
              className={
                point.uptake === highest ? 'uptake-highest' : point.uptake === lowest ? 'uptake-lowest' : ''
              }
            >
              <th scope="row" className="mono-cell">
                {point.pressure}
              </th>
              <td className="mono-cell">{Number(point.uptake.toPrecision(6)).toString()}</td>
              <td>
                {point.uptake === highest && point.uptake === lowest ? (
                  <span className="comparison-badge">Highest / lowest (equal)</span>
                ) : point.uptake === highest ? (
                  <span className="comparison-badge badge-high">Highest</span>
                ) : point.uptake === lowest ? (
                  <span className="comparison-badge badge-low">Lowest</span>
                ) : (
                  '—'
                )}
              </td>
            </tr>
          ))}
      </tbody>
    </table>
  )
}

function PressureCurve({ data }: { data: PressurePoint[] }) {
  const width = 740
  const height = 280
  const padding = { top: 24, right: 40, bottom: 58, left: 78 }
  const chartWidth = width - padding.left - padding.right
  const chartHeight = height - padding.top - padding.bottom

  const xMin = Math.log10(Math.min(...data.map((d) => d.pressure)))
  const xMax = Math.log10(Math.max(...data.map((d) => d.pressure)))
  const yMax = Math.max(0, ...data.map((d) => d.uptake)) * 1.12 || 1
  const yMin = Math.min(0, ...data.map((d) => d.uptake)) * 1.12

  const xScale = (p: number) => padding.left + ((Math.log10(p) - xMin) / (xMax - xMin)) * chartWidth
  const yScale = (v: number) => padding.top + chartHeight - ((v - yMin) / (yMax - yMin)) * chartHeight

  const sorted = [...data].sort((a, b) => a.pressure - b.pressure)
  const path = sorted.map((d, i) => `${i === 0 ? 'M' : 'L'} ${xScale(d.pressure)} ${yScale(d.uptake)}`).join(' ')
  const areaPath = `${path} L ${xScale(sorted[sorted.length - 1].pressure)} ${padding.top + chartHeight} L ${xScale(
    sorted[0].pressure
  )} ${padding.top + chartHeight} Z`

  const yTicks = Array.from({ length: 5 }, (_, i) => yMin + (i * (yMax - yMin)) / 4)

  return (
    <div className="chart-scroll" role="region" aria-label="Pressure uptake chart" tabIndex={0}>
      <svg className="pressure-curve" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id="isothermAreaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.24" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.01" />
          </linearGradient>
        </defs>
        {yTicks.map((t, i) => (
          <g key={i}>
            <line
              x1={padding.left}
              y1={yScale(t)}
              x2={padding.left + chartWidth}
              y2={yScale(t)}
              stroke="var(--border-subtle)"
              strokeDasharray="4 4"
            />
            <text
              x={padding.left - 10}
              y={yScale(t)}
              fill="var(--text-muted)"
              fontSize="11"
              fontFamily="var(--font-mono)"
              textAnchor="end"
              dy="4"
            >
              {t.toFixed(2)}
            </text>
          </g>
        ))}
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
        <path d={areaPath} fill="url(#isothermAreaGrad)" />
        <path d={path} fill="none" stroke="var(--accent)" strokeWidth="3" strokeLinecap="round" />
        {sorted.map((d) => (
          <g key={d.pressure}>
            <circle cx={xScale(d.pressure)} cy={yScale(d.uptake)} r="8" fill="var(--accent)" fillOpacity="0.16" />
            <circle
              cx={xScale(d.pressure)}
              cy={yScale(d.uptake)}
              r="4.5"
              fill="var(--surface)"
              stroke="var(--accent)"
              strokeWidth="2.5"
            />
            <text
              x={xScale(d.pressure)}
              y={padding.top + chartHeight + 20}
              fill="var(--text-muted)"
              fontSize="11"
              fontFamily="var(--font-mono)"
              textAnchor="middle"
            >
              {d.pressure} bar
            </text>
          </g>
        ))}
        <text
          x={padding.left + chartWidth / 2}
          y={height - 8}
          fill="var(--text-muted)"
          fontSize="12"
          fontWeight="600"
          textAnchor="middle"
        >
          Pressure (bar, logarithmic scale)
        </text>
        <text
          x={18}
          y={padding.top + chartHeight / 2}
          fill="var(--text-muted)"
          fontSize="12"
          fontWeight="600"
          transform={`rotate(-90, 18, ${padding.top + chartHeight / 2})`}
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
  const width = 680
  const height = rows.length * 34 + 70
  const barMax = 135
  const labelEnd = 260
  const zeroX = 430

  return (
    <div className="chart-scroll" role="region" aria-label="Selected pressure SHAP chart" tabIndex={0}>
      <svg className="shap-chart" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
        <text x={20} y={22} fill="var(--text-muted)" fontSize="12" fontFamily="var(--font-mono)" fontWeight="600">
          Model baseline: {data.baseline.toFixed(4)} mol/kg
        </text>
        <text x={labelEnd} y={46} fill="var(--text-muted)" fontSize="11" fontWeight="700" textAnchor="end">
          DESCRIPTOR
        </text>
        <text x={zeroX} y={46} fill="var(--text-muted)" fontSize="11" fontWeight="700" textAnchor="middle">
          SHAP CONTRIBUTION (mol/kg)
        </text>
        <text x={width - 12} y={46} fill="var(--text-muted)" fontSize="11" fontWeight="700" textAnchor="end">
          SIGNED PUSH
        </text>
        <line x1={zeroX} x2={zeroX} y1={54} y2={height - 8} stroke="var(--surface-2)" strokeWidth="1.5" />
        {rows.map((row, i) => {
          const y = 58 + i * 34
          const barWidth = (row.contribution / maxAbs) * barMax
          const barX = row.contribution >= 0 ? zeroX : zeroX + barWidth
          const absWidth = Math.max(Math.abs(barWidth), 2)
          return (
            <g key={row.name}>
              {i % 2 === 0 && (
                <rect x={12} y={y - 2} width={width - 24} height={30} rx="6" fill="var(--row-stripe)" />
              )}
              <text
                x={labelEnd}
                y={y + 18}
                fill="var(--text)"
                fontSize="12"
                fontFamily="var(--font-mono)"
                textAnchor="end"
              >
                {row.name}
              </text>
              <rect
                x={barX}
                y={y + 5}
                width={absWidth}
                height={18}
                fill={row.contribution >= 0 ? 'var(--success)' : 'var(--danger)'}
                rx="4"
              />
              <text
                x={width - 12}
                y={y + 18}
                fill={row.contribution >= 0 ? 'var(--success)' : 'var(--danger)'}
                fontSize="11.5"
                fontFamily="var(--font-mono)"
                fontWeight="600"
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

function FeatureTable({
  features,
  search,
  onSearchChange,
}: {
  features: Record<string, number>
  search: string
  onSearchChange: (val: string) => void
}) {
  const rows = Object.entries(features).filter(([name]) =>
    name.toLowerCase().includes(search.trim().toLowerCase())
  )
  return (
    <div className="feature-inspector">
      <div className="feature-search-bar">
        <input
          type="search"
          placeholder="Filter 48 descriptors by name (e.g. lcd, ring, metal, tpsa)…"
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
          aria-label="Filter model input features"
        />
        <span className="feature-count-badge">{rows.length} / 48 shown</span>
      </div>
      <div className="feature-table-scroll">
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
                <td className="mono-cell">{name}</td>
                <td className="mono-cell">{typeof value === 'number' ? value.toExponential(4) : value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
