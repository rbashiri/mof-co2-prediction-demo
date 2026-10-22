import { useEffect, useMemo, useRef, useState } from 'react'
import * as $3Dmol from '3dmol'

export type RenderStyle = 'ballstick' | 'stick' | 'sphere' | 'line'

interface CifMeta {
  a?: string
  b?: string
  c?: string
  alpha?: string
  beta?: string
  gamma?: string
  formula?: string
  spaceGroup?: string
}

function parseCifMetadata(cifText: string): CifMeta {
  const pick = (pattern: RegExp) => {
    const match = cifText.match(pattern)
    if (!match?.[1]) return undefined
    const raw = match[1].trim().replace(/^['"]|['"]$/g, '')
    const num = Number(raw.replace(/\(\d+\)$/, ''))
    return Number.isFinite(num) ? Number(num.toPrecision(5)).toString() : raw
  }

  return {
    a: pick(/_cell_length_a\s+([^\s#]+)/i),
    b: pick(/_cell_length_b\s+([^\s#]+)/i),
    c: pick(/_cell_length_c\s+([^\s#]+)/i),
    alpha: pick(/_cell_angle_alpha\s+([^\s#]+)/i),
    beta: pick(/_cell_angle_beta\s+([^\s#]+)/i),
    gamma: pick(/_cell_angle_gamma\s+([^\s#]+)/i),
    formula: pick(/_chemical_formula_sum\s+([^\r\n#]+)/i),
    spaceGroup:
      pick(/_symmetry_space_group_name_H-M\s+([^\r\n#]+)/i) ||
      pick(/_space_group_name_H-M_alt\s+([^\r\n#]+)/i),
  }
}

export default function CrystalViewer({ file }: { file: File | null }) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const viewerRef = useRef<$3Dmol.GLViewer | null>(null)

  const [cifText, setCifText] = useState<string>('')
  const [styleMode, setStyleMode] = useState<RenderStyle>('ballstick')
  const [showUnitCell, setShowUnitCell] = useState<boolean>(true)
  const [supercell, setSupercell] = useState<boolean>(false)
  const [spinning, setSpinning] = useState<boolean>(true)
  const [atomCount, setAtomCount] = useState<number | null>(null)
  const [viewerError, setViewerError] = useState<string | null>(null)

  useEffect(() => {
    if (!file) {
      setCifText('')
      setAtomCount(null)
      setViewerError(null)
      return
    }
    let active = true
    setViewerError(null)
    file
      .text()
      .then((text) => {
        if (!active) return
        if (!text.trim()) {
          setViewerError('Uploaded CIF file is empty.')
          setCifText('')
          return
        }
        // Normalize tabs to spaces because 3Dmol's CIF loop parser splits on ' ' only.
        setCifText(text.replace(/\t+/g, ' '))
      })
      .catch(() => {
        if (active) setViewerError('Unable to read CIF file for 3D preview.')
      })
    return () => {
      active = false
    }
  }, [file])

  const meta = useMemo(() => (cifText ? parseCifMetadata(cifText) : {}), [cifText])

  useEffect(() => {
    const el = containerRef.current
    if (!el || !cifText) {
      if (el) el.innerHTML = ''
      return
    }

    try {
      el.innerHTML = ''
      const viewer = $3Dmol.createViewer(el, {
        backgroundColor: '#0b222c',
        antialias: true,
      })
      viewerRef.current = viewer

      const model = viewer.addModel(cifText, 'cif', {
        doAssembly: true,
        duplicateAssemblyAtoms: true,
        normalizeAssembly: true,
        assignBonds: true,
      })

      if (supercell) {
        viewer.replicateUnitCell(2, 2, 2, model)
      }

      if (styleMode === 'ballstick') {
        viewer.setStyle(
          {},
          {
            stick: { radius: 0.14, colorscheme: 'Jmol' },
            sphere: { scale: 0.28, colorscheme: 'Jmol' },
          }
        )
      } else if (styleMode === 'stick') {
        viewer.setStyle({}, { stick: { radius: 0.18, colorscheme: 'Jmol' } })
      } else if (styleMode === 'sphere') {
        viewer.setStyle({}, { sphere: { scale: 0.65, colorscheme: 'Jmol' } })
      } else {
        viewer.setStyle({}, { line: { linewidth: 2, colorscheme: 'Jmol' } })
      }

      if (showUnitCell) {
        viewer.addUnitCell(model, {
          box: { color: '#5eead4', linewidth: 1.5 },
          alabel: 'a',
          blabel: 'b',
          clabel: 'c',
          alabelstyle: { fontColor: '#f8fafc', fontSize: 13, backgroundOpacity: 0.35 },
          blabelstyle: { fontColor: '#f8fafc', fontSize: 13, backgroundOpacity: 0.35 },
          clabelstyle: { fontColor: '#f8fafc', fontSize: 13, backgroundOpacity: 0.35 },
        })
      }

      const atoms = model?.selectedAtoms?.({}) ?? []
      setAtomCount(Array.isArray(atoms) ? atoms.length : null)

      viewer.zoomTo()
      viewer.render()
      viewer.spin(spinning ? 'y' : false, 0.5)
      setViewerError(null)
    } catch (err) {
      setViewerError(err instanceof Error ? err.message : 'Unable to render 3D structure.')
    }

    const observer = new ResizeObserver(() => {
      if (viewerRef.current) {
        viewerRef.current.resize()
        viewerRef.current.render()
      }
    })
    observer.observe(el)

    return () => {
      observer.disconnect()
      if (viewerRef.current) {
        try {
          viewerRef.current.spin(false)
          viewerRef.current.clear()
        } catch {
          // Ignore WebGL cleanup errors in headless/unmounted environments
        }
        viewerRef.current = null
      }
    }
  }, [cifText, styleMode, showUnitCell, supercell, spinning])

  const resetCamera = () => {
    if (!viewerRef.current) return
    viewerRef.current.zoomTo()
    viewerRef.current.render()
  }

  return (
    <section className="card crystal-viewer-card" aria-label="Interactive 3D MOF Crystal Structure Viewer">
      <div className="card-header-inline crystal-header">
        <div>
          <span className="section-eyebrow">INTERACTIVE CRYSTALLOGRAPHY · 3DMOL WEBGL</span>
          <h3>{file ? `3D Crystal Structure — ${file.name}` : '3D Crystal Structure'}</h3>
        </div>
        <div className="crystal-badges">
          {file && atomCount !== null && atomCount > 0 ? (
            <span className="source-badge">{atomCount.toLocaleString()} rendered sites</span>
          ) : (
            <span className="source-badge">Awaiting CIF</span>
          )}
          {file && meta.spaceGroup && <span className="model-estimates-badge">SG: {meta.spaceGroup}</span>}
        </div>
      </div>

      <div className="crystal-toolbar" role="toolbar" aria-label="3D structure controls">
        <div className="crystal-style-pills" role="group" aria-label="Representation style">
          {(
            [
              ['ballstick', 'Ball & Stick'],
              ['stick', 'Bonds'],
              ['sphere', 'Spacefill (vdW)'],
              ['line', 'Wireframe'],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              type="button"
              disabled={!file}
              className={`pressure-pill ${styleMode === key ? 'active' : ''}`}
              onClick={() => setStyleMode(key)}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="crystal-toggle-group">
          <button
            type="button"
            disabled={!file}
            className={`crystal-chip-btn ${showUnitCell ? 'active' : ''}`}
            onClick={() => setShowUnitCell((v) => !v)}
          >
            {showUnitCell ? '✓ Unit Cell Box' : 'Unit Cell Box'}
          </button>
          <button
            type="button"
            disabled={!file}
            className={`crystal-chip-btn ${supercell ? 'active' : ''}`}
            onClick={() => setSupercell((v) => !v)}
          >
            {supercell ? '2×2×2 Supercell' : '1×1×1 Unit Cell'}
          </button>
          <button
            type="button"
            disabled={!file}
            className={`crystal-chip-btn ${spinning ? 'active' : ''}`}
            onClick={() => setSpinning((v) => !v)}
          >
            {spinning ? '⟳ Auto-Rotate: ON' : '⟳ Auto-Rotate: OFF'}
          </button>
          <button type="button" disabled={!file} className="crystal-chip-btn" onClick={resetCamera}>
            ⌖ Reset View
          </button>
        </div>
      </div>

      <div className="crystal-stage-wrap">
        <div
          ref={containerRef}
          className="crystal-stage"
          role="img"
          aria-label={file ? `3D molecular visualization of ${file.name}` : 'Empty 3D crystal structure viewport'}
        />
        {!file && (
          <div className="crystal-empty-state">
            <svg viewBox="0 0 64 64" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
              <path d="M32 6 54 19v26L32 58 10 45V19L32 6Z" />
              <path d="M32 6v26m22-13L32 32 10 19m22 13v26" opacity="0.55" />
              <circle cx="32" cy="32" r="4.5" fill="currentColor" />
              <circle cx="32" cy="6" r="3" fill="currentColor" />
              <circle cx="54" cy="19" r="3" fill="currentColor" />
              <circle cx="54" cy="45" r="3" fill="currentColor" />
              <circle cx="32" cy="58" r="3" fill="currentColor" />
              <circle cx="10" cy="45" r="3" fill="currentColor" />
              <circle cx="10" cy="19" r="3" fill="currentColor" />
            </svg>
            <strong>3D Unit-Cell Viewport Ready</strong>
            <span>Upload a .cif file on the left to inspect its 3D atomic framework, pore channels, and unit cell.</span>
          </div>
        )}
        {viewerError && (
          <div className="crystal-overlay-msg" role="alert">
            {viewerError}
          </div>
        )}
        <div className="crystal-hud">
          <span>Drag to rotate · Scroll to zoom · Shift+Drag to pan</span>
          <span className="crystal-hud-legend">
            <span><i style={{ background: '#909090' }} />C</span>
            <span><i style={{ background: '#ff0d0d' }} />O</span>
            <span><i style={{ background: '#3050f8' }} />N</span>
            <span><i style={{ background: '#ffffff' }} />H</span>
            <span><i style={{ background: '#7d80b0' }} />Metal</span>
          </span>
        </div>
      </div>

      <div className="crystal-cell-strip">
        <div className="cell-param">
          <span>a</span>
          <strong>{meta.a ? `${meta.a} Å` : '—'}</strong>
        </div>
        <div className="cell-param">
          <span>b</span>
          <strong>{meta.b ? `${meta.b} Å` : '—'}</strong>
        </div>
        <div className="cell-param">
          <span>c</span>
          <strong>{meta.c ? `${meta.c} Å` : '—'}</strong>
        </div>
        <div className="cell-param">
          <span>α</span>
          <strong>{meta.alpha ? `${meta.alpha}°` : '—'}</strong>
        </div>
        <div className="cell-param">
          <span>β</span>
          <strong>{meta.beta ? `${meta.beta}°` : '—'}</strong>
        </div>
        <div className="cell-param">
          <span>γ</span>
          <strong>{meta.gamma ? `${meta.gamma}°` : '—'}</strong>
        </div>
      </div>
    </section>
  )
}
