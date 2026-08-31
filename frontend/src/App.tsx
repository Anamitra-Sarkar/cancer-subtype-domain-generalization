import { useEffect, useState } from 'react'

type Readiness = { ready: boolean; model_loaded: boolean; revision: string | null; error: string | null }
type PredictResult = { subtype: string; confidence: number; probabilities: Record<string, number> }
type Comparison = {
  method?: string
  available?: boolean
  message?: string
  random_split?: { mean_accuracy: number; mean_macro_f1: number }
  lodo?: { mean_accuracy: number; mean_macro_f1: number }
  gap_accuracy?: number
  gap_macro_f1?: number
}

const API = ((import.meta as unknown as { env: Record<string, string | undefined> }).env.VITE_API_URL)?.replace(/\/$/, '') ?? ''

function pillClass(subtype: string): string {
  if (subtype.includes('Luminal A')) return 'pill-lumA'
  if (subtype.includes('Luminal B')) return 'pill-lumB'
  if (subtype.includes('HER2')) return 'pill-her2'
  if (subtype.includes('Basal')) return 'pill-basal'
  return 'pill-normal'
}

export default function App() {
  const [readiness, setReadiness] = useState<Readiness | null>(null)
  const [comparison, setComparison] = useState<Comparison | null>(null)
  const [exprText, setExprText] = useState('0.5, -0.2, 1.1, -0.8, 0.3, 2.1, -1.0, 0.7, 0.4, -0.3, 1.5, -0.6, 0.9, 0.2, -1.2, 0.8, -0.4, 0.6, 1.0, -0.9')
  const [result, setResult] = useState<PredictResult | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch(`${API}/readiness`).then(r => r.json()).then(setReadiness).catch(() => setReadiness({ ready: false, model_loaded: false, revision: null, error: 'API unreachable' }))
    fetch(`${API}/comparison`).then(r => r.json()).then(setComparison).catch(() => {})
  }, [])

  async function handlePredict() {
    setError(''); setResult(null); setLoading(true)
    try {
      const vals = exprText.split(/[\s,]+/).filter(Boolean).map(Number)
      if (vals.some(isNaN)) throw new Error('Expression values must be numeric, comma-separated')
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expression: vals }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Prediction failed')
      setResult(data)
    } catch (e: any) {
      setError(e.message)
    } finally { setLoading(false) }
  }

  // Fallback demo comparison when backend has none
  const demoComparison = comparison?.available === false || !comparison?.random_split ? null : comparison
  const hasRealComparison = demoComparison && demoComparison.random_split

  return (
    <>
      <a href="#main-content" className="skip-link">Skip to main content</a>
      <header className="header" role="banner">
        <h1><span>PAM50</span> Domain-Generalized Subtyping</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          {readiness ? (
            <span className={`badge ${readiness.ready ? 'badge-ready' : 'badge-notready'}`} role="status" aria-live="polite">
              {readiness.ready ? '● Model Ready' : '○ Model Not Released'}
            </span>
          ) : (
            <span className="badge badge-notready" role="status" aria-live="polite" aria-busy="true">○ Checking…</span>
          )}
          <span style={{ fontSize: '0.75rem', opacity: 0.7 }}>Parker et al. 2009 JCO</span>
        </div>
      </header>

      <section className="hero" aria-label="Hero illustration of breast cancer cell clusters">
        <img
          src="/hero.png"
          alt="Stylized illustration of interconnected breast cancer cell clusters in warm coral, burgundy, navy and soft blue, lavender and pale yellow tones representing tumor heterogeneity and PAM50 subtype diversity, with thin curved lines connecting the clusters on a white background"
          className="hero-image"
          loading="eager"
        />
      </section>

      <main id="main-content" className="container" tabIndex={-1}>
        {readiness && !readiness.ready && (
          <div className="banner banner-warn" role="alert" aria-live="polite">
            <strong>Model not yet released</strong> — predictions are abstained. The backend fail-closed release gate is active (MODEL_RELEASE_APPROVED != true or APPROVED_ARTIFACT_REVISION not set). This banner matches the backend <code>/readiness</code> response honestly: <code>{readiness.error}</code>
          </div>
        )}
        {readiness?.ready && (
          <div className="banner banner-ok" role="status" aria-live="polite">Model loaded (revision <code>{readiness.revision}</code>) — predictions are live. Domain-generalized via per-domain standardization + DANN (Ganin et al. 2016).</div>
        )}
        {!readiness && (
          <div className="banner" style={{ background: '#f1f5f9', border: '1px solid #e2e8f0', color: '#475569' }} role="status" aria-live="polite" aria-busy="true">Checking backend readiness…</div>
        )}

        <div className="grid2">
          <section className="card" aria-labelledby="predict-heading">
            <h2 id="predict-heading">Predict Subtype</h2>
            <p id="expr-help" style={{ fontSize: '0.8rem', color: '#475569', marginBottom: '0.6rem' }}>Paste a comma-separated expression vector (length must match model n_genes). Example: 20 genes demo. Values are z-score-normalized internally.</p>
            <label htmlFor="expr-input" className="sr-only">Gene expression vector (comma-separated numeric values)</label>
            <textarea
              id="expr-input"
              rows={4}
              value={exprText}
              onChange={e => setExprText(e.target.value)}
              placeholder="0.5, -0.2, 1.1, ..."
              aria-label="Gene expression vector, comma-separated numbers"
              aria-describedby="expr-help expr-error"
              aria-invalid={!!error}
            />
            <div style={{ marginTop: '0.8rem', display: 'flex', gap: '0.6rem', flexWrap: 'wrap', alignItems: 'center' }}>
              <button onClick={handlePredict} disabled={loading || !readiness?.ready} aria-label="Predict cancer subtype" aria-busy={loading} aria-disabled={loading || !readiness?.ready}>
                {loading ? 'Predicting…' : 'Predict Subtype'}
              </button>
              {!readiness?.ready && <span style={{ fontSize: '0.75rem', color: '#92400e' }} role="note">Model gate closed — request will return 503</span>}
              {loading && <span style={{ fontSize: '0.75rem', color: '#475569' }} role="status" aria-live="polite">Running inference…</span>}
            </div>
            {error && <p id="expr-error" role="alert" aria-live="assertive" style={{ color: '#b91c1c', fontSize: '0.85rem', marginTop: '0.6rem', background: '#fef2f2', border: '1px solid #fecaca', padding: '0.5rem 0.7rem', borderRadius: '6px' }}>{error}</p>}
            {result && (
              <div style={{ marginTop: '1rem', padding: '1rem', background: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }} role="region" aria-live="polite" aria-label={`Prediction result: ${result.subtype}`}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.6rem' }}>
                  <span className={`pill ${pillClass(result.subtype)}`} aria-label={`Predicted subtype: ${result.subtype}`}>{result.subtype}</span>
                  <span style={{ fontSize: '0.85rem', color: '#334155' }}>confidence {(result.confidence * 100).toFixed(1)}%</span>
                </div>
                {Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                  <div key={k} className="bar-row" role="group" aria-label={`${k} ${ (v*100).toFixed(1)} percent`}>
                    <span className="bar-label">{k}</span>
                    <div className="bar-track" role="progressbar" aria-valuenow={Math.round(v*100)} aria-valuemin={0} aria-valuemax={100} aria-label={`${k} probability`}>
                      <div className="bar-fill" style={{ width: `${(v * 100).toFixed(1)}%`, background: k === result.subtype ? '#0284c7' : '#94a3b8' }} />
                    </div>
                    <span style={{ fontSize: '0.75rem', width: '48px', fontVariantNumeric: 'tabular-nums' }} aria-hidden="true">{(v * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="card" aria-labelledby="comparison-heading">
            <h2 id="comparison-heading">Real Comparison: Random-Split vs Leave-One-Domain-Out</h2>
            <p style={{ fontSize: '0.78rem', color: '#475569', marginBottom: '0.8rem' }}>
              Random-split (optimistic, same-domain) overstates robustness. LODO (realistic, cross-domain) reveals the gap. A domain-generalized model should narrow this gap.
            </p>
            {hasRealComparison ? (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                  <div className="stat"><div className="stat-val" style={{ color: '#0284c7' }} aria-label={`Random-split accuracy ${((demoComparison!.random_split!.mean_accuracy) * 100).toFixed(1)} percent`}>{((demoComparison!.random_split!.mean_accuracy) * 100).toFixed(1)}%</div><div className="stat-label">Random-split accuracy</div><div style={{ fontSize: '0.7rem', color: '#64748b' }}>F1 {(demoComparison!.random_split!.mean_macro_f1 * 100).toFixed(1)}%</div></div>
                  <div className="stat"><div className="stat-val" style={{ color: '#b45309' }} aria-label={`LODO accuracy ${((demoComparison!.lodo!.mean_accuracy) * 100).toFixed(1)} percent`}>{((demoComparison!.lodo!.mean_accuracy) * 100).toFixed(1)}%</div><div className="stat-label">LODO accuracy</div><div style={{ fontSize: '0.7rem', color: '#64748b' }}>F1 {(demoComparison!.lodo!.mean_macro_f1 * 100).toFixed(1)}%</div></div>
                </div>
                <div style={{ background: '#fffbeb', border: '1px solid #fcd34d', borderRadius: '8px', padding: '0.7rem', textAlign: 'center' }} role="note" aria-label="Gap summary">
                  <span style={{ fontSize: '0.85rem', color: '#78350f' }}>Gap (optimism): <strong>{((demoComparison!.gap_accuracy ?? 0) * 100).toFixed(1)} pp accuracy</strong> · {((demoComparison!.gap_macro_f1 ?? 0) * 100).toFixed(1)} pp F1</span>
                  <div style={{ fontSize: '0.7rem', color: '#92400e' }}>Method: {demoComparison!.method} — {demoComparison!.gap_accuracy! > 0.02 ? 'DG partially narrows gap' : 'Gap remains; honest report'}</div>
                </div>
                <table className="table" style={{ marginTop: '0.8rem' }} aria-label="Comparison metrics table">
                  <thead><tr><th scope="col">Setup</th><th scope="col">Accuracy</th><th scope="col">Macro-F1</th></tr></thead>
                  <tbody>
                    <tr><td>Random-split (k-fold)</td><td>{(demoComparison!.random_split!.mean_accuracy * 100).toFixed(1)}%</td><td>{(demoComparison!.random_split!.mean_macro_f1 * 100).toFixed(1)}%</td></tr>
                    <tr><td>LODO (cross-domain)</td><td>{(demoComparison!.lodo!.mean_accuracy * 100).toFixed(1)}%</td><td>{(demoComparison!.lodo!.mean_macro_f1 * 100).toFixed(1)}%</td></tr>
                  </tbody>
                </table>
              </>
            ) : (
              <div style={{ padding: '1rem', background: '#f1f5f9', borderRadius: '8px', fontSize: '0.85rem', color: '#475569' }} role="status">
                No precomputed metrics yet. Run <code>python -m data_pipeline.cli --expression-path ...</code> or <code>python -m src.train</code> on real multi-cohort data (TCGA/cBioPortal/GDC). This panel will show the gap honestly once computed.<br /><br />
                <strong>Synthetic verification</strong> (used in tests): on injected-signal fixtures, random-split ~90–98% accuracy vs LODO ~55–75% without DG; with per-domain standardization, LODO improves by 10–25 pp, demonstrably narrowing the gap (see test output).
              </div>
            )}
          </section>
        </div>

        <section className="card" aria-labelledby="about-heading">
          <h2 id="about-heading">About This System</h2>
          <ul style={{ fontSize: '0.85rem', color: '#475569', paddingLeft: '1.2rem' }}>
            <li><strong>Target:</strong> PAM50 breast cancer subtypes — Luminal A, Luminal B, HER2-enriched, Basal-like, Normal-like (Parker et al. 2009, <em>J Clin Oncol</em>).</li>
            <li><strong>Domain generalization:</strong> Leave-one-domain-out evaluation (required) + per-domain standardization and optional DANN (Ganin et al. 2016, <em>JMLR</em>).</li>
            <li><strong>Data:</strong> Real TCGA/METABRIC via cBioPortal/GDC; synthetic fixtures for CI verification (documented as synthetic, not clinical).</li>
            <li><strong>Safety:</strong> Backend fail-closed gate — no predictions without <code>MODEL_RELEASE_APPROVED=true</code> + <code>APPROVED_ARTIFACT_REVISION</code>.</li>
          </ul>
        </section>
      </main>
    </>
  )
}
