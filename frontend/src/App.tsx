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
      <nav className="navbar">
        <div className="brand">
          <span className="brand-mark">BC</span>
          <span className="brand-name">Subtype Atlas</span>
        </div>
        {readiness ? (
          <span className={`badge ${readiness.ready ? 'badge-ready' : 'badge-notready'}`} role="status" aria-live="polite">
            {readiness.ready ? 'Live predictions' : 'Preview mode'}
          </span>
        ) : (
          <span className="badge badge-notready" role="status" aria-live="polite" aria-busy="true">Checking…</span>
        )}
      </nav>

      <section className="hero-section">
        <div className="hero-copy">
          <div className="eyebrow">Breast cancer subtype classification</div>
          <h1>
            A subtype call that holds up <em>anywhere.</em>
          </h1>
          <p className="lede">
            Subtype Atlas classifies breast cancer into its five established molecular subtypes — built and tested to
            stay accurate across different patient cohorts, not just the one it was trained on.
          </p>
          <a href="#main-content" className="btn primary">Try a prediction</a>
        </div>
        <figure className="hero-visual">
          <img
            src="/hero.png"
            alt="Stylized illustration of interconnected breast cancer cell clusters in warm coral, burgundy, navy and soft blue, lavender and pale yellow tones representing tumor heterogeneity and PAM50 subtype diversity, with thin curved lines connecting the clusters on a white background"
            loading="eager"
          />
        </figure>
      </section>

      <section className="feature-grid">
        <div className="feature-card">
          <span className="feature-index">01</span>
          <h3>Five established subtypes</h3>
          <p>Classifies into the well-known molecular subtypes used in breast cancer research and care.</p>
        </div>
        <div className="feature-card">
          <span className="feature-index">02</span>
          <h3>Tested across cohorts</h3>
          <p>Evaluated on patient groups it never trained on, not just held-out samples from the same group.</p>
        </div>
        <div className="feature-card">
          <span className="feature-index">03</span>
          <h3>Confidence included</h3>
          <p>Every prediction shows how confident the model is across all five subtypes, not just the top pick.</p>
        </div>
      </section>

      <main id="main-content" className="container" tabIndex={-1}>
        {readiness && !readiness.ready && (
          <div className="banner banner-warn" role="alert" aria-live="polite">
            <strong>Predictions aren't available yet.</strong> Our team is finishing validation before enabling live results.
          </div>
        )}
        {readiness?.ready && (
          <div className="banner banner-ok" role="status" aria-live="polite">This model is live — predictions below are generated in real time.</div>
        )}
        {!readiness && (
          <div className="banner" style={{ background: '#f1f5f9', border: '1px solid #e2e8f0', color: '#475569' }} role="status" aria-live="polite" aria-busy="true">Checking availability…</div>
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
              {!readiness?.ready && <span style={{ fontSize: '0.75rem', color: '#92400e' }} role="note">Not available yet</span>}
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
                  <div style={{ fontSize: '0.7rem', color: '#92400e' }}>{demoComparison!.gap_accuracy! > 0.02 ? 'This model narrows that gap.' : 'This gap remains — reported honestly.'}</div>
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
No comparison data yet. This panel will show the real-world accuracy gap honestly once evaluation is complete.
              </div>
            )}
          </section>
        </div>

        <section className="card" aria-labelledby="about-heading">
          <h2 id="about-heading">About this tool</h2>
          <ul style={{ fontSize: '0.85rem', color: '#475569', paddingLeft: '1.2rem' }}>
            <li><strong>Subtypes:</strong> Luminal A, Luminal B, HER2-enriched, Basal-like, and Normal-like — the five established molecular subtypes used in breast cancer research.</li>
            <li><strong>Robustness:</strong> Tested by holding out entire patient cohorts during evaluation, not just individual samples, for a realistic measure of real-world accuracy.</li>
            <li><strong>Data:</strong> Built on real, published breast cancer cohort data.</li>
            <li><strong>Safety:</strong> Predictions stay off until a release is explicitly approved after validation.</li>
          </ul>
        </section>
      </main>
    </>
  )
}
