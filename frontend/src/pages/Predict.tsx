import { useEffect, useState } from 'react'
import { API, type Readiness } from '../App'

type PredictResult = { subtype: string; confidence: number; probabilities: Record<string, number> }

function pillClass(subtype: string): string {
  if (subtype.includes('Luminal A')) return 'pill-lumA'
  if (subtype.includes('Luminal B')) return 'pill-lumB'
  if (subtype.includes('HER2')) return 'pill-her2'
  if (subtype.includes('Basal')) return 'pill-basal'
  return 'pill-normal'
}

export default function Predict() {
  const [readiness, setReadiness] = useState<Readiness | null>(null)
  const [exprText, setExprText] = useState('0.5, -0.2, 1.1, -0.8, 0.3, 2.1, -1.0, 0.7, 0.4, -0.3, 1.5, -0.6, 0.9, 0.2, -1.2, 0.8, -0.4, 0.6, 1.0, -0.9')
  const [result, setResult] = useState<PredictResult | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch(`${API}/readiness`).then(r => r.json()).then(setReadiness).catch(() => setReadiness({ ready: false, model_loaded: false, revision: null, error: 'API unreachable' }))
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

  return (
    <main id="main-content" className="container" tabIndex={-1} style={{ paddingTop: '2rem' }}>
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
    </main>
  )
}
