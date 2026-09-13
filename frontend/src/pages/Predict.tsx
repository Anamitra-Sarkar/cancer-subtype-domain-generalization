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
  const [nGenes, setNGenes] = useState<number | null>(null)
  const [exprText, setExprText] = useState('')
  const [result, setResult] = useState<PredictResult | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch(`${API}/readiness`).then(r => r.json()).then(setReadiness).catch(() => setReadiness({ ready: false, model_loaded: false, revision: null, error: 'API unreachable' }))
    fetch(`${API}/model-info`).then(r => r.json()).then(j => {
      if (j.n_genes) {
        setNGenes(j.n_genes)
        setExprText(Array.from({ length: j.n_genes }, (_, i) => (((i * 37) % 41) / 10 - 2).toFixed(1)).join(', '))
      }
    }).catch(() => {})
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
      <section className="card" aria-labelledby="predict-heading">
        <h2 id="predict-heading">Predict Subtype</h2>
        <p id="expr-help" style={{ fontSize: '0.8rem', color: '#475569', marginBottom: '0.6rem' }}>Paste one expression value per model gene{nGenes ? ` (${nGenes} values, comma-separated)` : ''}, or use the example values already filled in.</p>
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
          <button onClick={handlePredict} disabled={loading} aria-label="Predict cancer subtype" aria-busy={loading} aria-disabled={loading}>
            {loading ? 'Predicting…' : 'Predict Subtype'}
          </button>
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
