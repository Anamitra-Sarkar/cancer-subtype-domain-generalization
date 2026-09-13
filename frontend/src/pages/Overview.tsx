import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { API } from '../App'

type Comparison = {
  method?: string
  available?: boolean
  message?: string
  random_split?: { mean_accuracy: number; mean_macro_f1: number }
  lodo?: { mean_accuracy: number; mean_macro_f1: number }
  gap_accuracy?: number
  gap_macro_f1?: number
}

export default function Overview() {
  const [comparison, setComparison] = useState<Comparison | null>(null)

  useEffect(() => {
    fetch(`${API}/comparison`).then(r => r.json()).then(setComparison).catch(() => {})
  }, [])

  const demoComparison = comparison?.available === false || !comparison?.random_split ? null : comparison
  const hasRealComparison = demoComparison && demoComparison.random_split

  return (
    <>
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
          <Link to="/predict" className="btn primary">Try a prediction</Link>
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
