import { useEffect, useState } from 'react'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import Overview from './pages/Overview'
import Predict from './pages/Predict'

export type Readiness = { ready: boolean; model_loaded: boolean; revision: string | null; error: string | null }

const API = ((import.meta as unknown as { env: Record<string, string | undefined> }).env.VITE_API_URL)?.replace(/\/$/, '') ?? ''
export { API }

function Layout({ children }: { children: React.ReactNode }) {
  const [readiness, setReadiness] = useState<Readiness | null>(null)

  useEffect(() => {
    fetch(`${API}/readiness`).then(r => r.json()).then(setReadiness).catch(() => setReadiness({ ready: false, model_loaded: false, revision: null, error: 'API unreachable' }))
  }, [])

  return (
    <>
      <a href="#main-content" className="skip-link">Skip to main content</a>
      <nav className="navbar">
        <div className="brand">
          <span className="brand-mark">BC</span>
          <span className="brand-name">Subtype Atlas</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.2rem' }}>
          <NavLink to="/" end className={({ isActive }) => `nav-link${isActive ? ' nav-link-active' : ''}`}>Overview</NavLink>
          <NavLink to="/predict" className={({ isActive }) => `nav-link${isActive ? ' nav-link-active' : ''}`}>Predict</NavLink>
          {readiness ? (
            <span className={`badge ${readiness.ready ? 'badge-ready' : 'badge-notready'}`} role="status" aria-live="polite">
              {readiness.ready ? 'Live predictions' : 'Preview mode'}
            </span>
          ) : (
            <span className="badge badge-notready" role="status" aria-live="polite" aria-busy="true">Checking…</span>
          )}
        </div>
      </nav>
      {children}
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/predict" element={<Predict />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}
