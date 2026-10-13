import { Route, Routes } from 'react-router-dom'

import HealthBanner from './components/HealthBanner'
import NavBar from './components/NavBar'
import CaptionPage from './pages/CaptionPage'
import DashboardPage from './pages/DashboardPage'
import EvaluationPage from './pages/EvaluationPage'
import HistoryPage from './pages/HistoryPage'

export default function App() {
  return (
    <div className="min-h-dvh">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-accent focus:px-4 focus:py-2 focus:text-white"
      >
        Skip to main content
      </a>
      <NavBar />
      <HealthBanner />
      <main id="main" className="mx-auto max-w-6xl px-4 py-8">
        <Routes>
          <Route path="/" element={<CaptionPage />} />
          <Route path="/caption" element={<CaptionPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
          <Route
            path="*"
            element={
              <p className="card px-6 py-12 text-center text-fg-muted">
                Page not found. Use the navigation above.
              </p>
            }
          />
        </Routes>
      </main>
    </div>
  )
}