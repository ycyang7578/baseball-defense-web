import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import './index.css'
import Layout from './components/Layout.jsx'
import Integrated from './pages/Integrated.jsx'
import OaaRankings from './pages/OaaRankings.jsx'
import { LanguageProvider } from './i18n/LanguageContext.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <LanguageProvider>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Integrated />} />
            <Route path="/integrated" element={<Navigate to="/" replace />} />
            <Route path="/infield" element={<Navigate to="/" replace />} />
            <Route path="/rankings" element={<OaaRankings />} />
            <Route path="/if-rankings" element={<Navigate to="/rankings" replace />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </LanguageProvider>
  </StrictMode>,
)
