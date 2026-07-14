import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import './index.css'
import Layout from './components/Layout.jsx'
import App from './App.jsx'
import Infield from './pages/Infield.jsx'
import Integrated from './pages/Integrated.jsx'
import OaaRankings from './pages/OaaRankings.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<App />} />
          <Route path="/infield" element={<Infield />} />
          <Route path="/integrated" element={<Integrated />} />
          <Route path="/rankings" element={<OaaRankings />} />
          <Route path="/if-rankings" element={<Navigate to="/rankings" replace />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  </StrictMode>,
)
