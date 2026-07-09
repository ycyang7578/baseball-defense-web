import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import Layout from './components/Layout.jsx'
import App from './App.jsx'
import Infield from './pages/Infield.jsx'
import Rankings from './pages/Rankings.jsx'
import InfieldRankings from './pages/InfieldRankings.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<App />} />
          <Route path="/infield" element={<Infield />} />
          <Route path="/rankings" element={<Rankings />} />
          <Route path="/if-rankings" element={<InfieldRankings />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  </StrictMode>,
)
