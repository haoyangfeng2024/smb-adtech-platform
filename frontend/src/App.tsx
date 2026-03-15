import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Campaigns from './pages/Campaigns'
import CampaignCreate from './pages/CampaignCreate'
import Analytics from './pages/Analytics'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="campaigns" element={<Campaigns />} />
          <Route path="campaigns/new" element={<CampaignCreate />} />
          <Route path="analytics" element={<Analytics />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
