import { useEffect, useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Chat      from './pages/Chat'
import Analyse   from './pages/Analyse'
import Compare   from './pages/Compare'
import Gaps      from './pages/Gaps'
import Sources   from './pages/Sources'
import Browse    from './pages/Browse'
import Stats     from './pages/Stats'
import IndexMgmt from './pages/IndexMgmt'
import { getHealth } from './api'

export default function App() {
  const [health, setHealth] = useState<any>(null)

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth({ ollama: false, n_chunks: 0, n_docs: 0 }))
    const t = setInterval(() => {
      getHealth()
        .then(setHealth)
        .catch(() => {})
    }, 15000)
    return () => clearInterval(t)
  }, [])

  return (
    <BrowserRouter>
      <div className="flex h-screen overflow-hidden">
        <Sidebar health={health} />
        <main className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/"        element={<Chat />} />
            <Route path="/analyse" element={<Analyse />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/gaps"    element={<Gaps />} />
            <Route path="/sources" element={<Sources />} />
            <Route path="/browse"  element={<Browse />} />
            <Route path="/stats"   element={<Stats />} />
            <Route path="/index"   element={<IndexMgmt />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
