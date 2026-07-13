import { useEffect, useState } from 'react'
import { fetchIfBatters, fetchIfYears, optimizeIntegrated } from '../api'
import GameStateForm from '../components/GameStateForm'
import SearchSelect from '../components/SearchSelect'
import IntegratedChart from '../components/IntegratedChart'

const displayName = (s) => (s && s.includes(', ')) ? s.split(', ').reverse().join(' ') : (s || '')
const ALL_POSITIONS = ['LF', 'CF', 'RF', '1B', '2B', '3B', 'SS']

export default function Integrated() {
  const [availYears, setAvailYears] = useState([])
  const [year, setYear]             = useState(null)
  const [batters, setBatters]       = useState([])
  const [batterId, setBatterId]     = useState('')
  const [gameState, setGameState]   = useState({ on1b: 0, on2b: 0, on3b: 0, outs: 0 })
  const [data, setData]             = useState(null)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)

  useEffect(() => {
    fetchIfYears().then(ys => {
      setAvailYears(ys)
      if (ys.length) setYear(ys[ys.length - 1])
    }).catch(console.error)
  }, [])

  useEffect(() => {
    if (year === null) return
    fetchIfBatters(year).then(data => { setBatters(data); setBatterId('') }).catch(console.error)
  }, [year])

  async function handleOptimize() {
    if (!batterId) return
    setLoading(true)
    setError(null)
    try {
      setData(await optimizeIntegrated({
        batterId: Number(batterId), year,
        on1b: gameState.on1b, on2b: gameState.on2b,
        on3b: gameState.on3b, outs: gameState.outs,
      }))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={s.root}>
      <div className="app-body" style={s.body}>

        {/* ── 左側控制面板 ── */}
        <div className="app-panel" style={s.panel}>
          <div style={s.panelHeader}>
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
              {availYears.map(y => (
                <button key={y} onClick={() => setYear(y)} style={{
                  ...s.yearBtn,
                  background: year === y ? 'var(--blue-600)' : 'var(--slate-100)',
                  color:      year === y ? 'white'   : 'var(--slate-600)',
                  border:     year === y ? '1px solid var(--blue-600)' : '1px solid transparent',
                }}>{y}</button>
              ))}
            </div>
          </div>

          <Sec title="打者">
            <SearchSelect
              options={batters.map(b => ({
                value: String(b.batter_id),
                label: `${displayName(b.name)}（${b.n_gb}）`,
              }))}
              value={batterId}
              onChange={setBatterId}
              placeholder="搜尋打者…"
            />
            <div style={{ fontSize: 9, color: '#cbd5e1', marginTop: 8, lineHeight: 1.6 }}>
              括號內為該年滾地球數。內外野七人一起排：飛球交給外野、滾地球交給內野
            </div>
          </Sec>

          <Sec title="比賽狀況">
            <GameStateForm state={gameState} onChange={setGameState} />
          </Sec>

          <div style={s.panelFooter}>
            <button
              onClick={handleOptimize}
              disabled={!batterId || loading}
              style={{ ...s.btn, opacity: (!batterId || loading) ? 0.5 : 1 }}
            >
              {loading ? '計算中…' : '計算七人最佳站位'}
            </button>
            {error && <div style={s.error}>{error}</div>}
          </div>
        </div>

        {/* ── 右側結果區 ── */}
        <div className="app-chart-area" style={s.chartArea}>
          <div style={{ width: '100%', maxWidth: 760 }}>
            <div style={{ position: 'relative' }}>
              {data ? (
                <>
                  <div style={{ borderRadius: '8px 8px 0 0', boxShadow: '0 4px 20px rgba(0,0,0,0.12)',
                                overflow: 'hidden' }}>
                    <TitleBar data={data} />
                    <IntegratedChart data={data} />
                  </div>
                  <StatsPanel data={data} />
                </>
              ) : (
                <EmptyState />
              )}
              {loading && <Overlay />}
            </div>
          </div>
        </div>
      </div>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        * { box-sizing: border-box; }
        @media (max-width: 768px) {
          .app-body { flex-direction: column; }
          .app-panel {
            width: 100% !important;
            min-width: 0 !important;
            min-height: 0 !important;
            border-right: none !important;
            border-bottom: 1px solid var(--slate-200);
          }
          .app-chart-area { min-height: 0 !important; padding: 16px !important; }
        }
      `}</style>
    </div>
  )
}

function TitleBar({ data }) {
  return (
    <div style={{ background: 'white', padding: '12px 18px 0' }}>
      <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--slate-800)' }}>
        {displayName(data.name)}（{data.year}, {data.stand}打）
      </div>
      <div style={{ fontSize: 11, color: 'var(--slate-400)', marginTop: 2 }}>
        壘況 {data.situation}・外野 {data.stats.n_of_balls} 球＋滾地 {data.stats.n_gb} 球
      </div>
    </div>
  )
}

function EmptyState() {
  return (
    <div style={{ background: 'white', borderRadius: 8, border: '1px solid var(--slate-200)',
                  padding: '64px 32px', textAlign: 'center',
                  boxShadow: '0 1px 4px rgba(0,0,0,0.04)' }}>
      <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--slate-700)', marginBottom: 8 }}>
        選擇打者與壘況開始分析
      </div>
      <div style={{ fontSize: 12, color: 'var(--slate-400)', lineHeight: 1.7, maxWidth: 320, margin: '0 auto' }}>
        七名野手一起排：外野三人對付飛球、內野四人對付滾地球，
        以「預期失分」同一把尺衡量，加總就是這套站位替球隊省下的分數
      </div>
    </div>
  )
}

function StatsPanel({ data }) {
  const { league, optimized, stats } = data
  const saved = stats.runs_saved_total
  const card = (label, savedVal, note) => (
    <div style={{ background: 'var(--slate-50)', border: '1px solid var(--slate-200)',
                  borderRadius: 7, padding: '8px 16px', minWidth: 130 }}>
      <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--slate-500)',
                    textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 5 }}>{label}</div>
      <div style={{ fontSize: 14, fontWeight: 700,
                    color: savedVal > 0 ? 'var(--green-600)' : 'var(--red-600)' }}>
        {savedVal > 0 ? '+' : ''}{savedVal.toFixed(1)} 分
      </div>
      <div style={{ fontSize: 10, color: 'var(--slate-500)', marginTop: 2 }}>{note}</div>
    </div>
  )
  return (
    <div style={{ background: 'white', borderRadius: '0 0 8px 8px', padding: '12px 18px',
                  borderTop: '1px solid var(--slate-200)', boxShadow: '0 4px 20px rgba(0,0,0,0.12)' }}>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'stretch', marginBottom: 12 }}>
        <div style={{ background: saved > 0 ? '#f0fdf4' : '#fef2f2',
                      border: `1px solid ${saved > 0 ? '#bbf7d0' : '#fecaca'}`,
                      borderRadius: 7, padding: '8px 16px', minWidth: 150,
                      display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.05em',
                        color: saved > 0 ? '#166534' : '#991b1b', marginBottom: 4 }}>
            總省分（vs 聯盟平均）
          </div>
          <div style={{ fontSize: 22, fontWeight: 700,
                        color: saved > 0 ? 'var(--green-600)' : 'var(--red-600)' }}>
            {saved > 0 ? '+' : ''}{saved.toFixed(1)} 分
          </div>
          <div style={{ fontSize: 10, color: 'var(--slate-500)', marginTop: 2 }}>
            以他該季全部擊球估計
          </div>
        </div>
        {card('外野三人', stats.runs_saved_of, `${stats.n_of_balls} 顆外野球`)}
        {card('內野四人', stats.runs_saved_if, `${stats.n_gb} 顆滾地球`)}
      </div>
      <div style={{ borderTop: '1px solid var(--slate-100)', paddingTop: 10 }}>
        <table style={{ borderCollapse: 'collapse', fontSize: 11 }}>
          <thead>
            <tr>
              <th style={spc.th} />
              <th style={spc.th}>聯盟平均</th>
              <th style={spc.th}>最佳化</th>
            </tr>
          </thead>
          <tbody>
            {ALL_POSITIONS.map(p => (
              <tr key={p}>
                <td style={{ ...spc.td, fontWeight: 700, color: 'var(--slate-700)' }}>{p}</td>
                {[league, optimized].map((set, i) => {
                  const pos = set.positions[p]
                  return (
                    <td key={i} style={spc.td}>
                      ({Math.round(pos.x)}, {Math.round(pos.y)}) ft
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const spc = {
  th: { padding: '2px 16px', textAlign: 'center', fontSize: 10, fontWeight: 600,
        color: 'var(--slate-400)', textTransform: 'uppercase', letterSpacing: '0.05em' },
  td: { padding: '4px 16px', textAlign: 'center', color: 'var(--slate-600)' },
}

function Overlay() {
  return (
    <div style={s.overlay}>
      <div style={s.spinner} />
      <p style={{ color: 'white', marginTop: 12, fontSize: 13, fontWeight: 500 }}>
        七人站位計算中（約需一分鐘）…
      </p>
    </div>
  )
}

function Sec({ title, children }) {
  return (
    <section style={{ padding: '12px 16px', borderTop: '1px solid var(--slate-100)' }}>
      <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--slate-400)',
                    textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8 }}>
        {title}
      </div>
      {children}
    </section>
  )
}

const s = {
  root: { minHeight: '100vh', background: 'var(--slate-100)', fontFamily: "'Inter', system-ui, sans-serif" },
  body: { display: 'flex', minHeight: '100vh', alignItems: 'flex-start' },
  panel: {
    width: 280, minWidth: 260, background: 'white', color: 'var(--slate-800)',
    display: 'flex', flexDirection: 'column',
    minHeight: '100vh', borderRight: '1px solid var(--slate-200)',
    overflowY: 'auto', flexShrink: 0,
  },
  panelHeader: {
    padding: '18px 16px 14px',
    borderBottom: '1px solid var(--slate-100)',
  },
  yearBtn: {
    padding: '3px 10px', fontSize: 12, fontWeight: 600, cursor: 'pointer',
    borderRadius: 5, transition: 'all 0.15s',
  },
  panelFooter: {
    padding: '12px 16px 18px',
    borderTop: '1px solid var(--slate-100)',
    display: 'flex', flexDirection: 'column', gap: 8,
  },
  btn: {
    width: '100%', padding: '9px 0', background: 'var(--blue-600)', color: 'white',
    border: 'none', borderRadius: 7, fontSize: 13, fontWeight: 600, cursor: 'pointer',
    letterSpacing: '0.01em',
  },
  error: {
    background: '#fef2f2', border: '1px solid #fca5a5',
    borderRadius: 6, padding: '6px 10px', fontSize: 11, color: 'var(--red-600)',
  },
  chartArea: {
    flex: 1, padding: '20px', display: 'flex', justifyContent: 'center',
    alignItems: 'flex-start', background: 'var(--slate-100)', minHeight: '100vh',
  },
  overlay: {
    position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.5)',
    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
    borderRadius: 8,
  },
  spinner: {
    width: 30, height: 30, border: '3px solid rgba(255,255,255,0.3)',
    borderTop: '3px solid white', borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
  },
}
