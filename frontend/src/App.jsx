import { useEffect, useState } from 'react'
import { fetchBatters, fetchTeams, fetchFielders, optimizePlot } from './api'
import GameStateForm from './components/GameStateForm'
import SearchSelect from './components/SearchSelect'

const POSITIONS = ['LF', 'CF', 'RF']

const displayName = (s) => (s && s.includes(', ')) ? s.split(', ').reverse().join(' ') : (s || '')

// OAA/100 機會（標準化速率，避免計次累積效果）
const oaaRate = (f) => {
  if (f.oaa === null || f.oaa === undefined || !f.n_opp) return null
  return (f.oaa / f.n_opp * 100).toFixed(1)
}

const fielderLabel = (f) => {
  const name = displayName(f.name)
  const rate = oaaRate(f)
  if (rate === null) return name
  const sign = rate >= 0 ? '+' : ''
  return `${name}  (模型 ${sign}${rate}/100)`
}

const EMPTY_FIELDERS = { LF: '', CF: '', RF: '' }

function buildFielders(sel) {
  const f = {}
  for (const p of POSITIONS) if (sel[p]) f[p] = sel[p]
  return Object.keys(f).length ? f : null
}

// 從 plotData.positions 取最具體的那組結果（custom > with_park > no_park > league_avg）
function getMainResult(data) {
  if (!data) return null
  const pos = data.positions
  return pos.custom || pos.with_park || pos.no_park || null
}

export default function App() {
  const [batters, setBatters]         = useState([])
  const [teams, setTeams]             = useState([])
  const [fielderOpts, setFielderOpts] = useState({ LF: [], CF: [], RF: [] })
  const [batterId, setBatterId]       = useState('')
  const [homeTeam, setHomeTeam]       = useState('')
  const [gameState, setGameState]     = useState({ on1b: 0, on2b: 0, on3b: 0, outs: 0 })

  // 一般模式
  const [selFielders, setSelFielders] = useState(EMPTY_FIELDERS)
  const [imgUrl, setImgUrl]           = useState(null)
  const [plotData, setPlotData]       = useState(null)

  // 比較模式
  const [minOpp, setMinOpp]           = useState(100)
  const [compareMode, setCompareMode] = useState(false)
  const [selFieldersB, setSelFieldersB] = useState(EMPTY_FIELDERS)
  const [imgUrlB, setImgUrlB]         = useState(null)
  const [plotDataB, setPlotDataB]     = useState(null)

  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState(null)

  useEffect(() => {
    fetchBatters().then(setBatters).catch(console.error)
    fetchTeams().then(setTeams).catch(console.error)
  }, [])

  useEffect(() => {
    fetchFielders(minOpp).then(setFielderOpts).catch(console.error)
  }, [minOpp])

  function toggleCompare() {
    setCompareMode(v => !v)
    setImgUrlB(null)
    setPlotDataB(null)
  }

  async function handleOptimize() {
    if (!batterId) return
    setLoading(true)
    setError(null)
    try {
      const base = {
        batterId: Number(batterId),
        on1b: gameState.on1b, on2b: gameState.on2b,
        on3b: gameState.on3b, outs: gameState.outs,
        homeTeam: homeTeam || null,
      }
      if (compareMode) {
        const [resA, resB] = await Promise.all([
          optimizePlot({ ...base, fielders: buildFielders(selFielders) }),
          optimizePlot({ ...base, fielders: buildFielders(selFieldersB) }),
        ])
        setImgUrl(prev  => { if (prev)  URL.revokeObjectURL(prev);  return resA.url })
        setImgUrlB(prev => { if (prev)  URL.revokeObjectURL(prev);  return resB.url })
        setPlotData(resA)
        setPlotDataB(resB)
      } else {
        const res = await optimizePlot({ ...base, fielders: buildFielders(selFielders) })
        setImgUrl(prev => { if (prev) URL.revokeObjectURL(prev); return res.url })
        setPlotData(res)
        setImgUrlB(null)
        setPlotDataB(null)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const fielderSection = (sel, setSel, label) => (
    <div>
      {label && <div style={{ fontSize: 10, color: '#6b7280', fontWeight: 700, marginBottom: 3 }}>{label}</div>}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {POSITIONS.map(p => (
          <div key={p} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 22, fontSize: 11, color: '#6b7280', fontWeight: 600 }}>{p}</span>
            <div style={{ flex: 1 }}>
              <SearchSelect
                options={[
                  { value: '', label: '聯盟平均' },
                  ...(fielderOpts[p] || []).map(f => ({ value: f.name, label: fielderLabel(f) })),
                ]}
                value={sel[p]}
                onChange={v => setSel(s => ({ ...s, [p]: v }))}
                placeholder="聯盟平均"
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )

  return (
    <div style={s.root}>
      <div style={s.body}>
        {/* ── 左側控制面板 ── */}
        <div style={s.panel}>
          <div style={s.panelHeader}>
            <div style={s.panelTitle}>⚾ Outfield Defense Optimizer</div>
          </div>

          <Sec title="打者">
            <SearchSelect
              options={batters.map(b => ({
                value: String(b.batter_id),
                label: `${displayName(b.name)}（${b.n_balls}）`,
              }))}
              value={batterId}
              onChange={setBatterId}
              placeholder="搜尋打者…"
            />
          </Sec>

          <Sec title="比賽狀況">
            <GameStateForm state={gameState} onChange={setGameState} />
          </Sec>

          <Sec title="球場（選填）">
            <select value={homeTeam} onChange={e => setHomeTeam(e.target.value)} style={s.select}>
              <option value="">— 通用 —</option>
              {teams.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </Sec>

          <Sec title="外野手">
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
              <span style={{ fontSize: 10, color: '#6b7280', whiteSpace: 'nowrap' }}>最低守備次數</span>
              <input
                type="range" min={0} max={400} step={25}
                value={minOpp}
                onChange={e => setMinOpp(Number(e.target.value))}
                style={{ flex: 1, accentColor: '#2563eb' }}
              />
              <span style={{ fontSize: 11, color: '#1e293b', minWidth: 28, textAlign: 'right' }}>{minOpp}</span>
            </div>
            <div style={{ fontSize: 9, color: '#94a3b8', marginBottom: 6, lineHeight: 1.5 }}>
              括號內為模型估計 OAA/100，基於賽季平均站位，非 Statcast 官方數值
            </div>
            <button
              onClick={toggleCompare}
              style={{ ...s.toggleBtn, background: compareMode ? '#7c3aed' : 'white', color: compareMode ? 'white' : '#374151' }}
            >
              {compareMode ? '✕ 關閉比較' : '⇔ 比較模式'}
            </button>
            {compareMode ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 8 }}>
                {fielderSection(selFielders,  setSelFielders,  '組合 A')}
                <div style={{ borderTop: '1px solid #e2e8f0' }} />
                {fielderSection(selFieldersB, setSelFieldersB, '組合 B')}
              </div>
            ) : (
              <div style={{ marginTop: 6 }}>
                {fielderSection(selFielders, setSelFielders, null)}
              </div>
            )}
          </Sec>

          <div style={s.panelFooter}>
            <button
              onClick={handleOptimize}
              disabled={!batterId || loading}
              style={{ ...s.btn, opacity: (!batterId || loading) ? 0.5 : 1 }}
            >
              {loading ? '計算中…' : '計算最佳站位'}
            </button>
            {error && <div style={s.error}>{error}</div>}
            <div style={s.hint}>圖由後端 matplotlib 繪製，與論文圖一致。</div>
          </div>
        </div>

        {/* ── 右側圖區 ── */}
        <div style={s.chartArea}>
          {compareMode && (imgUrl || imgUrlB) ? (
            <div style={{ width: '100%', maxWidth: 1400 }}>
              <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                <PlotBox imgUrl={imgUrl}   label="組合 A" loading={loading} />
                <PlotBox imgUrl={imgUrlB}  label="組合 B" loading={loading} />
              </div>
              {plotData && plotDataB && !loading && (
                <CompareStats dataA={plotData} dataB={plotDataB} />
              )}
            </div>
          ) : (
            <div style={{ width: '100%', maxWidth: 680 }}>
              <div style={{ position: 'relative' }}>
                {imgUrl
                  ? <img src={imgUrl} alt="defense plot"
                      style={{ width: '100%', display: 'block',
                               borderRadius: plotData ? '6px 6px 0 0' : 6,
                               boxShadow: '0 2px 12px rgba(0,0,0,0.15)' }} />
                  : <div style={s.placeholder}>選擇打者後按「計算最佳站位」</div>}
                {loading && <Overlay />}
              </div>
              {plotData && !loading && <StatsPanel data={plotData} />}
            </div>
          )}
        </div>
      </div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } } * { box-sizing: border-box; }`}</style>
    </div>
  )
}

function PlotBox({ imgUrl, label, loading }) {
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: '#475569',
                    textAlign: 'center', marginBottom: 4 }}>{label}</div>
      <div style={{ position: 'relative' }}>
        {imgUrl
          ? <img src={imgUrl} alt={label}
              style={{ width: '100%', display: 'block', borderRadius: 6,
                       boxShadow: '0 2px 12px rgba(0,0,0,0.15)' }} />
          : <div style={{ ...s.placeholder, minHeight: 200 }} />}
        {loading && <Overlay />}
      </div>
    </div>
  )
}

const fmtPos = (pos) => `(${Math.round(pos.x)}, ${Math.round(pos.y)})`

function CompareStats({ dataA, dataB }) {
  const rA = getMainResult(dataA)
  const rB = getMainResult(dataB)
  if (!rA || !rB) return null

  const dCatch = rA.catch_pct - rB.catch_pct
  const dRE    = rA.objective  - rB.objective

  const numRow = (label, valA, valB, delta, higherIsBetter = true) => {
    const better = higherIsBetter ? delta > 0 : delta < 0
    const color  = Math.abs(delta) < 0.01 ? '#475569' : (better ? '#16a34a' : '#dc2626')
    return (
      <tr key={label}>
        <td style={td.label}>{label}</td>
        <td style={td.val}>{valA}</td>
        <td style={td.val}>{valB}</td>
        <td style={{ ...td.val, color, fontWeight: 700 }}>{delta > 0 ? '+' : ''}{delta}</td>
      </tr>
    )
  }

  const posRow = (pos) => (
    <tr key={pos}>
      <td style={td.label}>{pos}</td>
      <td style={td.val}>{fmtPos(rA[pos])}</td>
      <td style={td.val}>{fmtPos(rB[pos])}</td>
      <td style={td.val}>—</td>
    </tr>
  )

  return (
    <div style={{ background: 'white', borderRadius: '0 0 6px 6px', padding: '10px 16px',
                  borderTop: '1px solid #e2e8f0', boxShadow: '0 2px 12px rgba(0,0,0,0.15)',
                  marginTop: -1 }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr>
            <th style={td.head} />
            <th style={td.head}>組合 A</th>
            <th style={td.head}>組合 B</th>
            <th style={td.head}>A − B</th>
          </tr>
        </thead>
        <tbody>
          {numRow('Catch %', rA.catch_pct.toFixed(1) + '%', rB.catch_pct.toFixed(1) + '%', dCatch.toFixed(1), true)}
          {numRow('RE24',    rA.objective.toFixed(2),        rB.objective.toFixed(2),        dRE.toFixed(2),   false)}
          <tr><td colSpan={4} style={{ padding: '4px 0' }}><hr style={{ border: 'none', borderTop: '1px solid #f1f5f9', margin: 0 }} /></td></tr>
          {['LF', 'CF', 'RF'].map(posRow)}
        </tbody>
      </table>
    </div>
  )
}

const td = {
  label: { padding: '4px 8px', color: '#64748b', fontWeight: 600, textAlign: 'left' },
  val:   { padding: '4px 12px', textAlign: 'center', color: '#1e293b' },
  head:  { padding: '4px 12px', textAlign: 'center', fontSize: 11,
           color: '#6b7280', fontWeight: 600, borderBottom: '1px solid #e2e8f0' },
}

function StatsPanel({ data }) {
  const { positions, stats } = data
  const park = stats.home_team || ''
  const entries = []
  if ('custom' in positions)    entries.push({ label: `Selected${park ? ` @ ${park}` : ''}`, key: 'custom' })
  else {
    if ('league_avg' in positions) entries.push({ label: 'League Avg', key: 'league_avg' })
    if ('with_park' in positions)  entries.push({ label: `RE24 Opt (park=${park})`, key: 'with_park' })
    else if ('no_park' in positions) entries.push({ label: 'RE24 Opt (no park)', key: 'no_park' })
  }
  let delta = null
  if ('league_avg' in positions) {
    const ref = positions.with_park || positions.no_park
    if (ref) delta = positions.league_avg.objective - ref.objective
  }
  return (
    <div style={{ background: 'white', borderRadius: '0 0 6px 6px', padding: '10px 16px',
                  borderTop: '1px solid #e2e8f0', boxShadow: '0 2px 12px rgba(0,0,0,0.15)' }}>
      {/* Catch % / RE24 */}
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'stretch', marginBottom: 10 }}>
        {entries.map(({ label, key }) => {
          const ps = positions[key]
          return (
            <div key={key} style={{ background: '#f8fafc', border: '1px solid #e2e8f0',
                                     borderRadius: 6, padding: '6px 14px', minWidth: 140 }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: '#475569', marginBottom: 4 }}>{label}</div>
              <div style={{ fontSize: 12, color: '#1e293b' }}>catch <strong>{ps.catch_pct.toFixed(1)}%</strong></div>
              <div style={{ fontSize: 12, color: '#1e293b' }}>RE24 <strong>{ps.objective.toFixed(2)}</strong></div>
            </div>
          )
        })}
        {delta !== null && (
          <div style={{ background: delta > 0 ? '#f0fdf4' : '#fef2f2',
                        border: `1px solid ${delta > 0 ? '#bbf7d0' : '#fecaca'}`,
                        borderRadius: 6, padding: '6px 14px', minWidth: 90,
                        display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: delta > 0 ? '#166534' : '#991b1b', marginBottom: 2 }}>Δ RE24</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: delta > 0 ? '#16a34a' : '#dc2626' }}>
              {delta > 0 ? '+' : ''}{delta.toFixed(2)}
            </div>
          </div>
        )}
      </div>
      {/* 站位座標 */}
      <div style={{ borderTop: '1px solid #f1f5f9', paddingTop: 8 }}>
        <table style={{ borderCollapse: 'collapse', fontSize: 11 }}>
          <thead>
            <tr>
              <th style={spc.th} />
              {entries.map(({ label, key }) => <th key={key} style={spc.th}>{label}</th>)}
            </tr>
          </thead>
          <tbody>
            {['LF', 'CF', 'RF'].map(p => (
              <tr key={p}>
                <td style={{ ...spc.td, fontWeight: 700, color: '#374151' }}>{p}</td>
                {entries.map(({ key }) => {
                  const pos = positions[key][p]
                  return (
                    <td key={key} style={spc.td}>
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
  th: { padding: '2px 14px', textAlign: 'center', fontSize: 10, fontWeight: 600,
        color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' },
  td: { padding: '3px 14px', textAlign: 'center', color: '#475569' },
}

function Overlay() {
  return (
    <div style={s.overlay}>
      <div style={s.spinner} />
      <p style={{ color: 'white', marginTop: 10, fontSize: 13 }}>最佳化計算中…</p>
    </div>
  )
}

function Sec({ title, children }) {
  return (
    <section style={{ padding: '10px 14px', borderTop: '1px solid #f1f5f9' }}>
      <div style={{ fontSize: 10, fontWeight: 700, color: '#94a3b8',
        textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 6 }}>
        {title}
      </div>
      {children}
    </section>
  )
}

const s = {
  root: { minHeight: '100vh', background: '#f8fafc', fontFamily: "'Inter', system-ui, sans-serif" },
  body: { display: 'flex', minHeight: '100vh', alignItems: 'flex-start' },
  panel: {
    width: 260, minWidth: 240, background: 'white', color: '#1e293b',
    display: 'flex', flexDirection: 'column',
    minHeight: '100vh', borderRight: '1px solid #e2e8f0',
    overflowY: 'auto',
  },
  panelHeader: {
    padding: '14px 14px 12px',
    borderBottom: '1px solid #f1f5f9',
  },
  panelTitle: { fontSize: 13, fontWeight: 700, color: '#1e293b' },
  panelFooter: {
    padding: '10px 14px 14px',
    borderTop: '1px solid #f1f5f9',
    marginTop: 'auto',
    display: 'flex', flexDirection: 'column', gap: 8,
  },
  select: {
    width: '100%', padding: '5px 6px', background: '#f8fafc',
    color: '#1e293b', border: '1px solid #d1d5db', borderRadius: 5, fontSize: 11,
  },
  toggleBtn: {
    width: '100%', padding: '5px 0', color: '#374151',
    border: '1px solid #d1d5db', borderRadius: 5, fontSize: 11, fontWeight: 600, cursor: 'pointer',
  },
  btn: {
    width: '100%', padding: '7px 0', background: '#2563eb', color: 'white',
    border: 'none', borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: 'pointer',
  },
  error: {
    background: '#fef2f2', border: '1px solid #fca5a5',
    borderRadius: 5, padding: '5px 8px', fontSize: 10, color: '#dc2626',
  },
  hint: { fontSize: 10, color: '#9ca3af', lineHeight: 1.5 },
  chartArea: {
    flex: 1, padding: '16px', display: 'flex', justifyContent: 'center',
    alignItems: 'flex-start', background: '#f8fafc',
  },
  placeholder: {
    padding: '60px 20px', textAlign: 'center', color: '#9ca3af',
    background: 'white', borderRadius: 6, fontSize: 13, border: '1px solid #e2e8f0',
  },
  overlay: {
    position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.55)',
    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
    borderRadius: 6,
  },
  spinner: {
    width: 28, height: 28, border: '3px solid #555',
    borderTop: '3px solid #3b82f6', borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
  },
}
