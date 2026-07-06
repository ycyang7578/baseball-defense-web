import { useEffect, useState } from 'react'
import { fetchBatters, fetchTeams, fetchFielders, fetchYears, optimizePlot } from './api'
import GameStateForm from './components/GameStateForm'
import SearchSelect from './components/SearchSelect'
import SprayChart from './components/SprayChart'

const POSITIONS = ['LF', 'CF', 'RF']

const displayName = (s) => (s && s.includes(', ')) ? s.split(', ').reverse().join(' ') : (s || '')

const oaaRate = (f) => {
  if (f.oaa === null || f.oaa === undefined || !f.n_opp) return null
  return (f.oaa / f.n_opp * 100).toFixed(1)
}

const fielderLabel = (f) => {
  const name = displayName(f.name)
  const rate = oaaRate(f)
  if (rate === null) return name
  const sign = rate >= 0 ? '+' : ''
  return `${name}  (${sign}${rate}/100)`
}

const EMPTY_FIELDERS = { LF: '', CF: '', RF: '' }

function buildFielders(sel) {
  const f = {}
  for (const p of POSITIONS) if (sel[p]) f[p] = sel[p]
  return Object.keys(f).length ? f : null
}

function getMainResult(data) {
  if (!data) return null
  const pos = data.positions
  return pos.custom || pos.with_park || pos.no_park || null
}

export default function App() {
  const [availYears, setAvailYears]   = useState([2025])
  const [year, setYear]               = useState(2025)
  const [batters, setBatters]         = useState([])
  const [teams, setTeams]             = useState([])
  const [fielderOpts, setFielderOpts] = useState({ LF: [], CF: [], RF: [] })
  const [batterId, setBatterId]       = useState('')
  const [homeTeam, setHomeTeam]       = useState('')
  const [gameState, setGameState]     = useState({ on1b: 0, on2b: 0, on3b: 0, outs: 0 })

  const [selFielders, setSelFielders]   = useState(EMPTY_FIELDERS)
  const [imgUrl, setImgUrl]             = useState(null)
  const [plotData, setPlotData]         = useState(null)

  const [minOpp, setMinOpp]             = useState(100)
  const [compareMode, setCompareMode]   = useState(false)
  const [homeTeamB, setHomeTeamB]       = useState('')
  const [selFieldersB, setSelFieldersB] = useState(EMPTY_FIELDERS)
  const [imgUrlB, setImgUrlB]           = useState(null)
  const [plotDataB, setPlotDataB]       = useState(null)

  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [showSpray, setShowSpray] = useState(false)

  useEffect(() => {
    fetchYears().then(ys => { setAvailYears(ys); setYear(ys[ys.length - 1]) }).catch(console.error)
    fetchTeams().then(setTeams).catch(console.error)
  }, [])

  useEffect(() => {
    fetchBatters(year).then(data => { setBatters(data); setBatterId('') }).catch(console.error)
    fetchFielders(minOpp, year).then(setFielderOpts).catch(console.error)
  }, [year, minOpp])

  // 切換年份時，先前選的守備員可能在新年份沒有模型參數（例如新秀球員），
  // 清掉避免送出「舊年份選的球員 + 新年份」這種無效組合給後端
  useEffect(() => {
    setSelFielders(EMPTY_FIELDERS)
    setSelFieldersB(EMPTY_FIELDERS)
  }, [year])

  function toggleCompare() {
    setCompareMode(v => !v)
    setHomeTeamB('')
    setImgUrlB(null)
    setPlotDataB(null)
    setShowSpray(false)
  }

  async function handleOptimize() {
    if (!batterId) return
    setLoading(true)
    setError(null)
    setShowSpray(false)
    try {
      const base = {
        batterId: Number(batterId),
        on1b: gameState.on1b, on2b: gameState.on2b,
        on3b: gameState.on3b, outs: gameState.outs,
      }
      if (compareMode) {
        const [resA, resB] = await Promise.all([
          optimizePlot({ ...base, year, homeTeam: homeTeam || null, fielders: buildFielders(selFielders) }),
          optimizePlot({ ...base, year, homeTeam: homeTeamB || null, fielders: buildFielders(selFieldersB) }),
        ])
        setImgUrl(prev  => { if (prev) URL.revokeObjectURL(prev); return resA.url })
        setImgUrlB(prev => { if (prev) URL.revokeObjectURL(prev); return resB.url })
        setPlotData(resA)
        setPlotDataB(resB)
      } else {
        const res = await optimizePlot({ ...base, year, homeTeam: homeTeam || null, fielders: buildFielders(selFielders) })
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
      {label && (
        <div style={{ fontSize: 10, fontWeight: 700, color: '#94a3b8',
                      textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
          {label}
        </div>
      )}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
        {POSITIONS.map(p => (
          <div key={p} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 24, fontSize: 11, color: '#64748b', fontWeight: 700,
                           flexShrink: 0 }}>{p}</span>
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
      <div className="app-body" style={s.body}>

        {/* ── 左側控制面板 ── */}
        <div className="app-panel" style={s.panel}>

          {/* 年份 */}
          <div style={s.panelHeader}>
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
              {availYears.map(y => (
                <button key={y} onClick={() => setYear(y)} style={{
                  ...s.yearBtn,
                  background: year === y ? '#2563eb' : '#f1f5f9',
                  color:      year === y ? 'white'   : '#475569',
                  border:     year === y ? '1px solid #2563eb' : '1px solid transparent',
                }}>{y}</button>
              ))}
            </div>
          </div>

          {/* 打者 */}
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

          {/* 比賽狀況 */}
          <Sec title="比賽狀況">
            <GameStateForm state={gameState} onChange={setGameState} />
          </Sec>

          {/* 球場 */}
          <Sec title={compareMode ? '球場（各組獨立）' : '球場'}>
            {compareMode ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                {[['A', homeTeam, setHomeTeam], ['B', homeTeamB, setHomeTeamB]].map(([lbl, val, set]) => (
                  <div key={lbl} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: 10, fontWeight: 700, color: '#94a3b8',
                                   minWidth: 14 }}>{lbl}</span>
                    <select value={val} onChange={e => set(e.target.value)} style={{ ...s.select, flex: 1 }}>
                      <option value="">— 通用 —</option>
                      {teams.map(t => <option key={t} value={t}>{t}</option>)}
                    </select>
                  </div>
                ))}
              </div>
            ) : (
              <select value={homeTeam} onChange={e => setHomeTeam(e.target.value)} style={s.select}>
                <option value="">— 通用 —</option>
                {teams.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
            )}
          </Sec>

          {/* 外野手 */}
          <Sec title="外野手">
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
              <span style={{ fontSize: 10, color: '#94a3b8', whiteSpace: 'nowrap' }}>最低守備次數</span>
              <input
                type="range" min={0} max={400} step={25}
                value={minOpp}
                onChange={e => setMinOpp(Number(e.target.value))}
                style={{ flex: 1, accentColor: '#2563eb' }}
              />
              <span style={{ fontSize: 11, color: '#475569', minWidth: 28,
                             textAlign: 'right', fontWeight: 600 }}>{minOpp}</span>
            </div>
            <div style={{ fontSize: 9, color: '#cbd5e1', marginBottom: 10, lineHeight: 1.6 }}>
              括號內為模型估計 OAA/100，非 Statcast 官方數值
            </div>
            {compareMode ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {fielderSection(selFielders,  setSelFielders,  '組合 A')}
                <div style={{ borderTop: '1px solid #f1f5f9' }} />
                {fielderSection(selFieldersB, setSelFieldersB, '組合 B')}
              </div>
            ) : (
              fielderSection(selFielders, setSelFielders, null)
            )}
          </Sec>

          {/* Footer */}
          <div style={s.panelFooter}>
            <button
              onClick={toggleCompare}
              style={{ ...s.compareBtn,
                background: compareMode ? '#ede9fe' : 'white',
                color:      compareMode ? '#6d28d9' : '#64748b',
                border:     `1px solid ${compareMode ? '#c4b5fd' : '#e2e8f0'}`,
              }}
            >
              {compareMode ? '✕ 關閉比較模式' : '⇔ 比較模式'}
            </button>
            <button
              onClick={handleOptimize}
              disabled={!batterId || loading}
              style={{ ...s.btn, opacity: (!batterId || loading) ? 0.5 : 1 }}
            >
              {loading ? '計算中…' : '計算最佳站位'}
            </button>
            {error && <div style={s.error}>{error}</div>}
          </div>
        </div>

        {/* ── 右側結果區 ── */}
        <div className="app-chart-area" style={s.chartArea}>
          {compareMode && (imgUrl || imgUrlB) ? (
            /* ── 比較模式 ── */
            <div style={{ width: '100%', maxWidth: 1400 }}>
              {!loading && (imgUrl || imgUrlB) && (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                              marginBottom: 10, flexWrap: 'wrap', gap: 8 }}>
                  <ChartToggle value={showSpray} onChange={setShowSpray} />
                  <div style={{ display: 'flex', gap: 8 }}>
                    {imgUrl  && <DownloadBtn href={imgUrl}  name={`defense_A_${batterId}_${year}.png`} label="↓ A" />}
                    {imgUrlB && <DownloadBtn href={imgUrlB} name={`defense_B_${batterId}_${year}.png`} label="↓ B" />}
                  </div>
                </div>
              )}
              <div className="compare-row" style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                <PlotBox imgUrl={imgUrl}  plotData={plotData}  label="組合 A" loading={loading} showSpray={showSpray} />
                <PlotBox imgUrl={imgUrlB} plotData={plotDataB} label="組合 B" loading={loading} showSpray={showSpray} />
              </div>
              {plotData && plotDataB && !loading && (
                <CompareStats dataA={plotData} dataB={plotDataB} />
              )}
            </div>
          ) : (
            /* ── 單張模式 ── */
            <div style={{ width: '100%', maxWidth: 700 }}>
              {imgUrl && !loading && (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                              marginBottom: 8 }}>
                  <ChartToggle value={showSpray} onChange={setShowSpray} />
                  <DownloadBtn href={imgUrl} name={`defense_${batterId}_${year}.png`} label="↓ 下載論文圖" />
                </div>
              )}
              <div style={{ position: 'relative' }}>
                {imgUrl ? (
                  <FieldChart imgUrl={imgUrl} data={plotData} showSpray={showSpray}
                    radius={plotData ? '8px 8px 0 0' : '8px'} />
                ) : (
                  <EmptyState />
                )}
                {loading && <Overlay />}
              </div>
              {plotData && !loading && <StatsPanel data={plotData} />}
            </div>
          )}
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
            border-bottom: 1px solid #e2e8f0;
          }
          .app-chart-area { min-height: 0 !important; padding: 16px !important; }
          .compare-row { flex-direction: column; }
        }
      `}</style>
    </div>
  )
}

function EmptyState() {
  return (
    <div style={{ background: 'white', borderRadius: 8, border: '1px solid #e2e8f0',
                  padding: '64px 32px', textAlign: 'center',
                  boxShadow: '0 1px 4px rgba(0,0,0,0.04)' }}>
      <div style={{ fontSize: 15, fontWeight: 600, color: '#334155', marginBottom: 8 }}>
        選擇打者開始分析
      </div>
      <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.7, maxWidth: 280, margin: '0 auto' }}>
        系統依據打者的飛球傾向，以 RE24 為目標函數計算最佳外野站位
      </div>
    </div>
  )
}

/* ── Shared UI helpers ──────────────────────────────── */
function ChartToggle({ value, onChange }) {
  return (
    <div style={{ display: 'flex', background: '#e2e8f0', borderRadius: 7, padding: 3, gap: 2 }}>
      {[{ key: false, label: '落點密度圖' }, { key: true, label: '互動圖' }].map(({ key, label }) => (
        <button key={String(key)} onClick={() => onChange(key)} style={{
          padding: '4px 14px', fontSize: 11, fontWeight: 600, cursor: 'pointer',
          border: 'none', borderRadius: 5,
          background: value === key ? 'white' : 'transparent',
          color:      value === key ? '#1e293b' : '#64748b',
          boxShadow:  value === key ? '0 1px 3px rgba(0,0,0,0.12)' : 'none',
        }}>{label}</button>
      ))}
    </div>
  )
}

function DownloadBtn({ href, name, label = '↓ 下載論文圖' }) {
  return (
    <a href={href} download={name}
      style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 6,
               padding: '4px 12px', fontSize: 11, fontWeight: 600,
               color: '#374151', textDecoration: 'none', whiteSpace: 'nowrap' }}>
      {label}
    </a>
  )
}

function FieldChart({ imgUrl, data, showSpray, radius = '8px' }) {
  return showSpray ? (
    <div style={{ borderRadius: radius, boxShadow: '0 4px 20px rgba(0,0,0,0.12)', overflow: 'hidden' }}>
      <SprayChart
        balls={data?.balls} positions={data?.positions} parkBoundary={data?.parkBoundary}
        title={data?.title} situation={data?.situation} stats={data?.stats} />
    </div>
  ) : (
    <img src={imgUrl} alt="defense plot"
      style={{ width: '100%', display: 'block', borderRadius: radius,
               boxShadow: '0 4px 20px rgba(0,0,0,0.12)' }} />
  )
}

function PlotBox({ imgUrl, plotData, label, loading, showSpray }) {
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b',
                    textAlign: 'center', marginBottom: 6, letterSpacing: '0.04em',
                    textTransform: 'uppercase' }}>{label}</div>
      <div style={{ position: 'relative' }}>
        {imgUrl
          ? <FieldChart imgUrl={imgUrl} data={plotData} showSpray={showSpray} />
          : <div style={{ background: 'white', border: '1px solid #e2e8f0',
                          borderRadius: 8, minHeight: 200 }} />}
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

  const posRow = (pos) => {
    const dx = Math.round(rA[pos].x - rB[pos].x)
    const dy = Math.round(rA[pos].y - rB[pos].y)
    const sign = (v) => v > 0 ? `+${v}` : `${v}`
    const color = (dx === 0 && dy === 0) ? '#475569' : '#2563eb'
    return (
      <tr key={pos}>
        <td style={td.label}>{pos}</td>
        <td style={td.val}>{fmtPos(rA[pos])}</td>
        <td style={td.val}>{fmtPos(rB[pos])}</td>
        <td style={{ ...td.val, color, fontWeight: 600 }}>
          {dx === 0 && dy === 0 ? '—' : `(${sign(dx)}, ${sign(dy)})`}
        </td>
      </tr>
    )
  }

  return (
    <div style={{ background: 'white', borderRadius: '0 0 8px 8px', padding: '12px 18px',
                  borderTop: '1px solid #e2e8f0', boxShadow: '0 4px 20px rgba(0,0,0,0.12)',
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
          <tr><td colSpan={4} style={{ padding: '4px 0' }}>
            <hr style={{ border: 'none', borderTop: '1px solid #f1f5f9', margin: 0 }} />
          </td></tr>
          {['LF', 'CF', 'RF'].map(posRow)}
        </tbody>
      </table>
    </div>
  )
}

const td = {
  label: { padding: '4px 8px', color: '#64748b', fontWeight: 600, textAlign: 'left' },
  val:   { padding: '4px 14px', textAlign: 'center', color: '#1e293b' },
  head:  { padding: '4px 14px', textAlign: 'center', fontSize: 11,
           color: '#94a3b8', fontWeight: 600, borderBottom: '1px solid #e2e8f0' },
}

function StatsPanel({ data }) {
  const { positions, stats } = data
  const park = stats.home_team || ''
  const entries = []
  if ('custom' in positions)    entries.push({ label: `Selected${park ? ` @ ${park}` : ''}`, key: 'custom' })
  else {
    if ('league_avg' in positions) entries.push({ label: 'League Avg', key: 'league_avg' })
    if ('with_park' in positions)  entries.push({ label: `RE24 Opt (${park})`, key: 'with_park' })
    else if ('no_park' in positions) entries.push({ label: 'RE24 Opt', key: 'no_park' })
  }
  let delta = null
  if ('league_avg' in positions) {
    const ref = positions.with_park || positions.no_park
    if (ref) delta = positions.league_avg.objective - ref.objective
  }
  return (
    <div style={{ background: 'white', borderRadius: '0 0 8px 8px', padding: '12px 18px',
                  borderTop: '1px solid #e2e8f0', boxShadow: '0 4px 20px rgba(0,0,0,0.12)' }}>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'stretch', marginBottom: 12 }}>
        {entries.map(({ label, key }) => {
          const ps = positions[key]
          return (
            <div key={key} style={{ background: '#f8fafc', border: '1px solid #e2e8f0',
                                     borderRadius: 7, padding: '8px 16px', minWidth: 140 }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: '#64748b',
                            textTransform: 'uppercase', letterSpacing: '0.05em',
                            marginBottom: 5 }}>{label}</div>
              <div style={{ fontSize: 12, color: '#334155', marginBottom: 2 }}>
                Catch <strong style={{ fontSize: 14 }}>{ps.catch_pct.toFixed(1)}%</strong>
              </div>
              <div style={{ fontSize: 12, color: '#334155' }}>
                RE24 <strong style={{ fontSize: 14 }}>{ps.objective.toFixed(2)}</strong>
              </div>
            </div>
          )
        })}
        {delta !== null && (
          <div style={{ background: delta > 0 ? '#f0fdf4' : '#fef2f2',
                        border: `1px solid ${delta > 0 ? '#bbf7d0' : '#fecaca'}`,
                        borderRadius: 7, padding: '8px 16px', minWidth: 90,
                        display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.05em',
                          color: delta > 0 ? '#166534' : '#991b1b', marginBottom: 4 }}>Δ RE24</div>
            <div style={{ fontSize: 20, fontWeight: 700,
                          color: delta > 0 ? '#16a34a' : '#dc2626' }}>
              {delta > 0 ? '+' : ''}{delta.toFixed(2)}
            </div>
          </div>
        )}
      </div>
      <div style={{ borderTop: '1px solid #f1f5f9', paddingTop: 10 }}>
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
                <td style={{ ...spc.td, fontWeight: 700, color: '#334155' }}>{p}</td>
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
  th: { padding: '2px 16px', textAlign: 'center', fontSize: 10, fontWeight: 600,
        color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' },
  td: { padding: '4px 16px', textAlign: 'center', color: '#475569' },
}

function Overlay() {
  return (
    <div style={s.overlay}>
      <div style={s.spinner} />
      <p style={{ color: 'white', marginTop: 12, fontSize: 13, fontWeight: 500 }}>最佳化計算中…</p>
    </div>
  )
}

function Sec({ title, children }) {
  return (
    <section style={{ padding: '12px 16px', borderTop: '1px solid #f1f5f9' }}>
      <div style={{ fontSize: 10, fontWeight: 700, color: '#94a3b8',
                    textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8 }}>
        {title}
      </div>
      {children}
    </section>
  )
}

const s = {
  root: { minHeight: '100vh', background: '#f1f5f9', fontFamily: "'Inter', system-ui, sans-serif" },
  body: { display: 'flex', minHeight: '100vh', alignItems: 'flex-start' },
  panel: {
    width: 280, minWidth: 260, background: 'white', color: '#1e293b',
    display: 'flex', flexDirection: 'column',
    minHeight: '100vh', borderRight: '1px solid #e2e8f0',
    overflowY: 'auto', flexShrink: 0,
  },
  panelHeader: {
    padding: '18px 16px 14px',
    borderBottom: '1px solid #f1f5f9',
  },
  panelTitle: {
    fontSize: 14, fontWeight: 700, color: '#0f172a', letterSpacing: '-0.01em',
  },
  yearBtn: {
    padding: '3px 10px', fontSize: 12, fontWeight: 600, cursor: 'pointer',
    borderRadius: 5, transition: 'all 0.15s',
  },
  panelFooter: {
    padding: '12px 16px 18px',
    borderTop: '1px solid #f1f5f9',
    display: 'flex', flexDirection: 'column', gap: 8,
  },
  select: {
    width: '100%', padding: '6px 8px', background: '#f8fafc',
    color: '#334155', border: '1px solid #e2e8f0', borderRadius: 6, fontSize: 11,
  },
  compareBtn: {
    width: '100%', padding: '6px 0', borderRadius: 6,
    fontSize: 11, fontWeight: 600, cursor: 'pointer',
  },
  btn: {
    width: '100%', padding: '9px 0', background: '#2563eb', color: 'white',
    border: 'none', borderRadius: 7, fontSize: 13, fontWeight: 600, cursor: 'pointer',
    letterSpacing: '0.01em',
  },
  error: {
    background: '#fef2f2', border: '1px solid #fca5a5',
    borderRadius: 6, padding: '6px 10px', fontSize: 11, color: '#dc2626',
  },
  chartArea: {
    flex: 1, padding: '20px', display: 'flex', justifyContent: 'center',
    alignItems: 'flex-start', background: '#f1f5f9', minHeight: '100vh',
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
