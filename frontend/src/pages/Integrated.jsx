import { useEffect, useState } from 'react'
import { fetchFielders, fetchIfFielderOptions, fetchIntegratedBatters, fetchIfYears,
         fetchTeams, optimizeIntegrated } from '../api'
import GameStateForm from '../components/GameStateForm'
import SearchSelect from '../components/SearchSelect'
import IntegratedChart from '../components/IntegratedChart'

const displayName = (s) => (s && s.includes(', ')) ? s.split(', ').reverse().join(' ') : (s || '')
const OF_POSITIONS = ['LF', 'CF', 'RF']
const IF_POSITIONS = ['1B', '2B', '3B', 'SS']
const ALL_POSITIONS = [...OF_POSITIONS, ...IF_POSITIONS]
const EMPTY_OF = { LF: '', CF: '', RF: '' }
const EMPTY_IF = { '1B': '', '2B': '', '3B': '', SS: '' }

// 外野選單（/api/fielders）：oaa/n_opp；內野選單（/api/if_fielder_options）：oaa/n_balls
const fielderLabel = (f, nKey) => {
  const name = displayName(f.name)
  const n = f[nKey]
  if (f.oaa === null || f.oaa === undefined || !n) return name
  const rate = (f.oaa / n * 100).toFixed(1)
  return `${name}  (${rate >= 0 ? '+' : ''}${rate}/100)`
}

// 外野後端吃球員名（load_player_params 的鍵）、內野吃 player_id
const buildOf = (sel) => {
  const f = {}
  for (const p of OF_POSITIONS) if (sel[p]) f[p] = sel[p]
  return Object.keys(f).length ? f : null
}
const buildIf = (sel) => {
  const f = {}
  for (const p of IF_POSITIONS) if (sel[p]) f[p] = Number(sel[p])
  return Object.keys(f).length ? f : null
}

export default function Integrated() {
  const [availYears, setAvailYears] = useState([])
  const [year, setYear]             = useState(null)
  const [batters, setBatters]       = useState([])
  const [batterId, setBatterId]     = useState('')
  const [teams, setTeams]           = useState([])
  const [gameState, setGameState]   = useState({ on1b: 0, on2b: 0, on3b: 0, outs: 0 })
  const [ofOpts, setOfOpts]         = useState(null)   // pos → options
  const [ifOpts, setIfOpts]         = useState(null)

  const [homeTeam, setHomeTeam]     = useState('')
  const [selOf, setSelOf]           = useState(EMPTY_OF)
  const [selIf, setSelIf]           = useState(EMPTY_IF)
  const [data, setData]             = useState(null)

  const [compareMode, setCompareMode] = useState(false)
  const [homeTeamB, setHomeTeamB]     = useState('')
  const [selOfB, setSelOfB]           = useState(EMPTY_OF)
  const [selIfB, setSelIfB]           = useState(EMPTY_IF)
  const [dataB, setDataB]             = useState(null)

  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)

  useEffect(() => {
    fetchIfYears().then(ys => {
      setAvailYears(ys)
      if (ys.length) setYear(ys[ys.length - 1])
    }).catch(console.error)
    fetchTeams().then(setTeams).catch(console.error)
  }, [])

  useEffect(() => {
    if (year === null) return
    fetchIntegratedBatters(year).then(data => { setBatters(data); setBatterId('') }).catch(console.error)
    fetchFielders(100, year).then(setOfOpts).catch(() => setOfOpts(null))
    fetchIfFielderOptions(year).then(setIfOpts).catch(() => setIfOpts(null))
  }, [year])

  // 換年份時清掉野手選擇：先前選的球員在新年份可能沒有模型參數
  useEffect(() => {
    setSelOf(EMPTY_OF); setSelIf(EMPTY_IF)
    setSelOfB(EMPTY_OF); setSelIfB(EMPTY_IF)
  }, [year])

  function toggleCompare() {
    setCompareMode(v => !v)
    setHomeTeamB('')
    setDataB(null)
  }

  async function handleOptimize() {
    if (!batterId) return
    setLoading(true)
    setError(null)
    try {
      const base = {
        batterId: Number(batterId), year,
        on1b: gameState.on1b, on2b: gameState.on2b,
        on3b: gameState.on3b, outs: gameState.outs,
      }
      if (compareMode) {
        const [resA, resB] = await Promise.all([
          optimizeIntegrated({ ...base, homeTeam,
            ofFielders: buildOf(selOf), ifFielders: buildIf(selIf) }),
          optimizeIntegrated({ ...base, homeTeam: homeTeamB,
            ofFielders: buildOf(selOfB), ifFielders: buildIf(selIfB) }),
        ])
        setData(resA)
        setDataB(resB)
      } else {
        setData(await optimizeIntegrated({ ...base, homeTeam,
          ofFielders: buildOf(selOf), ifFielders: buildIf(selIf) }))
        setDataB(null)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  // 七人野手選單（外野在上、內野在下）
  const fielderSection = (of, setOf, ifSel, setIf, label) => (
    <div>
      {label && (
        <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--slate-400)',
                      textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
          {label}
        </div>
      )}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
        {OF_POSITIONS.map(p => (
          <div key={p} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 24, fontSize: 11, color: 'var(--slate-500)', fontWeight: 700,
                           flexShrink: 0 }}>{p}</span>
            <div style={{ flex: 1 }}>
              <SearchSelect
                options={[
                  { value: '', label: '聯盟平均' },
                  ...(ofOpts?.[p] || []).map(f => ({ value: f.name, label: fielderLabel(f, 'n_opp') })),
                ]}
                value={of[p]}
                onChange={v => setOf(s => ({ ...s, [p]: v }))}
                placeholder="聯盟平均"
              />
            </div>
          </div>
        ))}
        {IF_POSITIONS.map(p => (
          <div key={p} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 24, fontSize: 11, color: 'var(--slate-500)', fontWeight: 700,
                           flexShrink: 0 }}>{p}</span>
            <div style={{ flex: 1 }}>
              <SearchSelect
                options={[
                  { value: '', label: '聯盟平均' },
                  ...(ifOpts?.[p] || []).filter(f => (f.n_balls || 0) >= 100)
                    .map(f => ({ value: String(f.player_id), label: fielderLabel(f, 'n_balls') })),
                ]}
                value={ifSel[p]}
                onChange={v => setIf(s => ({ ...s, [p]: v }))}
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
                label: `${displayName(b.name)}（${b.n_total}）`,
              }))}
              value={batterId}
              onChange={setBatterId}
              placeholder="搜尋打者…"
            />
            <div style={{ fontSize: 9, color: '#cbd5e1', marginTop: 8, lineHeight: 1.6 }}>
              括號內為該年場內球數。內外野七人一起排：飛球交給外野、滾地球交給內野
            </div>
          </Sec>

          <Sec title="比賽狀況">
            <GameStateForm state={gameState} onChange={setGameState} />
          </Sec>

          <Sec title={compareMode ? '球場（各組獨立）' : '球場'}>
            {compareMode ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                {[['A', homeTeam, setHomeTeam], ['B', homeTeamB, setHomeTeamB]].map(([lbl, val, set]) => (
                  <div key={lbl} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: 10, fontWeight: 700, color: 'var(--slate-400)',
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
            <div style={{ fontSize: 9, color: '#cbd5e1', marginTop: 6, lineHeight: 1.6 }}>
              指定球場會把打到牆的球視為必失分，外野站位跟著調整（計算較久）
            </div>
          </Sec>

          <Sec title="野手">
            <div style={{ fontSize: 9, color: '#cbd5e1', marginBottom: 10, lineHeight: 1.6 }}>
              括號內為模型估計 OAA/100，非 Statcast 官方數值。指定野手時以該球員的守備參數微調站位
            </div>
            {compareMode ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {fielderSection(selOf,  setSelOf,  selIf,  setSelIf,  '組合 A')}
                <div style={{ borderTop: '1px solid var(--slate-100)' }} />
                {fielderSection(selOfB, setSelOfB, selIfB, setSelIfB, '組合 B')}
              </div>
            ) : (
              fielderSection(selOf, setSelOf, selIf, setSelIf, null)
            )}
          </Sec>

          <div style={s.panelFooter}>
            <button
              onClick={toggleCompare}
              style={{ ...s.compareBtn,
                background: compareMode ? '#ede9fe' : 'white',
                color:      compareMode ? '#6d28d9' : 'var(--slate-500)',
                border:     `1px solid ${compareMode ? '#c4b5fd' : 'var(--slate-200)'}`,
              }}
            >
              {compareMode ? '✕ 關閉比較模式' : '⇔ 比較模式'}
            </button>
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
          {compareMode && (data || dataB) ? (
            <div style={{ width: '100%', maxWidth: 1700 }}>
              <div className="compare-row" style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                <ChartBox data={data}  park={homeTeam}  label="組合 A" loading={loading} />
                <ChartBox data={dataB} park={homeTeamB} label="組合 B" loading={loading} />
              </div>
              {data && dataB && !loading && <CompareStats dataA={data} dataB={dataB} />}
            </div>
          ) : (
            <div style={{ width: '100%', maxWidth: 1020 }}>
              <div style={{ position: 'relative' }}>
                {data ? (
                  <>
                    <div style={{ borderRadius: '8px 8px 0 0', boxShadow: '0 4px 20px rgba(0,0,0,0.12)',
                                  overflow: 'hidden' }}>
                      <TitleBar data={data} park={homeTeam} />
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
            border-bottom: 1px solid var(--slate-200);
          }
          .app-chart-area { min-height: 0 !important; padding: 16px !important; }
          .compare-row { flex-direction: column; }
        }
      `}</style>
    </div>
  )
}

function TitleBar({ data, park }) {
  const picked = ALL_POSITIONS.filter(p => data.fielders && data.fielders[p])
  return (
    <div style={{ background: 'white', padding: '12px 18px 0' }}>
      <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--slate-800)' }}>
        {displayName(data.name)}（{data.year}, {data.stand}打）{park ? ` @ ${park}` : ''}
      </div>
      <div style={{ fontSize: 11, color: 'var(--slate-400)', marginTop: 2 }}>
        壘況 {data.situation}・外野 {data.stats.n_of_balls} 球＋滾地 {data.stats.n_gb} 球
        {data.stats.n_popups > 0 && `＋高飛 ${data.stats.n_popups} 球（展示）`}
      </div>
      {picked.length > 0 && (
        <div style={{ fontSize: 11, color: 'var(--blue-600)', fontWeight: 600, marginTop: 2 }}>
          {picked.map(p => `${p} ${displayName(data.fielders[p])}`).join('・')}
        </div>
      )}
    </div>
  )
}

function ChartBox({ data, park, label, loading }) {
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--slate-500)',
                    textAlign: 'center', marginBottom: 6, letterSpacing: '0.04em',
                    textTransform: 'uppercase' }}>{label}</div>
      <div style={{ position: 'relative' }}>
        {data ? (
          <div style={{ borderRadius: 8, boxShadow: '0 4px 20px rgba(0,0,0,0.12)', overflow: 'hidden' }}>
            <TitleBar data={data} park={park} />
            <IntegratedChart data={data} />
          </div>
        ) : (
          <div style={{ background: 'white', border: '1px solid var(--slate-200)',
                        borderRadius: 8, minHeight: 200 }} />
        )}
        {loading && <Overlay />}
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

const fmtPos = (pos) => `(${Math.round(pos.x)}, ${Math.round(pos.y)})`

function CompareStats({ dataA, dataB }) {
  const sA = dataA.stats
  const sB = dataB.stats

  const numRow = (label, valA, valB, delta) => {
    const color = Math.abs(delta) < 0.01 ? 'var(--slate-600)'
      : (delta > 0 ? 'var(--green-600)' : 'var(--red-600)')
    return (
      <tr key={label}>
        <td style={td.label}>{label}</td>
        <td style={td.val}>{valA}</td>
        <td style={td.val}>{valB}</td>
        <td style={{ ...td.val, color, fontWeight: 700 }}>{delta > 0 ? '+' : ''}{delta.toFixed(2)}</td>
      </tr>
    )
  }

  const posRow = (pos) => {
    const pA = dataA.optimized.positions[pos]
    const pB = dataB.optimized.positions[pos]
    const dx = Math.round(pA.x - pB.x)
    const dy = Math.round(pA.y - pB.y)
    const sign = (v) => v > 0 ? `+${v}` : `${v}`
    const color = (dx === 0 && dy === 0) ? 'var(--slate-600)' : 'var(--blue-600)'
    return (
      <tr key={pos}>
        <td style={td.label}>{pos}</td>
        <td style={td.val}>{fmtPos(pA)}</td>
        <td style={td.val}>{fmtPos(pB)}</td>
        <td style={{ ...td.val, color, fontWeight: 600 }}>
          {dx === 0 && dy === 0 ? '—' : `(${sign(dx)}, ${sign(dy)})`}
        </td>
      </tr>
    )
  }

  return (
    <div style={{ background: 'white', borderRadius: 8, padding: '12px 18px', marginTop: 12,
                  boxShadow: '0 4px 20px rgba(0,0,0,0.12)' }}>
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
          {numRow('總省分', sA.runs_saved_total.toFixed(2), sB.runs_saved_total.toFixed(2),
                  sA.runs_saved_total - sB.runs_saved_total)}
          {numRow('外野省分', sA.runs_saved_of.toFixed(2), sB.runs_saved_of.toFixed(2),
                  sA.runs_saved_of - sB.runs_saved_of)}
          {numRow('內野省分', sA.runs_saved_if.toFixed(2), sB.runs_saved_if.toFixed(2),
                  sA.runs_saved_if - sB.runs_saved_if)}
          <tr><td colSpan={4} style={{ padding: '4px 0' }}>
            <hr style={{ border: 'none', borderTop: '1px solid var(--slate-100)', margin: 0 }} />
          </td></tr>
          {ALL_POSITIONS.map(posRow)}
        </tbody>
      </table>
    </div>
  )
}

const td = {
  label: { padding: '4px 8px', color: 'var(--slate-500)', fontWeight: 600, textAlign: 'left' },
  val:   { padding: '4px 14px', textAlign: 'center', color: 'var(--slate-800)' },
  head:  { padding: '4px 14px', textAlign: 'center', fontSize: 11,
           color: 'var(--slate-400)', fontWeight: 600, borderBottom: '1px solid var(--slate-200)' },
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
  select: {
    width: '100%', padding: '6px 8px', fontSize: 12, borderRadius: 6,
    border: '1px solid var(--slate-200)', background: 'white', color: 'var(--slate-800)',
  },
  compareBtn: {
    width: '100%', padding: '6px 0', borderRadius: 6,
    fontSize: 11, fontWeight: 600, cursor: 'pointer',
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
