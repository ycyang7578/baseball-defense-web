import { useEffect, useState } from 'react'
import { fetchIfBatters, fetchIfFielderOptions, fetchIfYears, fetchTeams, ifOptimize } from '../api'
import GameStateForm from '../components/GameStateForm'
import SearchSelect from '../components/SearchSelect'
import InfieldChart from '../components/InfieldChart'

const displayName = (s) => (s && s.includes(', ')) ? s.split(', ').reverse().join(' ') : (s || '')
const IF_POSITIONS = ['1B', '2B', '3B', 'SS']
const EMPTY_FIELDERS = { '1B': '', '2B': '', '3B': '', SS: '' }

const oaaRate = (f) => {
  if (f.oaa === null || f.oaa === undefined || !f.n_balls) return null
  return (f.oaa / f.n_balls * 100).toFixed(1)
}

const fielderLabel = (f) => {
  const name = displayName(f.name)
  const rate = oaaRate(f)
  if (rate === null) return name
  const sign = rate >= 0 ? '+' : ''
  return `${name}  (${sign}${rate}/100)`
}

function buildFielders(sel) {
  const f = {}
  for (const p of IF_POSITIONS) if (sel[p]) f[p] = Number(sel[p])
  return Object.keys(f).length ? f : null
}

export default function Infield() {
  const [availYears, setAvailYears] = useState([])
  const [year, setYear]             = useState(null)
  const [batters, setBatters]       = useState([])
  const [batterId, setBatterId]     = useState('')
  const [teams, setTeams]           = useState([])
  const [gameState, setGameState]   = useState({ on1b: 0, on2b: 0, on3b: 0, outs: 0 })
  const [fielderOpts, setFielderOpts] = useState(null)   // pos → options（null=功能未啟用）
  const [minBalls, setMinBalls]       = useState(100)

  const [homeTeam, setHomeTeam]       = useState('')
  const [selFielders, setSelFielders] = useState(EMPTY_FIELDERS)
  const [data, setData]               = useState(null)

  const [compareMode, setCompareMode]   = useState(false)
  const [homeTeamB, setHomeTeamB]       = useState('')
  const [selFieldersB, setSelFieldersB] = useState(EMPTY_FIELDERS)
  const [dataB, setDataB]               = useState(null)

  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [colorBy, setColorBy] = useState('optimized')
  const [probMin, setProbMin] = useState(0)
  const [probMax, setProbMax] = useState(1)

  useEffect(() => {
    fetchIfYears().then(ys => {
      setAvailYears(ys)
      if (ys.length) setYear(ys[ys.length - 1])
    }).catch(console.error)
    fetchTeams().then(setTeams).catch(console.error)
  }, [])

  useEffect(() => {
    if (year === null) return
    fetchIfBatters(year).then(data => { setBatters(data); setBatterId('') }).catch(console.error)
    fetchIfFielderOptions(year).then(setFielderOpts).catch(() => setFielderOpts(null))
  }, [year])

  // 切換年份時，先前選的野手可能在新年份沒有站位資料，清掉避免無效組合
  useEffect(() => {
    setSelFielders(EMPTY_FIELDERS)
    setSelFieldersB(EMPTY_FIELDERS)
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
          ifOptimize({ ...base, fielders: buildFielders(selFielders) }),
          ifOptimize({ ...base, fielders: buildFielders(selFieldersB) }),
        ])
        setData(resA)
        setDataB(resB)
      } else {
        setData(await ifOptimize({ ...base, fielders: buildFielders(selFielders) }))
        setDataB(null)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const filteredOpts = (pos) =>
    (fielderOpts?.[pos] || []).filter(f => (f.n_balls || 0) >= minBalls)

  const fielderSection = (sel, setSel, label) => (
    <div>
      {label && (
        <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--slate-400)',
                      textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
          {label}
        </div>
      )}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
        {IF_POSITIONS.map(p => (
          <div key={p} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 24, fontSize: 11, color: 'var(--slate-500)', fontWeight: 700,
                           flexShrink: 0 }}>{p}</span>
            <div style={{ flex: 1 }}>
              <SearchSelect
                options={[
                  { value: '', label: '聯盟平均' },
                  ...filteredOpts(p).map(f => ({ value: String(f.player_id), label: fielderLabel(f) })),
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
                  background: year === y ? 'var(--blue-600)' : 'var(--slate-100)',
                  color:      year === y ? 'white'   : 'var(--slate-600)',
                  border:     year === y ? '1px solid var(--blue-600)' : '1px solid transparent',
                }}>{y}</button>
              ))}
            </div>
          </div>

          {/* 打者 */}
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
              括號內為該年滾地球數。站位限制依禁趨位規則：二壘兩側各兩人、不可換邊、站在內野土上
            </div>
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
              內野場地為標準規格，站位不受球場影響（僅顯示於標題）
            </div>
          </Sec>

          {/* 內野手 */}
          <Sec title="內野手">
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
              <span style={{ fontSize: 10, color: 'var(--slate-400)', whiteSpace: 'nowrap' }}>最低守備次數</span>
              <input
                type="range" min={0} max={400} step={25}
                value={minBalls}
                onChange={e => setMinBalls(Number(e.target.value))}
                style={{ flex: 1, accentColor: 'var(--blue-600)' }}
              />
              <span style={{ fontSize: 11, color: 'var(--slate-600)', minWidth: 28,
                             textAlign: 'right', fontWeight: 600 }}>{minBalls}</span>
            </div>
            <div style={{ fontSize: 9, color: '#cbd5e1', marginBottom: 10, lineHeight: 1.6 }}>
              括號內為模型估計 OAA/100，非 Statcast 官方數值。指定野手時以該球員的守備參數微調站位
            </div>
            {fielderOpts === null ? (
              <div style={{ fontSize: 10, color: 'var(--slate-400)' }}>野手選單未啟用</div>
            ) : compareMode ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {fielderSection(selFielders,  setSelFielders,  '組合 A')}
                <div style={{ borderTop: '1px solid var(--slate-100)' }} />
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
              {loading ? '計算中…' : '計算最佳站位'}
            </button>
            {error && <div style={s.error}>{error}</div>}
          </div>
        </div>

        {/* ── 右側結果區 ── */}
        <div className="app-chart-area" style={s.chartArea}>
          {compareMode && (data || dataB) ? (
            /* ── 比較模式 ── */
            <div style={{ width: '100%', maxWidth: 1400 }}>
              {!loading && (data || dataB) && (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                              marginBottom: 10, flexWrap: 'wrap', gap: 8 }}>
                  <ColorToggle value={colorBy} onChange={setColorBy} />
                  <ProbRange min={probMin} max={probMax} onMin={setProbMin} onMax={setProbMax} />
                </div>
              )}
              <div className="compare-row" style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                <ChartBox data={data}  park={homeTeam}  label="組合 A" loading={loading}
                  colorBy={colorBy} probMin={probMin} probMax={probMax} />
                <ChartBox data={dataB} park={homeTeamB} label="組合 B" loading={loading}
                  colorBy={colorBy} probMin={probMin} probMax={probMax} />
              </div>
              {data && dataB && !loading && <CompareStats dataA={data} dataB={dataB} />}
            </div>
          ) : (
            /* ── 單張模式 ── */
            <div style={{ width: '100%', maxWidth: 700 }}>
              {data && !loading && (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                              marginBottom: 8, flexWrap: 'wrap', gap: 8 }}>
                  <ColorToggle value={colorBy} onChange={setColorBy} />
                  <ProbRange min={probMin} max={probMax} onMin={setProbMin} onMax={setProbMax} />
                </div>
              )}
              <div style={{ position: 'relative' }}>
                {data ? (
                  <>
                    <div style={{ borderRadius: '8px 8px 0 0', boxShadow: '0 4px 20px rgba(0,0,0,0.12)',
                                  overflow: 'hidden' }}>
                      <TitleBar data={data} park={homeTeam} />
                      <InfieldChart data={data} colorBy={colorBy} probMin={probMin} probMax={probMax} />
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
  const picked = IF_POSITIONS.filter(p => data.fielders && data.fielders[p])
  return (
    <div style={{ background: 'white', padding: '12px 18px 0' }}>
      <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--slate-800)' }}>
        {displayName(data.name)}（{data.year}, {data.stand}打）{park ? ` @ ${park}` : ''}
      </div>
      <div style={{ fontSize: 11, color: 'var(--slate-400)', marginTop: 2 }}>
        壘況 {data.situation}・{data.stats.n_gb} 顆滾地球
        {picked.length > 0 && (
          <span style={{ color: 'var(--blue-600)', fontWeight: 600 }}>
            {'　'}
            {picked.map(p => `${p} ${displayName(data.fielders[p])}`).join('・')}
          </span>
        )}
      </div>
    </div>
  )
}

function ChartBox({ data, park, label, loading, colorBy, probMin, probMax }) {
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--slate-500)',
                    textAlign: 'center', marginBottom: 6, letterSpacing: '0.04em',
                    textTransform: 'uppercase' }}>{label}</div>
      <div style={{ position: 'relative' }}>
        {data ? (
          <div style={{ borderRadius: 8, boxShadow: '0 4px 20px rgba(0,0,0,0.12)', overflow: 'hidden' }}>
            <TitleBar data={data} park={park} />
            <InfieldChart data={data} colorBy={colorBy} probMin={probMin} probMax={probMax} />
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
        選擇打者開始分析
      </div>
      <div style={{ fontSize: 12, color: 'var(--slate-400)', lineHeight: 1.7, maxWidth: 300, margin: '0 auto' }}>
        系統依據打者的滾地球傾向與壘況，在禁趨位規則的限制下計算內野四人的最佳站位
      </div>
    </div>
  )
}

function ColorToggle({ value, onChange }) {
  return (
    <div style={{ display: 'flex', background: 'var(--slate-200)', borderRadius: 7, padding: 3, gap: 2 }}>
      {[{ key: 'league', label: '平均站位上色' }, { key: 'optimized', label: '最佳化上色' }].map(({ key, label }) => (
        <button key={key} onClick={() => onChange(key)} style={{
          padding: '4px 14px', fontSize: 11, fontWeight: 600, cursor: 'pointer',
          border: 'none', borderRadius: 5,
          background: value === key ? 'white' : 'transparent',
          color:      value === key ? 'var(--slate-800)' : 'var(--slate-500)',
          boxShadow:  value === key ? '0 1px 3px rgba(0,0,0,0.12)' : 'none',
        }}>{label}</button>
      ))}
    </div>
  )
}

function ProbRange({ min, max, onMin, onMax }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10, color: 'var(--slate-500)' }}>
      <span>P(out)</span>
      <input type="range" min={0} max={1} step={0.05} value={min}
        onChange={e => onMin(Math.min(Number(e.target.value), max))}
        style={{ width: 70, accentColor: 'var(--blue-600)' }} />
      <span style={{ minWidth: 58, textAlign: 'center', fontWeight: 600, color: 'var(--slate-600)' }}>
        {(min * 100).toFixed(0)}–{(max * 100).toFixed(0)}%
      </span>
      <input type="range" min={0} max={1} step={0.05} value={max}
        onChange={e => onMax(Math.max(Number(e.target.value), min))}
        style={{ width: 70, accentColor: 'var(--blue-600)' }} />
    </div>
  )
}

function StatsPanel({ data }) {
  const { league, optimized, stats } = data
  const saved = stats.runs_saved
  return (
    <div style={{ background: 'white', borderRadius: '0 0 8px 8px', padding: '12px 18px',
                  borderTop: '1px solid var(--slate-200)', boxShadow: '0 4px 20px rgba(0,0,0,0.12)' }}>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'stretch', marginBottom: 12 }}>
        {[{ label: '聯盟平均', set: league }, { label: '最佳化', set: optimized }].map(({ label, set }) => (
          <div key={label} style={{ background: 'var(--slate-50)', border: '1px solid var(--slate-200)',
                                    borderRadius: 7, padding: '8px 16px', minWidth: 140 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--slate-500)',
                          textTransform: 'uppercase', letterSpacing: '0.05em',
                          marginBottom: 5 }}>{label}</div>
            <div style={{ fontSize: 12, color: 'var(--slate-700)', marginBottom: 2 }}>
              出局率 <strong style={{ fontSize: 14 }}>{(set.exp_outs * 100).toFixed(1)}%</strong>
            </div>
            <div style={{ fontSize: 12, color: 'var(--slate-700)' }}>
              預期失分 <strong style={{ fontSize: 14 }}>{set.runs.toFixed(1)}</strong>
            </div>
          </div>
        ))}
        <div style={{ background: saved > 0 ? '#f0fdf4' : '#fef2f2',
                      border: `1px solid ${saved > 0 ? '#bbf7d0' : '#fecaca'}`,
                      borderRadius: 7, padding: '8px 16px', minWidth: 120,
                      display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.05em',
                        color: saved > 0 ? '#166534' : '#991b1b', marginBottom: 4 }}>省分</div>
          <div style={{ fontSize: 20, fontWeight: 700,
                        color: saved > 0 ? 'var(--green-600)' : 'var(--red-600)' }}>
            {saved > 0 ? '+' : ''}{saved.toFixed(1)} 分
          </div>
          <div style={{ fontSize: 10, color: 'var(--slate-500)', marginTop: 2 }}>
            每 450 顆滾地球約 {stats.runs_per_450 > 0 ? '+' : ''}{stats.runs_per_450} 分、
            {stats.outs_per_450 > 0 ? '+' : ''}{stats.outs_per_450} 個出局
          </div>
        </div>
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
            {IF_POSITIONS.map(p => (
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
  const rA = dataA.optimized
  const rB = dataB.optimized

  const dOuts = (rA.exp_outs - rB.exp_outs) * 100
  const dRuns = rA.runs - rB.runs

  const numRow = (label, valA, valB, delta, higherIsBetter = true) => {
    const better = higherIsBetter ? delta > 0 : delta < 0
    const color  = Math.abs(delta) < 0.01 ? 'var(--slate-600)' : (better ? 'var(--green-600)' : 'var(--red-600)')
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
    const dx = Math.round(rA.positions[pos].x - rB.positions[pos].x)
    const dy = Math.round(rA.positions[pos].y - rB.positions[pos].y)
    const sign = (v) => v > 0 ? `+${v}` : `${v}`
    const color = (dx === 0 && dy === 0) ? 'var(--slate-600)' : 'var(--blue-600)'
    return (
      <tr key={pos}>
        <td style={td.label}>{pos}</td>
        <td style={td.val}>{fmtPos(rA.positions[pos])}</td>
        <td style={td.val}>{fmtPos(rB.positions[pos])}</td>
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
          {numRow('出局率', (rA.exp_outs * 100).toFixed(1) + '%', (rB.exp_outs * 100).toFixed(1) + '%', dOuts, true)}
          {numRow('預期失分', rA.runs.toFixed(2), rB.runs.toFixed(2), dRuns, false)}
          <tr><td colSpan={4} style={{ padding: '4px 0' }}>
            <hr style={{ border: 'none', borderTop: '1px solid var(--slate-100)', margin: 0 }} />
          </td></tr>
          {IF_POSITIONS.map(posRow)}
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
      <p style={{ color: 'white', marginTop: 12, fontSize: 13, fontWeight: 500 }}>最佳化計算中…</p>
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
    width: '100%', padding: '6px 8px', background: 'var(--slate-50)',
    color: 'var(--slate-700)', border: '1px solid var(--slate-200)', borderRadius: 6, fontSize: 11,
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
