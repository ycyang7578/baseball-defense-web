import { useEffect, useState } from 'react'
import { fetchIfBatters, fetchIfFielderOptions, fetchIfResult,
         fetchIfResultCustom, fetchIfYears } from '../api'
import SearchSelect from '../components/SearchSelect'
import InfieldChart from '../components/InfieldChart'

const displayName = (s) => (s && s.includes(', ')) ? s.split(', ').reverse().join(' ') : (s || '')
const IF_POSITIONS = ['1B', '2B', '3B', 'SS']

export default function Infield() {
  const [availYears, setAvailYears] = useState([])
  const [year, setYear]             = useState(null)
  const [batters, setBatters]       = useState([])
  const [batterId, setBatterId]     = useState('')
  const [data, setData]             = useState(null)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)
  const [colorBy, setColorBy]       = useState('optimized')
  const [probMin, setProbMin]       = useState(0)
  const [probMax, setProbMax]       = useState(1)
  const [fielderOpts, setFielderOpts] = useState(null)   // pos → options（null=功能未啟用）
  const [fielders, setFielders]       = useState({ '1B': '', '2B': '', '3B': '', SS: '' })

  useEffect(() => {
    fetchIfYears().then(ys => {
      setAvailYears(ys)
      if (ys.length) setYear(ys[ys.length - 1])
    }).catch(console.error)
  }, [])

  useEffect(() => {
    if (year === null) return
    fetchIfBatters(year).then(data => { setBatters(data); setBatterId('') }).catch(console.error)
    setFielders({ '1B': '', '2B': '', '3B': '', SS: '' })
    fetchIfFielderOptions(year).then(setFielderOpts).catch(() => setFielderOpts(null))
  }, [year])

  const anyFielder = IF_POSITIONS.some(p => fielders[p])

  async function handleShow() {
    if (!batterId) return
    setLoading(true)
    setError(null)
    try {
      setData(anyFielder
        ? await fetchIfResultCustom(Number(batterId), year, fielders)
        : await fetchIfResult(Number(batterId), year))
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
              括號內為該年滾地球數。站位限制依禁趨位規則：二壘兩側各兩人、不可換邊、站在內野土上
            </div>
          </Sec>

          {fielderOpts && (
            <Sec title="野手（預設聯盟平均）">
              {IF_POSITIONS.map(pos => (
                <div key={pos} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--slate-500)', width: 22 }}>{pos}</span>
                  <div style={{ flex: 1 }}>
                    <SearchSelect
                      options={(fielderOpts[pos] || []).map(f => ({
                        value: String(f.player_id), label: displayName(f.name),
                      }))}
                      value={fielders[pos]}
                      onChange={v => setFielders(prev => ({ ...prev, [pos]: v }))}
                      placeholder="聯盟平均"
                    />
                  </div>
                </div>
              ))}
              <div style={{ fontSize: 9, color: '#cbd5e1', marginTop: 4, lineHeight: 1.6 }}>
                指定野手時以該球員的守備參數微調站位（從最佳解出發局部調整）
              </div>
            </Sec>
          )}

          <div style={s.panelFooter}>
            <button
              onClick={handleShow}
              disabled={!batterId || loading}
              style={{ ...s.btn, opacity: (!batterId || loading) ? 0.5 : 1 }}
            >
              {loading ? '載入中…' : '顯示最佳站位'}
            </button>
            {error && <div style={s.error}>{error}</div>}
          </div>
        </div>

        {/* ── 右側結果區 ── */}
        <div className="app-chart-area" style={s.chartArea}>
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
                    <TitleBar data={data} />
                    <InfieldChart data={data} colorBy={colorBy} probMin={probMin} probMax={probMax} />
                  </div>
                  <IFStatsPanel data={data} />
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
        無人在壘・Standard 佈陣・{data.stats.n_gb} 顆滾地球
        {data.fielders && (
          <span style={{ color: 'var(--blue-600)', fontWeight: 600 }}>
            {'　'}
            {IF_POSITIONS.filter(p => data.fielders[p])
              .map(p => `${p} ${displayName(data.fielders[p])}`).join('・')}
          </span>
        )}
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
        系統依據打者的滾地球傾向，在禁趨位規則的限制下計算內野四人的最佳站位
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

function IFStatsPanel({ data }) {
  const { league, optimized, stats } = data
  const gain = stats.gain
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
            <div style={{ fontSize: 12, color: 'var(--slate-700)' }}>
              期望出局率 <strong style={{ fontSize: 14 }}>{(set.exp_outs * 100).toFixed(1)}%</strong>
            </div>
          </div>
        ))}
        <div style={{ background: gain > 0 ? '#f0fdf4' : '#fef2f2',
                      border: `1px solid ${gain > 0 ? '#bbf7d0' : '#fecaca'}`,
                      borderRadius: 7, padding: '8px 16px', minWidth: 120,
                      display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.05em',
                        color: gain > 0 ? '#166534' : '#991b1b', marginBottom: 4 }}>增益</div>
          <div style={{ fontSize: 20, fontWeight: 700,
                        color: gain > 0 ? 'var(--green-600)' : 'var(--red-600)' }}>
            {gain > 0 ? '+' : ''}{(gain * 100).toFixed(2)}%
          </div>
          <div style={{ fontSize: 10, color: 'var(--slate-500)', marginTop: 2 }}>
            每 450 顆滾地球約 {stats.outs_per_450 > 0 ? '+' : ''}{stats.outs_per_450} 個出局
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

const spc = {
  th: { padding: '2px 16px', textAlign: 'center', fontSize: 10, fontWeight: 600,
        color: 'var(--slate-400)', textTransform: 'uppercase', letterSpacing: '0.05em' },
  td: { padding: '4px 16px', textAlign: 'center', color: 'var(--slate-600)' },
}

function Overlay() {
  return (
    <div style={s.overlay}>
      <div style={s.spinner} />
      <p style={{ color: 'white', marginTop: 12, fontSize: 13, fontWeight: 500 }}>載入中…</p>
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
