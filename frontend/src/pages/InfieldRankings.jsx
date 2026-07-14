import { useState, useEffect, useMemo } from 'react'
import { fetchIfFielders } from '../api'
import { TEAM_ABBR, TeamLogo, PlayerAvatar, displayName, oaaColor } from '../components/playerDisplay'

const POSITIONS = ['1B', '2B', '3B', 'SS']
const TABS = ['ALL', '1B', '2B', '3B', 'SS']
// 2025=樣本外評分年；2023/2024 落在訓練年內屬 in-sample、數字偏樂觀（展示用）
const YEARS = [2023, 2024, 2025]

const ACTIVE_BG  = '#eff6ff'
const ACTIVE_HDR = '#dbeafe'

export default function InfieldRankings() {
  const [pos, setPos]           = useState('SS')
  const [minBalls, setMinBalls] = useState(100)
  const [fielders, setFielders] = useState({})
  const [loading, setLoading]   = useState(true)
  const [sortKey, setSortKey]   = useState('rate')
  const [sortDir, setSortDir]   = useState('desc')
  const [year, setYear]         = useState(YEARS[YEARS.length - 1])
  const [teamFilter, setTeamFilter] = useState('')

  useEffect(() => { setTeamFilter('') }, [year])

  useEffect(() => {
    setLoading(true)
    fetchIfFielders(minBalls, year)
      .then(f => { setFielders(f); setLoading(false) })
      .catch(() => setLoading(false))
  }, [minBalls, year])

  function handleSort(key) {
    if (sortKey === key) setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    else { setSortKey(key); setSortDir('desc') }
  }

  const allTeams = useMemo(() => {
    const ids = new Set()
    POSITIONS.forEach(p => (fielders[p] || []).forEach(f => { if (f.team_id) ids.add(f.team_id) }))
    return [...ids].sort((a, b) => (TEAM_ABBR[a] || '').localeCompare(TEAM_ABBR[b] || ''))
  }, [fielders])

  const rows = useMemo(() => {
    const source = pos === 'ALL'
      ? POSITIONS.flatMap(p => (fielders[p] || []).map(f => ({ ...f, position: p })))
      : (fielders[pos] || []).map(f => ({ ...f, position: pos }))

    const base = source.map(f => ({
      name:           f.name,
      player_id:      f.player_id ?? null,
      team_id:        f.team_id   ?? null,
      position:       f.position,
      oaa:            f.oaa,
      n_balls:        f.n_balls,
      rate:           f.oaa != null && f.n_balls ? f.oaa / f.n_balls * 100 : null,
      official_oaa:   f.official_oaa,
      official_n_opp: f.official_n_opp,
      official_rate:  f.official_oaa != null && f.official_n_opp
                        ? f.official_oaa / f.official_n_opp * 100 : null,
    }))
    const filtered = teamFilter
      ? base.filter(r => r.team_id === Number(teamFilter))
      : base
    return [...filtered].sort((a, b) => {
      if (sortKey === 'name' || sortKey === 'position') {
        const cmp = sortKey === 'position'
          ? a.position.localeCompare(b.position) || a.name.localeCompare(b.name)
          : a.name.localeCompare(b.name)
        return sortDir === 'desc' ? -cmp : cmp
      }
      const av = a[sortKey] ?? -Infinity
      const bv = b[sortKey] ?? -Infinity
      return sortDir === 'desc' ? bv - av : av - bv
    }).map((r, i) => ({ ...r, rank: i + 1 }))
  }, [fielders, pos, sortKey, sortDir, teamFilter])

  const isActive = key => sortKey === key

  const thStyle = (key, extra = {}) => ({
    ...s.th, ...s.thSort,
    ...(isActive(key) ? { background: ACTIVE_HDR, color: 'var(--blue-800)' } : {}),
    ...extra,
  })

  const tdStyle = (key, extra = {}) => ({
    ...s.td,
    ...(isActive(key) ? { background: ACTIVE_BG } : {}),
    ...extra,
  })

  const fmtSigned = (v, digits = 1) =>
    v != null ? (v >= 0 ? '+' : '') + Number(v).toFixed(digits) : '—'

  return (
    <div style={s.page}>
      <div style={s.header}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
          <h1 style={{ ...s.title, flexShrink: 0 }}>內野手守備排名</h1>
          <div style={s.yearTabs}>
            {YEARS.map(y => (
              <button key={y} onClick={() => setYear(y)}
                style={{ ...s.yearTab, ...(year === y ? s.yearTabActive : {}) }}>
                {y}
              </button>
            ))}
          </div>
        </div>
        <span style={s.subtitle}>
          模型 OAA 以滾地球出局難度計算（基於賽季平均站位，非 Statcast 官方數值）；右側官方欄位為 Statcast Infield OAA
        </span>
      </div>

      <div style={s.controls}>
        <div style={s.tabs}>
          {TABS.map(p => (
            <button key={p} onClick={() => setPos(p)}
              style={{ ...s.tab, ...(pos === p ? s.tabActive : {}) }}>
              {p}
            </button>
          ))}
        </div>
        <div style={s.oppRow}>
          <span style={s.oppLabel}>球隊篩選</span>
          <select value={teamFilter} onChange={e => setTeamFilter(e.target.value)}
            style={{ fontSize: 13, padding: '3px 6px', borderRadius: 5,
                     border: '1px solid var(--gray-300)', color: 'var(--gray-700)' }}>
            <option value="">全部球隊</option>
            {allTeams.map(tid => (
              <option key={tid} value={tid}>{TEAM_ABBR[tid] || tid}</option>
            ))}
          </select>
          {teamFilter && (
            <button onClick={() => setTeamFilter('')}
              style={{ fontSize: 11, padding: '2px 7px', borderRadius: 4,
                       border: '1px solid var(--gray-300)', cursor: 'pointer', background: 'white' }}>
              ✕
            </button>
          )}
        </div>
        <div style={s.oppRow}>
          <span style={s.oppLabel}>最低守備機會（模型）</span>
          <input type="range" min={0} max={400} step={25} value={minBalls}
            onChange={e => setMinBalls(Number(e.target.value))}
            style={{ width: 130, accentColor: 'var(--blue-600)' }} />
          <span style={s.oppVal}>{minBalls}</span>
        </div>
      </div>

      {loading ? (
        <div style={s.loading}>載入中…</div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={s.table}>
            <thead>
              <tr>
                <th style={s.th}>#</th>
                <th onClick={() => handleSort('name')}
                  style={thStyle('name', { textAlign: 'left', minWidth: 180 })}>球員</th>
                <th style={s.th}>球隊</th>
                {pos === 'ALL' && (
                  <th onClick={() => handleSort('position')} style={thStyle('position')}>守位</th>
                )}
                <th onClick={() => handleSort('n_balls')} style={thStyle('n_balls')}>模型機會</th>
                <th onClick={() => handleSort('oaa')}     style={thStyle('oaa')}>模型OAA</th>
                <th onClick={() => handleSort('rate')}    style={thStyle('rate')}>OAA/100</th>
                <th onClick={() => handleSort('official_oaa')}
                  style={thStyle('official_oaa', { borderLeft: '1px solid var(--slate-200)' })}>官方OAA</th>
                <th onClick={() => handleSort('official_n_opp')} style={thStyle('official_n_opp')}>官方機會</th>
                <th onClick={() => handleSort('official_rate')}  style={thStyle('official_rate')}>官方OAA/100</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(r => (
                <tr key={r.name + r.position} style={s.tr}>
                  <td style={{ ...s.td, color: 'var(--gray-400)', fontSize: 11 }}>{r.rank}</td>
                  <td style={tdStyle('name', { fontWeight: 500, textAlign: 'left' })}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 9, whiteSpace: 'nowrap' }}>
                      <PlayerAvatar playerId={r.player_id} name={r.name} />
                      <span>{displayName(r.name)}</span>
                    </div>
                  </td>
                  <td style={s.td}>
                    <TeamLogo teamId={r.team_id} />
                  </td>
                  {pos === 'ALL' && (
                    <td style={tdStyle('position', { color: 'var(--gray-500)' })}>{r.position}</td>
                  )}
                  <td style={tdStyle('n_balls')}>{r.n_balls}</td>
                  <td style={{ ...tdStyle('oaa'), ...oaaColor(r.oaa) }}>{fmtSigned(r.oaa)}</td>
                  <td style={{ ...tdStyle('rate'), ...oaaColor(r.rate), fontWeight: 600 }}>{fmtSigned(r.rate)}</td>
                  <td style={{ ...tdStyle('official_oaa'), ...oaaColor(r.official_oaa),
                               borderLeft: '1px solid var(--slate-100)' }}>
                    {fmtSigned(r.official_oaa, 0)}
                  </td>
                  <td style={tdStyle('official_n_opp')}>{r.official_n_opp ?? '—'}</td>
                  <td style={{ ...tdStyle('official_rate'), ...oaaColor(r.official_rate) }}>
                    {fmtSigned(r.official_rate)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

const s = {
  page:     { padding: '24px 28px' },
  header:   { marginBottom: 18 },
  title:    { margin: '0 0 4px', fontSize: 20, fontWeight: 700 },
  subtitle: { fontSize: 11, color: 'var(--gray-400)' },
  controls: { display: 'flex', alignItems: 'center', gap: 24, marginBottom: 18, flexWrap: 'wrap' },
  tabs:     { display: 'flex', gap: 6 },
  tab: {
    padding: '6px 18px', border: '1px solid var(--gray-300)', borderRadius: 6,
    background: 'white', color: 'var(--gray-500)', cursor: 'pointer', fontSize: 14, fontWeight: 600,
  },
  tabActive: { background: 'var(--blue-600)', color: '#fff', border: '1px solid var(--blue-600)' },
  yearTabs: { display: 'flex', gap: 4 },
  yearTab: {
    padding: '3px 10px', border: '1px solid var(--gray-300)', borderRadius: 5,
    background: 'white', color: 'var(--gray-500)', cursor: 'pointer', fontSize: 13, fontWeight: 600,
  },
  yearTabActive: { background: 'var(--blue-800)', color: '#fff', border: '1px solid var(--blue-800)' },
  oppRow:   { display: 'flex', alignItems: 'center', gap: 8 },
  oppLabel: { fontSize: 12, color: 'var(--gray-500)' },
  oppVal:   { fontSize: 13, color: 'var(--slate-800)', minWidth: 30 },
  loading:  { color: 'var(--gray-400)', marginTop: 40, textAlign: 'center' },
  table:    { borderCollapse: 'collapse', fontSize: 12, whiteSpace: 'nowrap' },
  th: {
    padding: '7px 10px', fontSize: 11, color: 'var(--gray-500)',
    textTransform: 'uppercase', letterSpacing: '0.04em',
    borderBottom: '2px solid var(--slate-200)', textAlign: 'center',
    userSelect: 'none', background: 'var(--slate-50)',
  },
  thSort:   { cursor: 'pointer' },
  tr:       { borderBottom: '1px solid var(--slate-100)' },
  td:       { padding: '6px 10px', color: 'var(--slate-800)', textAlign: 'center' },
}
