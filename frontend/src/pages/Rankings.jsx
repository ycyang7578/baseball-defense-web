import { useState, useEffect, useMemo } from 'react'
import { fetchYears, fetchFielders, fetchStarStats } from '../api'

const POSITIONS = ['LF', 'CF', 'RF']
const TABS = ['ALL', 'LF', 'CF', 'RF']
const STARS = [5, 4, 3, 2, 1]
const STAR_LABELS = {
  5: '5 Star (0-25%)',
  4: '4 Star (26-50%)',
  3: '3 Star (51-75%)',
  2: '2 Star (76-90%)',
  1: '1 Star (91-95%)',
}

const ACTIVE_BG  = '#eff6ff'
const ACTIVE_HDR = '#dbeafe'

const HEADSHOT_URL = (id) =>
  `https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_60,q_auto:best/v1/people/${id}/headshot/67/current`

const TEAM_LOGO_URL = (teamId) =>
  `https://www.mlbstatic.com/team-logos/${teamId}.svg`

const TEAM_ABBR = {
  108: 'LAA', 109: 'ARI', 110: 'BAL', 111: 'BOS', 112: 'CHC',
  113: 'CIN', 114: 'CLE', 115: 'COL', 116: 'DET', 117: 'HOU',
  118: 'KC',  119: 'LAD', 120: 'WSH', 121: 'NYM', 133: 'OAK',
  134: 'PIT', 135: 'SD',  136: 'SEA', 137: 'SF',  138: 'STL',
  139: 'TB',  140: 'TEX', 141: 'TOR', 142: 'MIN', 143: 'PHI',
  144: 'ATL', 145: 'CWS', 146: 'MIA', 147: 'NYY', 158: 'MIL',
}

const displayName = (s) =>
  s && s.includes(', ') ? s.split(', ').reverse().join(' ') : (s || '')

function TeamLogo({ teamId }) {
  if (!teamId) return <div style={{ width: 28, height: 28 }} />
  return (
    <img
      src={TEAM_LOGO_URL(teamId)}
      alt={TEAM_ABBR[teamId]}
      title={TEAM_ABBR[teamId]}
      style={{ width: 28, height: 28, objectFit: 'contain', flexShrink: 0 }}
    />
  )
}

function PlayerAvatar({ playerId, name }) {
  if (!playerId) {
    return (
      <div style={{
        width: 28, height: 28, borderRadius: '50%',
        background: '#e2e8f0', flexShrink: 0,
      }} />
    )
  }
  return (
    <img
      src={HEADSHOT_URL(playerId)}
      alt={name}
      style={{
        width: 28, height: 28, borderRadius: '50%',
        objectFit: 'cover', flexShrink: 0,
        background: '#e2e8f0',
      }}
    />
  )
}

export default function Rankings() {
  const [pos, setPos]             = useState('CF')
  const [minOpp, setMinOpp]       = useState(100)
  const [fielders, setFielders]   = useState({})
  const [starData, setStarData]   = useState({})
  const [loading, setLoading]     = useState(true)
  const [sortKey, setSortKey]     = useState('rate')
  const [sortDir, setSortDir]     = useState('desc')
  const [availYears, setAvailYears] = useState([2025])
  const [year, setYear]           = useState(2025)

  useEffect(() => {
    fetchYears().then(ys => {
      setAvailYears(ys)
      setYear(ys[ys.length - 1])   // 預設最新年份
    }).catch(() => {})
  }, [])

  useEffect(() => {
    setLoading(true)
    Promise.all([fetchFielders(minOpp, year), fetchStarStats(year)])
      .then(([f, s]) => { setFielders(f); setStarData(s); setLoading(false) })
      .catch(() => setLoading(false))
  }, [minOpp, year])

  function handleSort(key) {
    if (sortKey === key) setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    else { setSortKey(key); setSortDir('desc') }
  }

  const rows = useMemo(() => {
    const source = pos === 'ALL'
      ? POSITIONS.flatMap(p => (fielders[p] || []).map(f => ({ ...f, position: p })))
      : (fielders[pos] || []).map(f => ({ ...f, position: pos }))

    const base = source.map(f => {
      const sd = starData[f.name] || null
      const row = {
        name:      f.name,
        player_id: f.player_id ?? null,
        team_id:   f.team_id   ?? null,
        position:  f.position,
        oaa:       f.oaa,
        n_opp:     f.n_opp,
        rate:      f.oaa != null && f.n_opp ? f.oaa / f.n_opp * 100 : null,
        stars:     sd ? sd.stars : null,
        all:       sd ? sd.all   : null,
      }
      if (sd) {
        STARS.forEach(i => {
          const sv = sd.stars[i]
          row[`s${i}_outs`] = sv.outs
          row[`s${i}_opp`]  = sv.opp
          row[`s${i}_pct`]  = sv.opp ? sv.outs / sv.opp * 100 : null
        })
        row['all_outs'] = sd.all.outs
        row['all_opp']  = sd.all.opp
        row['all_pct']  = sd.all.opp ? sd.all.outs / sd.all.opp * 100 : null
      }
      return row
    })
    return [...base].sort((a, b) => {
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
  }, [fielders, starData, pos, sortKey, sortDir])

  const isActive = key => sortKey === key

  const thStyle = (key, extra = {}) => ({
    ...s.th, ...s.thSort,
    ...(isActive(key) ? { background: ACTIVE_HDR, color: '#1e40af' } : {}),
    ...extra,
  })

  const tdStyle = (key, extra = {}) => ({
    ...s.td,
    ...(isActive(key) ? { background: ACTIVE_BG } : {}),
    ...extra,
  })

  const pct = (outs, opp) => opp ? (outs / opp * 100).toFixed(1) + '%' : '—'

  return (
    <div style={s.page}>
      <div style={s.header}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
          <h1 style={s.title}>外野手守備排名</h1>
          <div style={s.yearTabs}>
            {availYears.map(y => (
              <button key={y} onClick={() => setYear(y)}
                style={{ ...s.yearTab, ...(year === y ? s.yearTabActive : {}) }}>
                {y}
              </button>
            ))}
          </div>
        </div>
        <span style={s.subtitle}>OAA 與星級均由本模型以全量守備機會計算，基於賽季平均站位，非 Statcast 官方數值</span>
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
          <span style={s.oppLabel}>最低守備機會（模型）</span>
          <input type="range" min={0} max={400} step={25} value={minOpp}
            onChange={e => setMinOpp(Number(e.target.value))}
            style={{ width: 130, accentColor: '#2563eb' }} />
          <span style={s.oppVal}>{minOpp}</span>
        </div>
      </div>

      {loading ? (
        <div style={s.loading}>載入中…</div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={s.table}>
            <thead>
              <tr>
                <th style={s.th} rowSpan={2}>#</th>
                <th onClick={() => handleSort('name')}
                  style={thStyle('name', { verticalAlign: 'middle', textAlign: 'left', minWidth: 180 })}
                  rowSpan={2}>球員</th>
                <th style={s.th} rowSpan={2}>球隊</th>
                {pos === 'ALL' && (
                  <th onClick={() => handleSort('position')}
                    style={thStyle('position', { verticalAlign: 'middle' })}
                    rowSpan={2}>守位</th>
                )}
                <th onClick={() => handleSort('n_opp')} style={thStyle('n_opp', { verticalAlign: 'middle' })} rowSpan={2}>模型機會</th>
                <th onClick={() => handleSort('oaa')}   style={thStyle('oaa',   { verticalAlign: 'middle' })} rowSpan={2}>模型OAA</th>
                <th onClick={() => handleSort('rate')}  style={thStyle('rate',  { verticalAlign: 'middle' })} rowSpan={2}>OAA/100</th>
                {STARS.map(i => (
                  <th key={i} colSpan={3}
                    style={{ ...s.th, ...s.thGroup, borderLeft: '1px solid #e2e8f0' }}>
                    {STAR_LABELS[i]}
                  </th>
                ))}
                <th colSpan={3} style={{ ...s.th, ...s.thGroup, borderLeft: '1px solid #e2e8f0' }}>All Plays</th>
              </tr>
              <tr>
                {STARS.map(i =>
                  ['outs', 'opp', 'pct'].map((h, hi) => (
                    <th key={`${i}-${h}`}
                      onClick={() => handleSort(`s${i}_${h}`)}
                      style={thStyle(`s${i}_${h}`, hi === 0 ? { borderLeft: '1px solid #e2e8f0' } : {})}>
                      {h === 'pct' ? '%' : h.charAt(0).toUpperCase() + h.slice(1)}
                    </th>
                  ))
                )}
                {['outs', 'opp', 'pct'].map((h, hi) => (
                  <th key={`all-${h}`}
                    onClick={() => handleSort(`all_${h}`)}
                    style={thStyle(`all_${h}`, hi === 0 ? { borderLeft: '1px solid #e2e8f0' } : {})}>
                    {h === 'pct' ? '%' : h.charAt(0).toUpperCase() + h.slice(1)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map(r => (
                <tr key={r.name + r.position} style={s.tr}>
                  <td style={{ ...s.td, color: '#9ca3af', fontSize: 11 }}>{r.rank}</td>
                  <td style={tdStyle('name', { fontWeight: 500, textAlign: 'left' })}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 9, whiteSpace: 'nowrap' }}>
                      <PlayerAvatar playerId={r.player_id} name={r.name} />
                      {displayName(r.name)}
                    </div>
                  </td>
                  <td style={s.td}>
                    <TeamLogo teamId={r.team_id} />
                  </td>
                  {pos === 'ALL' && (
                    <td style={tdStyle('position', { color: '#6b7280' })}>{r.position}</td>
                  )}
                  <td style={tdStyle('n_opp')}>{r.n_opp}</td>
                  <td style={{ ...tdStyle('oaa'), ...oaaColor(r.oaa) }}>
                    {r.oaa != null ? (r.oaa >= 0 ? '+' : '') + r.oaa.toFixed(1) : '—'}
                  </td>
                  <td style={{ ...tdStyle('rate'), ...oaaColor(r.rate), fontWeight: 600 }}>
                    {r.rate != null ? (r.rate >= 0 ? '+' : '') + r.rate.toFixed(1) : '—'}
                  </td>
                  {STARS.map(i => {
                    const sv = r.stars ? r.stars[i] : null
                    return ['outs', 'opp', 'pct'].map((h, hi) => (
                      <td key={`${r.name}-s${i}-${h}`}
                        style={tdStyle(`s${i}_${h}`, {
                          ...(hi === 0 ? { borderLeft: '1px solid #f1f5f9' } : {}),
                          color: h === 'pct' ? '#6b7280' : undefined,
                        })}>
                        {sv ? (h === 'pct' ? pct(sv.outs, sv.opp) : sv[h]) : '—'}
                      </td>
                    ))
                  })}
                  {['outs', 'opp', 'pct'].map((h, hi) => {
                    const a = r.all
                    return (
                      <td key={`${r.name}-all-${h}`}
                        style={tdStyle(`all_${h}`, {
                          fontWeight: 500,
                          ...(hi === 0 ? { borderLeft: '1px solid #e2e8f0' } : {}),
                          color: h === 'pct' ? '#6b7280' : undefined,
                        })}>
                        {a ? (h === 'pct' ? pct(a.outs, a.opp) : a[h]) : '—'}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function oaaColor(val) {
  if (val == null) return {}
  if (val > 2)  return { color: '#16a34a' }
  if (val > 0)  return { color: '#4ade80', filter: 'brightness(0.75)' }
  if (val < -2) return { color: '#dc2626' }
  if (val < 0)  return { color: '#f87171', filter: 'brightness(0.8)' }
  return {}
}

const s = {
  page:     { padding: '24px 28px' },
  header:   { marginBottom: 18 },
  title:    { margin: '0 0 4px', fontSize: 20, fontWeight: 700 },
  subtitle: { fontSize: 11, color: '#9ca3af' },
  controls: { display: 'flex', alignItems: 'center', gap: 24, marginBottom: 18, flexWrap: 'wrap' },
  tabs:     { display: 'flex', gap: 6 },
  tab: {
    padding: '6px 18px', border: '1px solid #d1d5db', borderRadius: 6,
    background: 'white', color: '#6b7280', cursor: 'pointer', fontSize: 14, fontWeight: 600,
  },
  tabActive: { background: '#2563eb', color: '#fff', border: '1px solid #2563eb' },
  yearTabs: { display: 'flex', gap: 4 },
  yearTab: {
    padding: '3px 10px', border: '1px solid #d1d5db', borderRadius: 5,
    background: 'white', color: '#6b7280', cursor: 'pointer', fontSize: 13, fontWeight: 600,
  },
  yearTabActive: { background: '#1e40af', color: '#fff', border: '1px solid #1e40af' },
  oppRow:   { display: 'flex', alignItems: 'center', gap: 8 },
  oppLabel: { fontSize: 12, color: '#6b7280' },
  oppVal:   { fontSize: 13, color: '#1e293b', minWidth: 30 },
  loading:  { color: '#9ca3af', marginTop: 40, textAlign: 'center' },
  table:    { borderCollapse: 'collapse', fontSize: 12, whiteSpace: 'nowrap' },
  th: {
    padding: '7px 10px', fontSize: 11, color: '#6b7280',
    textTransform: 'uppercase', letterSpacing: '0.04em',
    borderBottom: '2px solid #e2e8f0', textAlign: 'center',
    userSelect: 'none', background: '#f8fafc',
  },
  thGroup:  { fontWeight: 700, color: '#374151', fontSize: 12 },
  thSort:   { cursor: 'pointer' },
  tr:       { borderBottom: '1px solid #f1f5f9' },
  td:       { padding: '6px 10px', color: '#1e293b', textAlign: 'center' },
}
