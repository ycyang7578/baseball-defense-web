const BASE = '/api'

export async function fetchBatters(year = 2025) {
  const res = await fetch(`${BASE}/batters?year=${year}`)
  if (!res.ok) throw new Error('Failed to fetch batters')
  return res.json()
}

export async function fetchTeams() {
  const res = await fetch(`${BASE}/teams`)
  if (!res.ok) throw new Error('Failed to fetch teams')
  return res.json()
}

export async function fetchYears() {
  const res = await fetch(`${BASE}/years`)
  if (!res.ok) throw new Error('Failed to fetch available years')
  return res.json()
}

export async function fetchFielders(minOpp = 100, year = 2025) {
  const res = await fetch(`${BASE}/fielders?min_opp=${minOpp}&year=${year}`)
  if (!res.ok) throw new Error('Failed to fetch fielders')
  return res.json()
}

export async function fetchPlayerTrend(name) {
  const res = await fetch(`${BASE}/player_trend?name=${encodeURIComponent(name)}`)
  if (!res.ok) throw new Error('Failed to fetch trend')
  return res.json()
}

export async function fetchStarStats(year = 2025) {
  const res = await fetch(`${BASE}/star_stats?year=${year}`)
  if (!res.ok) throw new Error('Failed to fetch star stats')
  return res.json()
}

function _body({ batterId, year, on1b, on2b, on3b, outs, homeTeam, fielders }) {
  return JSON.stringify({
    batter_id: batterId,
    year: year || 2025,
    on_1b: on1b,
    on_2b: on2b,
    on_3b: on3b,
    outs,
    home_team: homeTeam || null,
    fielders: fielders && Object.keys(fielders).length ? fielders : null,
  })
}

export async function optimize(params) {
  const res = await fetch(`${BASE}/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: _body(params),
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Optimization failed')
  }
  return res.json()
}

// 回傳 { url, positions, stats, situation, title }
export async function optimizePlot(params) {
  const res = await fetch(`${BASE}/optimize_plot`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: _body(params),
  })
  if (!res.ok) {
    let detail = 'Optimization failed'
    try { detail = (await res.json()).detail || detail } catch {}
    throw new Error(detail)
  }
  const data = await res.json()
  const bytes = Uint8Array.from(atob(data.image_b64), c => c.charCodeAt(0))
  const blob = new Blob([bytes], { type: 'image/png' })
  return {
    url: URL.createObjectURL(blob),
    positions: data.positions,
    stats: data.stats,
    situation: data.situation,
    title: data.title,
    balls: data.balls || [],
    parkBoundary: data.park_boundary || null,
  }
}
