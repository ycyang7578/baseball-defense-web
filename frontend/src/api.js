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

// ── 內野（全部離線預算，查表即回）──────────────────────

export async function fetchIfYears() {
  const res = await fetch(`${BASE}/if_years`)
  if (!res.ok) throw new Error('Failed to fetch infield years')
  return res.json()
}

export async function fetchIfBatters(year) {
  const res = await fetch(`${BASE}/if_batters?year=${year}`)
  if (!res.ok) throw new Error('Failed to fetch infield batters')
  return res.json()
}

export async function fetchIfResult(batterId, year) {
  const res = await fetch(`${BASE}/if_result?batter_id=${batterId}&year=${year}`)
  if (!res.ok) {
    let detail = 'Failed to fetch infield result'
    try { detail = (await res.json()).detail || detail } catch {}
    throw new Error(detail)
  }
  return res.json()
}

export async function fetchIfFielders(minBalls = 100, year = 2025) {
  const res = await fetch(`${BASE}/if_fielders?min_balls=${minBalls}&year=${year}`)
  if (!res.ok) throw new Error('Failed to fetch infield fielders')
  return res.json()
}

export async function fetchIfFielderOptions(year) {
  const res = await fetch(`${BASE}/if_fielder_options?year=${year}`)
  if (!res.ok) throw new Error('Failed to fetch infield fielder options')
  return res.json()
}

export async function fetchIfResultCustom(batterId, year, fielderIds) {
  const params = new URLSearchParams({ batter_id: batterId, year })
  const keys = { '1B': 'fielder_1b', '2B': 'fielder_2b', '3B': 'fielder_3b', SS: 'fielder_ss' }
  for (const [pos, key] of Object.entries(keys)) {
    if (fielderIds[pos]) params.set(key, fielderIds[pos])
  }
  const res = await fetch(`${BASE}/if_result_custom?${params}`)
  if (!res.ok) {
    let detail = 'Failed to fetch custom infield result'
    try { detail = (await res.json()).detail || detail } catch {}
    throw new Error(detail)
  }
  return res.json()
}

// ── 內外野整合（外野線上優化＋內野錨定式精修，計算需時間）─────────

export async function optimizeIntegrated({ batterId, year, on1b, on2b, on3b, outs }) {
  const res = await fetch(`${BASE}/optimize_integrated`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      batter_id: batterId, year: year || 2025,
      on_1b: on1b, on_2b: on2b, on_3b: on3b, outs,
    }),
  })
  if (!res.ok) {
    let detail = 'Integrated optimization failed'
    try { detail = (await res.json()).detail || detail } catch {}
    throw new Error(detail)
  }
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
