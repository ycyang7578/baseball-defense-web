const BASE = '/api'

export async function fetchTeams() {
  const res = await fetch(`${BASE}/teams`)
  if (!res.ok) throw new Error('Failed to fetch teams')
  return res.json()
}

export async function fetchFielders(minOpp = 100, year = 2025) {
  const res = await fetch(`${BASE}/fielders?min_opp=${minOpp}&year=${year}`)
  if (!res.ok) throw new Error('Failed to fetch fielders')
  return res.json()
}

// ── 內野（排名／選單）──────────────────────────────────

export async function fetchIfYears() {
  const res = await fetch(`${BASE}/if_years`)
  if (!res.ok) throw new Error('Failed to fetch infield years')
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

// ── 內外野整合（外野線上優化＋內野錨定式精修，計算需時間）─────────

export async function fetchIntegratedBatters(year) {
  const res = await fetch(`${BASE}/integrated_batters?year=${year}`)
  if (!res.ok) throw new Error('Failed to fetch integrated batters')
  return res.json()
}

export async function optimizeIntegrated({ batterId, year, on1b, on2b, on3b, outs,
                                           homeTeam, ofFielders, ifFielders }) {
  const res = await fetch(`${BASE}/optimize_integrated`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      batter_id: batterId, year: year || 2025,
      on_1b: on1b, on_2b: on2b, on_3b: on3b, outs,
      home_team: homeTeam || null,
      of_fielders: ofFielders && Object.keys(ofFielders).length ? ofFielders : null,
      if_fielders: ifFielders && Object.keys(ifFielders).length ? ifFielders : null,
    }),
  })
  if (!res.ok) {
    let detail = 'Integrated optimization failed'
    try { detail = (await res.json()).detail || detail } catch {}
    throw new Error(detail)
  }
  return res.json()
}
