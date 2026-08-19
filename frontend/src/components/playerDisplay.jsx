// Player display components and constants shared by Rankings (outfield) and InfieldRankings (infield)

export const HEADSHOT_URL = (id) =>
  `https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_60,q_auto:best/v1/people/${id}/headshot/67/current`

export const TEAM_LOGO_URL = (teamId) =>
  `https://www.mlbstatic.com/team-logos/${teamId}.svg`

export const TEAM_ABBR = {
  108: 'LAA', 109: 'ARI', 110: 'BAL', 111: 'BOS', 112: 'CHC',
  113: 'CIN', 114: 'CLE', 115: 'COL', 116: 'DET', 117: 'HOU',
  118: 'KC',  119: 'LAD', 120: 'WSH', 121: 'NYM', 133: 'OAK',
  134: 'PIT', 135: 'SD',  136: 'SEA', 137: 'SF',  138: 'STL',
  139: 'TB',  140: 'TEX', 141: 'TOR', 142: 'MIN', 143: 'PHI',
  144: 'ATL', 145: 'CWS', 146: 'MIA', 147: 'NYY', 158: 'MIL',
}

export const displayName = (s) =>
  s && s.includes(', ') ? s.split(', ').reverse().join(' ') : (s || '')

export function TeamLogo({ teamId }) {
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

export function PlayerAvatar({ playerId, name }) {
  if (!playerId) {
    return (
      <div style={{
        width: 28, height: 28, borderRadius: '50%',
        background: 'var(--slate-200)', flexShrink: 0,
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
        background: 'var(--slate-200)',
      }}
    />
  )
}

export function oaaColor(val) {
  if (val == null) return {}
  if (val > 2)  return { color: 'var(--green-600)' }
  if (val > 0)  return { color: '#4ade80', filter: 'brightness(0.75)' }
  if (val < -2) return { color: 'var(--red-600)' }
  if (val < 0)  return { color: '#f87171', filter: 'brightness(0.8)' }
  return {}
}
