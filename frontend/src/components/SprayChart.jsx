import { useState, useCallback } from 'react'

// viewBox: x ∈ [-430, 430], SVG y = -field_y (home plate at SVG y=0 = bottom)
const VB = '-430 -490 860 520'

const POS_COLORS = { LF: '#60a5fa', CF: '#34d399', RF: '#fb923c' }

// HSL color: catch_prob 0→red(0°), 1→green(120°)
const probColor = (p) => `hsl(${Math.round(p * 120)}, 90%, 52%)`

// Generic outfield arc when no park boundary (sample points on r=410ft circle ±45°)
function genericArcPath() {
  const pts = []
  for (let deg = -45; deg <= 45; deg += 3) {
    const rad = (deg * Math.PI) / 180
    pts.push(`${(Math.sin(rad) * 410).toFixed(1)},${(-Math.cos(rad) * 410).toFixed(1)}`)
  }
  return `M${pts[0]}L${pts.slice(1).join('L')}`
}

// Convert park boundary array to SVG path (y negated)
function parkPath(pts) {
  if (!pts || pts.length === 0) return null
  return `M${pts.map(p => `${p.x.toFixed(1)},${(-p.y).toFixed(1)}`).join('L')}Z`
}

// Infield diamond vertices (90ft bases → feet from home plate)
const B = (90 / Math.sqrt(2)).toFixed(1)   // ≈ 63.6
const DIAMOND = `0,0 ${B},${-B} 0,${-(2 * B * Math.SQRT2 / 2).toFixed(1)} ${-B},${-B}`
// 2B = (0, 90√2) ≈ (0, 127.3)
const SECOND_BASE_Y = -(90 * Math.SQRT2).toFixed(1)

export default function SprayChart({ balls = [], positions = null, parkBoundary = null }) {
  const [tooltip, setTooltip] = useState(null) // { ball, tx, ty }

  const mainResult = positions
    ? (positions.custom || positions.with_park || positions.no_park)
    : null
  const leagueAvg = positions?.league_avg

  const handleEnter = useCallback((ball, e) => {
    // tx, ty in SVG viewBox coords (field x, -field y)
    setTooltip({ ball, tx: ball.x, ty: -ball.y })
  }, [])

  const boundary = parkPath(parkBoundary)
  const arc      = genericArcPath()

  // Clamp tooltip so it stays inside viewBox
  const tip = tooltip ? {
    tx: Math.max(-360, Math.min(360, tooltip.tx)),
    ty: Math.min(-30, Math.max(-450, tooltip.ty - 28)),
  } : null

  return (
    <div style={{ position: 'relative', width: '100%' }}>
      <svg viewBox={VB} style={{ width: '100%', display: 'block' }}>
        <defs>
          <linearGradient id="probGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%"   stopColor="hsl(0,90%,52%)" />
            <stop offset="50%"  stopColor="hsl(60,90%,52%)" />
            <stop offset="100%" stopColor="hsl(120,90%,52%)" />
          </linearGradient>
          <clipPath id="fieldClip">
            <path d={boundary || arc} />
          </clipPath>
        </defs>

        {/* ── 背景草皮 ── */}
        <rect x="-430" y="-490" width="860" height="520" fill="#1a4a25" />

        {/* ── 界外線（foul lines）── */}
        <line x1="0" y1="0" x2="-450" y2="-450" stroke="white" strokeWidth="1.5" opacity="0.35" />
        <line x1="0" y1="0" x2="450"  y2="-450" stroke="white" strokeWidth="1.5" opacity="0.35" />

        {/* ── 球場圍牆 ── */}
        {boundary
          ? <path d={boundary} fill="none" stroke="#fbbf24" strokeWidth="3" />
          : <path d={arc}      fill="none" stroke="#94a3b8" strokeWidth="2" strokeDasharray="8 5" />
        }

        {/* ── 內野（土色多邊形）── */}
        <polygon points={DIAMOND}
          fill="#8b6340" fillOpacity="0.25"
          stroke="white" strokeWidth="0.8" strokeOpacity="0.3" />

        {/* ── 投手丘 ── */}
        <circle cx="0" cy="-60.5" r="9" fill="#8b6340" fillOpacity="0.3" />

        {/* ── 本壘板標記 ── */}
        <circle cx="0" cy="0" r="5" fill="white" fillOpacity="0.5" />

        {/* ── 球散點 ── */}
        {balls.map((ball, i) =>
          ball.is_wall_ball ? (
            <circle key={i} cx={ball.x} cy={-ball.y} r="5"
              fill="#94a3b8" fillOpacity="0.25" />
          ) : (
            <circle key={i} cx={ball.x} cy={-ball.y} r="7"
              fill={probColor(ball.catch_prob)} fillOpacity="0.78"
              stroke="white" strokeWidth="0.6"
              style={{ cursor: 'pointer' }}
              onMouseEnter={e => handleEnter(ball, e)}
              onMouseLeave={() => setTooltip(null)}
            />
          )
        )}

        {/* ── 聯盟平均站位（虛線圓圈）── */}
        {leagueAvg && ['LF', 'CF', 'RF'].map(pos => {
          const c = leagueAvg[pos]
          return (
            <circle key={`avg-${pos}`}
              cx={c.x} cy={-c.y} r="14"
              fill="none"
              stroke={POS_COLORS[pos]} strokeWidth="2"
              strokeDasharray="5 4" opacity="0.55" />
          )
        })}

        {/* ── 最佳/自選站位（實心菱形 + 標籤）── */}
        {mainResult && ['LF', 'CF', 'RF'].map(pos => {
          const c = mainResult[pos]
          const col = POS_COLORS[pos]
          return (
            <g key={pos} transform={`translate(${c.x},${-c.y})`}>
              <polygon points="0,-13 10,0 0,13 -10,0"
                fill={col} stroke="white" strokeWidth="1.5" />
              <text textAnchor="middle" y="-19"
                fill={col} fontSize="18" fontWeight="800"
                stroke="#0f172a" strokeWidth="3" paintOrder="stroke"
              >{pos}</text>
            </g>
          )
        })}

        {/* ── Tooltip ── */}
        {tooltip && tip && (() => {
          const { ball } = tooltip
          const pct = (ball.catch_prob * 100).toFixed(0)
          return (
            <g transform={`translate(${tip.tx},${tip.ty})`} pointerEvents="none">
              <rect x="-65" y="-46" width="130" height="44"
                rx="5" fill="rgba(15,23,42,0.92)" />
              <text textAnchor="middle" y="-26"
                fill="white" fontSize="15" fontWeight="700">
                接殺機率 {pct}%
              </text>
              <text textAnchor="middle" y="-9"
                fill="#94a3b8" fontSize="12">
                {ball.is_wall_ball
                  ? '打牆球'
                  : `落點 (${Math.round(ball.x)}, ${Math.round(ball.y)}) ft`}
              </text>
            </g>
          )
        })()}

        {/* ── 顏色圖例 ── */}
        <g transform="translate(-200, 20)">
          <rect x="0" y="0" width="160" height="12" rx="3"
            fill="url(#probGrad)" />
          <text x="0"   y="26" fill="#cbd5e1" fontSize="12" textAnchor="start">難接 0%</text>
          <text x="160" y="26" fill="#cbd5e1" fontSize="12" textAnchor="end">易接 100%</text>
        </g>

        {/* ── 打牆球圖例 ── */}
        <g transform="translate(80, 20)">
          <circle cx="6" cy="6" r="5" fill="#94a3b8" fillOpacity="0.4" />
          <text x="16" y="12" fill="#94a3b8" fontSize="12">打牆球（不計入）</text>
        </g>
      </svg>
    </div>
  )
}
