import { useEffect, useState } from 'react'

// ── Layout (same as SprayChart) ─────────────────────────
const PL = 52, PT = 62, PW = 560, PH = 460, PR = 88, PB = 44
const SVG_W = PL + PW + PR, SVG_H = PT + PH + PB
const X0 = -280, X1 = 280, Y0 = -10, Y1 = 450

const tx = x => PL + (x - X0) / (X1 - X0) * PW
const ty = y => PT + (Y1 - y) / (Y1 - Y0) * PH
const ARC_Y = Math.sqrt(400 * 400 - 280 * 280)
const X_TICKS = [-200, -100, 0, 100, 200]
const Y_TICKS = [0, 100, 200, 300, 400]

// ── Marker helpers (same as SprayChart) ─────────────────
function starPts(cx, cy, r) {
  return Array.from({ length: 10 }, (_, i) => {
    const a = Math.PI * i / 5 - Math.PI / 2
    const rad = i % 2 === 0 ? r : r * 0.38
    return `${(cx + Math.cos(a) * rad).toFixed(2)},${(cy + Math.sin(a) * rad).toFixed(2)}`
  }).join(' ')
}
function diamondPts(cx, cy, r) {
  return `${cx},${cy - r} ${cx + r * 0.7},${cy} ${cx},${cy + r} ${cx - r * 0.7},${cy}`
}

const STYLES = {
  league_avg: { color: '#1565C0', shape: 'diamond', r: 7,  dy: -20 },
  no_park:    { color: '#C0392B', shape: 'circle',  r: 8,  dy: -24 },
  with_park:  { color: '#7B2FBE', shape: 'star',    r: 12, dy: +22 },
  custom:     { color: '#7B2FBE', shape: 'star',    r: 12, dy: +22 },
}

function PosMarker({ cx, cy, st, code }) {
  const { color, shape, r, dy } = st
  return (
    <g>
      {shape === 'diamond' && <polygon points={diamondPts(cx, cy, r)} fill={color} stroke="white" strokeWidth="1.5" />}
      {shape === 'circle'  && <circle  cx={cx} cy={cy} r={r}          fill={color} stroke="white" strokeWidth="1.5" />}
      {shape === 'star'    && <polygon points={starPts(cx, cy, r)}     fill={color} stroke="white" strokeWidth="1.2" />}
      <rect x={cx - 12} y={cy + dy - 7} width={24} height={14}
        rx="2.5" fill={color} stroke="white" strokeWidth="0.8" opacity="0.95" />
      <text x={cx} y={cy + dy} textAnchor="middle" dominantBaseline="middle"
        fill="white" fontSize="9.5" fontWeight="bold">{code}</text>
    </g>
  )
}

function LegendItem({ color, shape, label, y }) {
  return (
    <g>
      {shape === 'line'    && <line x1={2} y1={y} x2={18} y2={y} stroke={color} strokeWidth="2.2" />}
      {shape === 'star'    && <polygon points={starPts(10, y, 6)} fill={color} stroke="black" strokeWidth="0.4" />}
      {shape === 'diamond' && <polygon points={diamondPts(10, y, 6)} fill={color} stroke="white" strokeWidth="1" />}
      {shape === 'circle'  && <circle cx={10} cy={y} r={5} fill={color} stroke="white" strokeWidth="1" />}
      <text x={24} y={y} dominantBaseline="middle" fontSize="9.5" fill="#333">{label}</text>
    </g>
  )
}

// ── Canvas density renderer ──────────────────────────────
function useDensityImg(balls) {
  const [src, setSrc] = useState(null)

  useEffect(() => {
    if (!balls || balls.length === 0) { setSrc(null); return }

    const canvas = document.createElement('canvas')
    canvas.width = PW
    canvas.height = PH
    const ctx = canvas.getContext('2d')

    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, PW, PH)

    // Each ball = radial gradient blob; overlapping accumulates density
    const R = 30
    for (const ball of balls) {
      if (ball.is_wall_ball) continue
      const bx = (ball.x - X0) / (X1 - X0) * PW
      const by = (Y1 - ball.y) / (Y1 - Y0) * PH
      const g = ctx.createRadialGradient(bx, by, 0, bx, by, R)
      g.addColorStop(0,   'rgba(66,114,196,0.10)')
      g.addColorStop(0.4, 'rgba(66,114,196,0.06)')
      g.addColorStop(1,   'rgba(66,114,196,0)')
      ctx.fillStyle = g
      ctx.fillRect(
        Math.max(0, bx - R), Math.max(0, by - R),
        Math.min(PW, bx + R) - Math.max(0, bx - R),
        Math.min(PH, by + R) - Math.max(0, by - R),
      )
    }

    setSrc(canvas.toDataURL('image/png'))
  }, [balls])

  return src
}

// ── Main component ───────────────────────────────────────
export default function DensityChart({
  balls = [], positions = null, parkBoundary = null,
  title = '', situation = '', stats = null,
}) {
  const densityImg = useDensityImg(balls)
  const park = stats?.home_team || ''
  const pos  = positions || {}

  const drawKeys = 'custom'    in pos ? ['custom']
    : 'with_park' in pos              ? ['with_park']
    : ['league_avg', 'no_park'].filter(k => k in pos)

  const legend = []
  if (parkBoundary)
    legend.push({ color: '#00CC55', shape: 'line', label: 'Park Boundary' })
  const nWall = stats?.n_wall_balls ?? balls.filter(b => b.is_wall_ball).length
  if (nWall > 0)
    legend.push({ color: '#FF6B00', shape: 'star', label: `Wall Ball (${nWall})` })
  for (const key of drawKeys) {
    const label = key === 'with_park'  ? `RE24 Opt (park=${park})`
      : key === 'no_park'              ? 'RE24 Opt (no park)'
      : key === 'league_avg'           ? 'League Avg'
      : 'Selected Fielders'
    legend.push({ color: STYLES[key].color, shape: STYLES[key].shape, label })
  }
  const legH = legend.length * 18 + 8

  const cbX = PL + PW + 18, cbY = PT + PH * 0.1, cbH = PH * 0.8, cbW = 16

  return (
    <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{ width: '100%', display: 'block', background: 'white' }}>
      <defs>
        <clipPath id="dc-clip">
          <rect x={PL} y={PT} width={PW} height={PH} />
        </clipPath>
        <linearGradient id="dc-blue" x1="0" y1="1" x2="0" y2="0">
          <stop offset="0%"   stopColor="white" />
          <stop offset="35%"  stopColor="#c6dbef" />
          <stop offset="65%"  stopColor="#6baed6" />
          <stop offset="100%" stopColor="#08519c" />
        </linearGradient>
      </defs>

      <rect width={SVG_W} height={SVG_H} fill="white" />
      <rect x={PL} y={PT} width={PW} height={PH} fill="white" stroke="#aaa" strokeWidth="0.8" />

      <g clipPath="url(#dc-clip)">
        {/* Canvas density heatmap */}
        {densityImg && (
          <image href={densityImg} x={PL} y={PT} width={PW} height={PH} />
        )}

        {/* 400 ft dashed arc */}
        <path
          d={`M ${tx(280).toFixed(1)},${ty(ARC_Y).toFixed(1)} A 400 400 0 0 0 ${tx(-280).toFixed(1)},${ty(ARC_Y).toFixed(1)}`}
          fill="none" stroke="gray" strokeWidth="2" strokeDasharray="8 5" />

        {/* Foul lines */}
        <line x1={tx(0)} y1={ty(0)} x2={tx(250)}  y2={ty(250)} stroke="black" strokeWidth="1.5" />
        <line x1={tx(0)} y1={ty(0)} x2={tx(-250)} y2={ty(250)} stroke="black" strokeWidth="1.5" />

        {/* Infield diamond */}
        <polyline fill="none" stroke="black" strokeWidth="2"
          points={`${tx(0)},${ty(0)} ${tx(63.64)},${ty(63.64)} ${tx(0)},${ty(127.28)} ${tx(-63.64)},${ty(63.64)} ${tx(0)},${ty(0)}`} />
        {[[0,0],[63.64,63.64],[0,127.28],[-63.64,63.64]].map(([x, y], i) => (
          <circle key={i} cx={tx(x)} cy={ty(y)} r="5" fill="white" stroke="black" strokeWidth="1.5" />
        ))}

        {/* Park boundary */}
        {parkBoundary && (
          <polyline fill="none" stroke="#00CC55" strokeWidth="2.2" opacity="0.9"
            points={parkBoundary.map(p => `${tx(p.x).toFixed(1)},${ty(p.y).toFixed(1)}`).join(' ')} />
        )}

        {/* Fielder markers */}
        {drawKeys.flatMap(key =>
          ['LF', 'CF', 'RF'].map(code => (
            <PosMarker key={`${key}-${code}`}
              cx={tx(pos[key][code].x)} cy={ty(pos[key][code].y)}
              st={STYLES[key]} code={code} />
          ))
        )}

        {/* Legend */}
        <g transform={`translate(${PL + 6},${PT + PH - legH - 8})`}>
          <rect x={-3} y={-3} width={155} height={legH + 6}
            rx="4" fill="white" fillOpacity="0.9" stroke="#ccc" strokeWidth="0.8" />
          {legend.map(({ color, shape, label }, i) => (
            <LegendItem key={i} color={color} shape={shape} label={label} y={i * 18 + 10} />
          ))}
        </g>
      </g>

      {/* Axes */}
      <line x1={PL} y1={PT + PH} x2={PL + PW} y2={PT + PH} stroke="black" strokeWidth="1" />
      <line x1={PL} y1={PT}      x2={PL}       y2={PT + PH} stroke="black" strokeWidth="1" />
      {X_TICKS.map(x => (
        <g key={x}>
          <line x1={tx(x)} y1={PT + PH} x2={tx(x)} y2={PT + PH + 5} stroke="black" strokeWidth="1" />
          <text x={tx(x)} y={PT + PH + 15} textAnchor="middle" fontSize="10" fill="#333">{x}</text>
        </g>
      ))}
      {Y_TICKS.map(y => (
        <g key={y}>
          <line x1={PL - 5} y1={ty(y)} x2={PL} y2={ty(y)} stroke="black" strokeWidth="1" />
          <text x={PL - 8} y={ty(y)} textAnchor="end" dominantBaseline="middle" fontSize="10" fill="#333">{y}</text>
        </g>
      ))}
      <text x={PL + PW / 2} y={SVG_H - 6} textAnchor="middle" fontSize="11" fill="#444">x coordinate (ft)</text>
      <text x={14} y={PT + PH / 2} textAnchor="middle" fontSize="11" fill="#444"
        transform={`rotate(-90,14,${PT + PH / 2})`}>y coordinate (ft)</text>

      {/* Density colorbar (blue scale) */}
      <rect x={cbX} y={cbY} width={cbW} height={cbH}
        fill="url(#dc-blue)" stroke="#bbb" strokeWidth="0.5" />
      <text x={cbX + cbW / 2} y={cbY - 8}
        textAnchor="middle" fontSize="9.5" fill="#444">高</text>
      <text x={cbX + cbW / 2} y={cbY + cbH + 12}
        textAnchor="middle" fontSize="9.5" fill="#444">低</text>
      <text x={cbX + cbW / 2} y={PT + PH / 2}
        textAnchor="middle" dominantBaseline="middle" fontSize="10" fill="#444"
        transform={`rotate(90,${cbX + cbW / 2},${PT + PH / 2})`}>落點密度</text>

      {/* Title */}
      {title && (
        <text x={PL + PW / 2} y={18} textAnchor="middle" fontSize="14" fontWeight="bold" fill="#111">
          {title}
        </text>
      )}
      {(situation || stats) && (
        <text x={PL + PW / 2} y={40} textAnchor="middle" fontSize="10.5" fill="#555">
          {[
            situation && `Situation: ${situation}`,
            stats?.n_balls      != null && `n = ${stats.n_balls}`,
            stats?.n_wall_balls != null && `n_wall = ${stats.n_wall_balls}`,
          ].filter(Boolean).join('   |   ')}
        </text>
      )}
    </svg>
  )
}
