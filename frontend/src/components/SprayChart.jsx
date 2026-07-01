import { useState, useMemo } from 'react'

// ── Layout (1 SVG unit ≈ 1 ft; same viewport as matplotlib xlim/ylim) ──
const PL = 52, PT = 62, PW = 560, PH = 460, PR = 88, PB = 44
const SVG_W = PL + PW + PR   // 700
const SVG_H = PT + PH + PB   // 566
const X0 = -280, X1 = 280, Y0 = -10, Y1 = 450

const tx = x => PL + (x - X0) / (X1 - X0) * PW
const ty = y => PT + (Y1 - y) / (Y1 - Y0) * PH


// ── RdYlGn colormap (matching matplotlib) ──────────────
const RDYLGN = [
  [0.00, [165,   0,  38]],
  [0.10, [215,  48,  39]],
  [0.25, [244, 109,  67]],
  [0.40, [253, 174,  97]],
  [0.50, [255, 255, 191]],
  [0.60, [217, 239, 139]],
  [0.75, [166, 217, 106]],
  [0.90, [102, 189,  99]],
  [1.00, [  0, 104,  55]],
]

function rdylgn(p) {
  p = Math.max(0, Math.min(1, p))
  for (let i = 0; i < RDYLGN.length - 1; i++) {
    const [t0, c0] = RDYLGN[i], [t1, c1] = RDYLGN[i + 1]
    if (p <= t1) {
      const t = (p - t0) / (t1 - t0)
      return `rgb(${c0.map((v, j) => Math.round(v + (c1[j] - v) * t)).join(',')})`
    }
  }
  return 'rgb(0,104,55)'
}

// ── Marker shapes ───────────────────────────────────────
function starPts(cx, cy, r) {
  return Array.from({ length: 10 }, (_, i) => {
    const a = (Math.PI * i) / 5 - Math.PI / 2
    const rad = i % 2 === 0 ? r : r * 0.38
    return `${(cx + Math.cos(a) * rad).toFixed(2)},${(cy + Math.sin(a) * rad).toFixed(2)}`
  }).join(' ')
}

function diamondPts(cx, cy, r) {
  return `${cx},${cy - r} ${cx + r * 0.7},${cy} ${cx},${cy + r} ${cx - r * 0.7},${cy}`
}

// ── Marker styles (colours matching matplotlib plot.py) ─
const STYLES = {
  league_avg: { color: '#1565C0', shape: 'diamond', r: 7,  dy: -20 },
  no_park:    { color: '#C0392B', shape: 'circle',  r: 8,  dy: -24 },
  with_park:  { color: '#7B2FBE', shape: 'star',    r: 12, dy: +22 },
  custom:     { color: '#7B2FBE', shape: 'star',    r: 12, dy: +22 },
}

const OWNER_COLORS = { LF: '#4472C4', CF: '#27AE60', RF: '#E67E22', null: '#aaa' }

function PosMarker({ cx, cy, st, code, isActive, onClick }) {
  const { color, shape, r, dy } = st
  return (
    <g onClick={onClick} style={{ cursor: 'pointer' }}>
      {isActive && (
        <circle cx={cx} cy={cy} r={r + 10} fill={color} fillOpacity="0.15"
          stroke={color} strokeWidth="2.5" opacity="0.8" />
      )}
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

// ── 400 ft arc (= matplotlib Arc((0,0),800,800,theta1=45,theta2=135)) ──
// Endpoints at x=±280: y = sqrt(400²-280²) ≈ 285.7
const ARC_Y = Math.sqrt(400 * 400 - 280 * 280)

const X_TICKS = [-200, -100, 0, 100, 200]
const Y_TICKS = [0, 100, 200, 300, 400]

// ── Legend item ─────────────────────────────────────────
function LegendItem({ color, shape, label, y }) {
  return (
    <g>
      {shape === 'line'    && <line     x1={2}  y1={y}   x2={18}  y2={y}   stroke={color} strokeWidth="2.2" />}
      {shape === 'star'    && <polygon  points={starPts(10, y, 6)}           fill={color} stroke="black" strokeWidth="0.4" />}
      {shape === 'diamond' && <polygon  points={diamondPts(10, y, 6)}        fill={color} stroke="white" strokeWidth="1" />}
      {shape === 'circle'  && <circle   cx={10}  cy={y}  r={5}               fill={color} stroke="white" strokeWidth="1" />}
      <text x={24} y={y} dominantBaseline="middle" fontSize="9.5" fill="#333">{label}</text>
    </g>
  )
}

export default function SprayChart({
  balls = [], positions = null, parkBoundary = null,
  title = '', situation = '', stats = null,
}) {
  const [hovered, setHovered] = useState(null)
  const [activePos, setActivePos] = useState(null)
  const [colorMode, setColorMode] = useState('prob')   // 'prob' | 'owner'
  const [probMin, setProbMin] = useState(0)             // 0–100 integer
  const [probMax, setProbMax] = useState(100)

  const park = stats?.home_team || ''
  const pos  = positions || {}

  // Which marker sets to draw (matches matplotlib logic)
  const drawKeys = 'custom'    in pos ? ['custom']
    : 'with_park' in pos              ? ['with_park']
    : ['league_avg', 'no_park'].filter(k => k in pos)

  // Fallback: compute responsible fielder client-side if backend doesn't provide it
  const ballOwner = useMemo(() => {
    if (balls.every(b => b.responsible !== undefined)) return null  // backend has it
    const key = drawKeys[0]
    if (!key || !pos[key]) return null
    const posSet = pos[key]
    return balls.map(b => {
      if (b.is_wall_ball || b.catch_prob < 0.05) return null
      return ['LF', 'CF', 'RF'].reduce((best, code) => {
        const dx = b.x - posSet[code].x, dy = b.y - posSet[code].y
        const d = dx * dx + dy * dy
        return d < best.d ? { code, d } : best
      }, { code: null, d: Infinity }).code
    })
  }, [balls, pos, drawKeys])


  // Legend
  const legend = []
  if (parkBoundary)
    legend.push({ color: '#00CC55', shape: 'line',  label: 'Park Boundary' })
  const nWall = stats?.n_wall_balls ?? balls.filter(b => b.is_wall_ball).length
  if (nWall > 0)
    legend.push({ color: '#FF6B00', shape: 'star',  label: `Wall Ball (${nWall})` })
  for (const key of drawKeys) {
    const label = key === 'with_park'  ? `RE24 Opt (park=${park})`
      : key === 'no_park'              ? 'RE24 Opt (no park)'
      : key === 'league_avg'           ? 'League Avg'
      : 'Selected Fielders'
    legend.push({ color: STYLES[key].color, shape: STYLES[key].shape, label })
  }

  // Tooltip position
  const tip = hovered ? (() => {
    const bx = tx(hovered.x), by = ty(hovered.y)
    const tipX = Math.max(PL + 70, Math.min(PL + PW - 70, bx))
    const tipY = (by - PT) > 55 ? by - 55 : by + 15
    return { tipX, tipY }
  })() : null

  const legH = legend.length * 18 + 8

  return (
    <div style={{ display: 'flex', flexDirection: 'column', background: 'white' }}>
      {/* ── Controls ── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '16px',
        padding: '6px 12px', borderBottom: '1px solid #e2e8f0', flexWrap: 'wrap',
      }}>
        {/* Color mode toggle */}
        <button onClick={() => setColorMode(m => m === 'prob' ? 'owner' : 'prob')} style={{
          padding: '3px 10px', borderRadius: '4px', cursor: 'pointer', fontSize: '12px',
          border: '1px solid #cbd5e1',
          background: colorMode === 'owner' ? '#1e40af' : 'white',
          color: colorMode === 'owner' ? 'white' : '#334155',
        }}>
          {colorMode === 'prob' ? '切換：責任歸屬色' : '切換：接殺機率色'}
        </button>

        {/* Prob range slider */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#475569' }}>
          <span>接殺機率</span>
          <span style={{ minWidth: '28px', textAlign: 'right' }}>{probMin}%</span>
          <input type="range" min={0} max={100} value={probMin}
            onChange={e => setProbMin(Math.min(+e.target.value, probMax))}
            style={{ width: '72px', accentColor: '#4472C4' }} />
          <span>—</span>
          <input type="range" min={0} max={100} value={probMax}
            onChange={e => setProbMax(Math.max(+e.target.value, probMin))}
            style={{ width: '72px', accentColor: '#4472C4' }} />
          <span style={{ minWidth: '28px' }}>{probMax}%</span>
        </div>
      </div>

    <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{ width: '100%', display: 'block', background: 'white' }}>
      <defs>
        <clipPath id="sc-clip">
          <rect x={PL} y={PT} width={PW} height={PH} />
        </clipPath>
        {/* Colorbar gradient */}
        <linearGradient id="sc-grad" x1="0" y1="1" x2="0" y2="0">
          {RDYLGN.map(([t, [r, g, b]]) => (
            <stop key={t} offset={`${t * 100}%`} stopColor={`rgb(${r},${g},${b})`} />
          ))}
        </linearGradient>
      </defs>

      {/* White background + plot area border */}
      <rect width={SVG_W} height={SVG_H} fill="white" />
      <rect x={PL} y={PT} width={PW} height={PH} fill="white" stroke="#aaa" strokeWidth="0.8" />

      <g clipPath="url(#sc-clip)">

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

        {/* Ball scatter — color by catch prob or responsibility; filter by prob range */}
        {balls.map((b, i) => {
          if (b.is_wall_ball) return null
          const pct = b.catch_prob * 100
          if (pct < probMin || pct > probMax) return null
          const owner = b.responsible !== undefined ? b.responsible : ballOwner?.[i]
          const mine = !activePos || owner === activePos
          const fill = colorMode === 'owner'
            ? (OWNER_COLORS[owner] ?? OWNER_COLORS.null)
            : rdylgn(b.catch_prob)
          return (
            <circle key={i} cx={tx(b.x)} cy={ty(b.y)}
              r={mine ? 6 : 5}
              fill={fill}
              fillOpacity={mine ? 0.92 : 0.10}
              stroke={mine ? 'white' : 'gray'}
              strokeWidth={mine ? 0.8 : 0.2}
              style={{ cursor: 'crosshair' }}
              onMouseEnter={() => setHovered(b)}
              onMouseLeave={() => setHovered(null)} />
          )
        })}

        {/* Wall balls (orange stars) */}
        {balls.filter(b => b.is_wall_ball).map((b, i) => (
          <polygon key={i} points={starPts(tx(b.x), ty(b.y), 7)}
            fill="#FF6B00" stroke="black" strokeWidth="0.4" opacity="0.9" />
        ))}

        {/* Fielder position markers (click to highlight responsible balls) */}
        {drawKeys.flatMap(key =>
          ['LF', 'CF', 'RF'].map(code => (
            <PosMarker key={`${key}-${code}`}
              cx={tx(pos[key][code].x)} cy={ty(pos[key][code].y)}
              st={STYLES[key]} code={code}
              isActive={activePos === code}
              onClick={() => setActivePos(p => p === code ? null : code)} />
          ))
        )}

        {/* Hover tooltip */}
        {hovered && tip && (
          <g pointerEvents="none">
            <rect x={tip.tipX - 66} y={tip.tipY} width={132} height={40}
              rx="4" fill="rgba(15,23,42,0.9)" />
            <text x={tip.tipX} y={tip.tipY + 14} textAnchor="middle"
              fill="white" fontSize="13" fontWeight="700">
              接殺機率 {(hovered.catch_prob * 100).toFixed(0)}%
            </text>
            <text x={tip.tipX} y={tip.tipY + 29} textAnchor="middle" fill="#94a3b8" fontSize="11">
              落點 ({Math.round(hovered.x)}, {Math.round(hovered.y)}) ft
            </text>
          </g>
        )}

        {/* Legend (lower-left, inside plot) */}
        <g transform={`translate(${PL + 6},${PT + PH - legH - 8})`}>
          <rect x={-3} y={-3} width={155} height={legH + 6}
            rx="4" fill="white" fillOpacity="0.9" stroke="#ccc" strokeWidth="0.8" />
          {legend.map(({ color, shape, label }, i) => (
            <LegendItem key={i} color={color} shape={shape} label={label} y={i * 18 + 10} />
          ))}
        </g>
      </g>

      {/* ── Axes ── */}
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
      <text x={PL + PW / 2} y={SVG_H - 6} textAnchor="middle" fontSize="11" fill="#444">
        x coordinate (ft)
      </text>
      <text x={14} y={PT + PH / 2} textAnchor="middle" fontSize="11" fill="#444"
        transform={`rotate(-90,14,${PT + PH / 2})`}>
        y coordinate (ft)
      </text>

      {/* ── Colorbar / Owner legend ── */}
      {(() => {
        const cbX = PL + PW + 18, cbY = PT + PH * 0.1, cbH = PH * 0.8, cbW = 16
        if (colorMode === 'prob') return (
          <>
            <rect x={cbX} y={cbY} width={cbW} height={cbH}
              fill="url(#sc-grad)" stroke="#bbb" strokeWidth="0.5" />
            {[0, 0.25, 0.5, 0.75, 1].map(t => {
              const cy = cbY + cbH * (1 - t)
              return (
                <g key={t}>
                  <line x1={cbX + cbW} y1={cy} x2={cbX + cbW + 5} y2={cy} stroke="#333" strokeWidth="1" />
                  <text x={cbX + cbW + 8} y={cy} dominantBaseline="middle" fontSize="9.5" fill="#444">
                    {(t * 100).toFixed(0)}%
                  </text>
                </g>
              )
            })}
            <text x={cbX + cbW / 2} y={PT + PH / 2} textAnchor="middle" dominantBaseline="middle"
              fontSize="10" fill="#444"
              transform={`rotate(90,${cbX + cbW / 2},${PT + PH / 2})`}>
              Catch Probability
            </text>
          </>
        )
        return (
          <>
            {[['LF', '#4472C4'], ['CF', '#27AE60'], ['RF', '#E67E22'], ['其他', '#aaa']].map(([label, color], i) => (
              <g key={label} transform={`translate(${cbX},${cbY + i * 26})`}>
                <rect width={16} height={16} rx="3" fill={color} />
                <text x={22} y={8} dominantBaseline="middle" fontSize="11" fill="#333">{label}</text>
              </g>
            ))}
            <text x={cbX + cbW / 2} y={cbY + 4 * 26 + 8} textAnchor="middle" fontSize="10" fill="#444">責任</text>
          </>
        )
      })()}

      {/* ── Title ── */}
      {title && (
        <text x={PL + PW / 2} y={18} textAnchor="middle" fontSize="14" fontWeight="bold" fill="#111">
          {title}
        </text>
      )}
      {(situation || stats) && (
        <text x={PL + PW / 2} y={40} textAnchor="middle" fontSize="10.5" fill="#555">
          {[
            situation && `Situation: ${situation}`,
            stats?.n_balls     != null && `n = ${stats.n_balls}`,
            stats?.n_wall_balls != null && `n_wall = ${stats.n_wall_balls}`,
          ].filter(Boolean).join('   |   ')}
        </text>
      )}
    </svg>
    </div>
  )
}
