import { useState } from 'react'

// ── Layout（1 SVG unit ≈ 1 ft，本壘原點，+x 朝一壘側）────────────
const PL = 52, PT = 56, PW = 560, PR = 88, PB = 40
const X0 = -170, X1 = 170, Y0 = -18, Y1 = 215
const PH = PW * (Y1 - Y0) / (X1 - X0)          // 等比例，不變形
const SVG_W = PL + PW + PR
const SVG_H = PT + PH + PB

const tx = x => PL + (x - X0) / (X1 - X0) * PW
const ty = y => PT + (Y1 - y) / (Y1 - Y0) * PH

// ── RdYlGn colormap（同 SprayChart）─────────────────────
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

// ── Marker shapes（同 SprayChart 慣例：聯盟平均=藍菱形、最佳化=紫星）──
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

const STYLES = {
  league:    { color: '#1565C0', shape: 'diamond', r: 7,  dy: -18 },
  optimized: { color: '#7B2FBE', shape: 'star',    r: 12, dy: +22 },
}

function PosMarker({ cx, cy, st, code }) {
  const { color, shape, r, dy } = st
  return (
    <g>
      {shape === 'diamond' && <polygon points={diamondPts(cx, cy, r)} fill={color} stroke="white" strokeWidth="1.5" />}
      {shape === 'star'    && <polygon points={starPts(cx, cy, r)}    fill={color} stroke="white" strokeWidth="1.2" />}
      <rect x={cx - 12} y={cy + dy - 7} width={24} height={14}
        rx="2.5" fill={color} stroke="white" strokeWidth="0.8" opacity="0.95" />
      <text x={cx} y={cy + dy} textAnchor="middle" dominantBaseline="middle"
        fill="white" fontSize="9.5" fontWeight="bold">{code}</text>
    </g>
  )
}

// ── 內野土外緣：以投手板 (0, 60.5) 為圓心、半徑 95 呎的弧（同 src/if_optimize.py）──
const MOUND_Y = 60.5, DIRT_R = 95
function dirtArcPath() {
  const pts = []
  for (let deg = -45; deg <= 45; deg += 1.5) {
    const rad = deg * Math.PI / 180
    const r = MOUND_Y * Math.cos(rad) + Math.sqrt(DIRT_R ** 2 - (MOUND_Y * Math.sin(rad)) ** 2)
    pts.push(`${tx(r * Math.sin(rad)).toFixed(1)},${ty(r * Math.cos(rad)).toFixed(1)}`)
  }
  return pts
}

const BASE_D = 8   // 壘包邊長（呎，示意）
function basePts(bx, by) {
  return `${tx(bx)},${ty(by + BASE_D / 2)} ${tx(bx + BASE_D / 2)},${ty(by)} ${tx(bx)},${ty(by - BASE_D / 2)} ${tx(bx - BASE_D / 2)},${ty(by)}`
}

const B1 = [90 * Math.SQRT1_2, 90 * Math.SQRT1_2]
const B2 = [0, 90 * Math.SQRT2]
const B3 = [-90 * Math.SQRT1_2, 90 * Math.SQRT1_2]

// 球點半徑：EV 60→110 mph 映射到 165→205 呎（土外緣外的展示帶，越強勁越深）
const ballR = ev => 165 + Math.max(0, Math.min(1, (ev - 60) / 50)) * 40

function LegendItem({ color, shape, label, y }) {
  const cx = SVG_W - PR + 14
  return (
    <g>
      {shape === 'diamond' && <polygon points={diamondPts(cx, y, 6)} fill={color} stroke="white" strokeWidth="1" />}
      {shape === 'star'    && <polygon points={starPts(cx, y, 8)}    fill={color} stroke="white" strokeWidth="1" />}
      {shape === 'circle'  && <circle cx={cx} cy={y} r={4.5} fill={color} stroke="#999" strokeWidth="0.5" />}
      <text x={cx + 12} y={y + 3.5} fontSize="9.5" fill="#555">{label}</text>
    </g>
  )
}

export default function InfieldChart({ data, colorBy, probMin = 0, probMax = 1 }) {
  const [hovered, setHovered] = useState(null)
  if (!data) return null
  const { league, optimized, balls } = data
  const pKey = colorBy === 'league' ? 'p_out_league' : 'p_out_opt'
  const shown = balls.filter(b => b[pKey] >= probMin && b[pKey] <= probMax)
  const arcPts = dirtArcPath()

  const tip = hovered ? (() => {
    const b = hovered
    const rad = b.spray_deg * Math.PI / 180
    const r = ballR(b.launch_speed)
    return {
      x: tx(r * Math.sin(rad)), y: ty(r * Math.cos(rad)),
      lines: [
        `${b.is_out ? '出局' : '安打/失誤'}　EV ${b.launch_speed.toFixed(0)} mph`,
        `P(out) 平均 ${(b.p_out_league * 100).toFixed(0)}% → 最佳化 ${(b.p_out_opt * 100).toFixed(0)}%`,
      ],
    }
  })() : null

  return (
    <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{ width: '100%', height: 'auto', display: 'block', background: 'white' }}>
      {/* 草地扇形 */}
      <path d={`M ${tx(0)} ${ty(0)} L ${tx(Y1 * Math.SQRT1_2 * 1.5)} ${ty(Y1 * 1.05)} L ${tx(0)} ${ty(Y1 * 1.4)} L ${tx(-Y1 * Math.SQRT1_2 * 1.5)} ${ty(Y1 * 1.05)} Z`}
        fill="#e8f2e4" />
      {/* 內野土（本壘到土外緣弧） */}
      <path d={`M ${tx(0)} ${ty(0)} L ${arcPts[0]} L ${arcPts.join(' L ')} Z`} fill="#e7d6bd" />
      {/* 內野草皮方塊 */}
      <polygon points={`${tx(0)},${ty(12)} ${tx(B1[0] - 8.5)},${ty(B1[1] + 3.5)} ${tx(0)},${ty(B2[1] - 5)} ${tx(B3[0] + 8.5)},${ty(B3[1] + 3.5)}`}
        fill="#d9ead3" />
      {/* 壘線（本壘→一壘/三壘延伸為邊線） */}
      <line x1={tx(0)} y1={ty(0)} x2={tx(Y1 * Math.SQRT1_2 * 1.45)} y2={ty(Y1 * 1.02)} stroke="#fff" strokeWidth="2.5" />
      <line x1={tx(0)} y1={ty(0)} x2={tx(-Y1 * Math.SQRT1_2 * 1.45)} y2={ty(Y1 * 1.02)} stroke="#fff" strokeWidth="2.5" />
      <line x1={tx(B1[0])} y1={ty(B1[1])} x2={tx(B2[0])} y2={ty(B2[1])} stroke="#fff" strokeWidth="2" />
      <line x1={tx(B3[0])} y1={ty(B3[1])} x2={tx(B2[0])} y2={ty(B2[1])} stroke="#fff" strokeWidth="2" />
      {/* 投手丘與壘包 */}
      <circle cx={tx(0)} cy={ty(MOUND_Y)} r={(tx(9) - tx(0))} fill="#dbc7a9" />
      <polygon points={basePts(...B1)} fill="white" stroke="#bbb" strokeWidth="0.8" />
      <polygon points={basePts(...B2)} fill="white" stroke="#bbb" strokeWidth="0.8" />
      <polygon points={basePts(...B3)} fill="white" stroke="#bbb" strokeWidth="0.8" />
      <polygon points={basePts(0, 0)} fill="white" stroke="#bbb" strokeWidth="0.8" />

      {/* 滾地球（沿 spray angle 的展示帶；滾地球無可靠落點座標，深度=擊球初速示意） */}
      {shown.map((b, i) => {
        const rad = b.spray_deg * Math.PI / 180
        const r = ballR(b.launch_speed)
        return (
          <circle key={i}
            cx={tx(r * Math.sin(rad))} cy={ty(r * Math.cos(rad))}
            r={hovered === b ? 6.5 : 4.5}
            fill={rdylgn(b[pKey])}
            stroke={b.is_out ? '#555' : 'white'} strokeWidth={b.is_out ? 1.2 : 0.7}
            opacity="0.88"
            onMouseEnter={() => setHovered(b)} onMouseLeave={() => setHovered(null)}
            style={{ cursor: 'pointer' }} />
        )
      })}

      {/* 站位：聯盟平均（藍菱形）vs 最佳化（紫星） */}
      {['1B', '2B', '3B', 'SS'].map(p => (
        <PosMarker key={`l-${p}`} cx={tx(league.positions[p].x)} cy={ty(league.positions[p].y)}
          st={STYLES.league} code={p} />
      ))}
      {['1B', '2B', '3B', 'SS'].map(p => (
        <PosMarker key={`o-${p}`} cx={tx(optimized.positions[p].x)} cy={ty(optimized.positions[p].y)}
          st={STYLES.optimized} code={p} />
      ))}

      {/* Legend */}
      <LegendItem color="#1565C0" shape="diamond" label="聯盟平均" y={PT + 8} />
      <LegendItem color="#7B2FBE" shape="star"    label="最佳化"   y={PT + 28} />
      <LegendItem color={rdylgn(0.9)} shape="circle" label="P(out) 高" y={PT + 52} />
      <LegendItem color={rdylgn(0.1)} shape="circle" label="P(out) 低" y={PT + 70} />
      <text x={SVG_W - PR + 8} y={PT + 92} fontSize="8.5" fill="#999">
        {shown.length}/{balls.length} 球
      </text>

      {/* Tooltip */}
      {tip && (
        <g pointerEvents="none">
          <rect x={tip.x - 92} y={tip.y - 46} width={184} height={34} rx="5"
            fill="rgba(15,23,42,0.92)" />
          {tip.lines.map((ln, i) => (
            <text key={i} x={tip.x} y={tip.y - 32 + i * 13} textAnchor="middle"
              fill="white" fontSize="9.5">{ln}</text>
          ))}
        </g>
      )}
    </svg>
  )
}
