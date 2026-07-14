import { useMemo, useState } from 'react'

// ── Layout（1 SVG unit ≈ 1 ft，本壘原點，+x 朝一壘側）────────────
// 視野涵蓋全場：內野土到外野深處（同 InfieldChart 的座標慣例，範圍放大）
const PL = 52, PT = 56, PW = 560, PR = 92, PB = 40
// Y0=-60：本壘後方界外 popup 99% 落在 -49 內（precomputed_batter_popups 實測），
// 再深的夾回下緣
const X0 = -270, X1 = 270, Y0 = -60, Y1 = 430
const PH = PW * (Y1 - Y0) / (X1 - X0)          // 等比例，不變形
const SVG_W = PL + PW + PR
const SVG_H = PT + PH + PB
const MAX_R = 415   // 超出視野的深球夾回邊緣（沿同方向）

const tx = x => PL + (x - X0) / (X1 - X0) * PW
const ty = y => PT + (Y1 - y) / (Y1 - Y0) * PH

// ── RdYlGn colormap（同 SprayChart / InfieldChart）─────────
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

// ── Marker shapes（同站慣例：最佳化=紫星；聯盟平均不畫，只留數字比較）──
function starPts(cx, cy, r) {
  return Array.from({ length: 10 }, (_, i) => {
    const a = (Math.PI * i) / 5 - Math.PI / 2
    const rad = i % 2 === 0 ? r : r * 0.38
    return `${(cx + Math.cos(a) * rad).toFixed(2)},${(cy + Math.sin(a) * rad).toFixed(2)}`
  }).join(' ')
}

const OPT_STYLE = { color: '#7B2FBE', r: 10, dy: +20 }
const OWNER_COLORS = { LF: '#4472C4', CF: '#27AE60', RF: '#E67E22', null: '#aaa' }

function PosMarker({ cx, cy, code, isActive, onClick }) {
  const { color, r, dy } = OPT_STYLE
  return (
    <g onClick={onClick} style={onClick ? { cursor: 'pointer' } : undefined}>
      {isActive && (
        <circle cx={cx} cy={cy} r={r + 8} fill={color} fillOpacity="0.15"
          stroke={color} strokeWidth="2.2" opacity="0.8" />
      )}
      <polygon points={starPts(cx, cy, r)} fill={color} stroke="white" strokeWidth="1.2" />
      <rect x={cx - 11} y={cy + dy - 6.5} width={22} height={13}
        rx="2.5" fill={color} stroke="white" strokeWidth="0.8" opacity="0.95" />
      <text x={cx} y={cy + dy} textAnchor="middle" dominantBaseline="middle"
        fill="white" fontSize="9" fontWeight="bold">{code}</text>
    </g>
  )
}

// ── 內野土外緣：投手板 (0, 60.5) 圓心、半徑 95 呎的弧（同 src/if_optimize.py）──
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

const BASE_D = 8
function basePts(bx, by) {
  return `${tx(bx)},${ty(by + BASE_D / 2)} ${tx(bx + BASE_D / 2)},${ty(by)} ${tx(bx)},${ty(by - BASE_D / 2)} ${tx(bx - BASE_D / 2)},${ty(by)}`
}

const B1 = [90 * Math.SQRT1_2, 90 * Math.SQRT1_2]
const B2 = [0, 90 * Math.SQRT2]
const B3 = [-90 * Math.SQRT1_2, 90 * Math.SQRT1_2]

// 球點畫在 Statcast 記錄座標；太深的球沿同方向夾回視野邊緣，
// 本壘後方過深的界外球夾回下緣
function clampXY(x, y) {
  const r = Math.hypot(x, y)
  const k = r > MAX_R ? MAX_R / r : 1
  return [x * k, Math.max(y * k, Y0 + 6)]
}

function LegendItem({ color, shape, label, y }) {
  const cx = SVG_W - PR + 14
  return (
    <g>
      {shape === 'star'   && <polygon points={starPts(cx, y, 8)} fill={color} stroke="white" strokeWidth="1" />}
      {shape === 'circle' && <circle cx={cx} cy={y} r={4.5} fill={color} stroke="#999" strokeWidth="0.5" />}
      <text x={cx + 12} y={y + 3.5} fontSize="9.5" fill="#555">{label}</text>
    </g>
  )
}

const OF_POSITIONS = ['LF', 'CF', 'RF']
const IF_POSITIONS = ['1B', '2B', '3B', 'SS']

// popup 接殺機率＝聯盟實證常數（2025 例行賽 98.5% 出局；站哪都接得到，
// 所以不參與優化）。外野模型算 popup 是 OOD、校準比常數差，勿改回模型算。
const POPUP_CATCH = 0.985

export default function IntegratedChart({ data }) {
  const [hovered, setHovered] = useState(null)   // { kind: 'of'|'if'|'popup', ball }
  const [activePos, setActivePos] = useState(null)   // 'LF'|'CF'|'RF'|null
  const [colorMode, setColorMode] = useState('prob') // 'prob' | 'owner'
  const [probMin, setProbMin] = useState(0)          // 0–100 integer
  const [probMax, setProbMax] = useState(100)

  // 外野責任歸屬：距最佳化站位最近者（同外野主頁的前端 fallback 演算法）
  const ofPositions = data?.optimized?.positions
  const of_balls_all = data?.of_balls
  const ballOwner = useMemo(() => {
    if (!ofPositions || !of_balls_all) return null
    return of_balls_all.map(b => {
      if (b.is_wall_ball || b.catch_prob < 0.05) return null
      return OF_POSITIONS.reduce((best, code) => {
        const dx = b.x - ofPositions[code].x, dy = b.y - ofPositions[code].y
        const d = dx * dx + dy * dy
        return d < best.d ? { code, d } : best
      }, { code: null, d: Infinity }).code
    })
  }, [ofPositions, of_balls_all])

  if (!data) return null
  const { optimized, of_balls, if_balls, popup_balls = [], park_boundary = null } = data
  const nWall = data.stats?.n_wall_balls ?? 0
  const arcPts = dirtArcPath()
  const inRange = (p) => {
    const pct = p * 100
    return pct >= probMin && pct <= probMax
  }

  const tip = hovered ? (() => {
    const { kind, ball } = hovered
    const [bx, by] = clampXY(ball.x, ball.y)
    const lines = kind === 'of'
      ? [`外野球　接殺機率 ${(ball.catch_prob * 100).toFixed(0)}%`
         + (ballOwner?.[of_balls.indexOf(ball)] ? `　歸屬 ${ballOwner[of_balls.indexOf(ball)]}` : '')]
      : kind === 'popup'
      ? [
          `內野高飛　${ball.is_out ? '出局' : '安打/失誤'}　接殺機率 ~99%（實證）`,
          '站哪都接得到，不參與站位優化',
        ]
      : [
          `滾地球　${ball.is_out ? '出局' : '安打/失誤'}　EV ${ball.launch_speed.toFixed(0)} mph`,
          `P(out) 平均 ${(ball.p_out_league * 100).toFixed(0)}% → 最佳化 ${(ball.p_out_opt * 100).toFixed(0)}%`,
        ]
    return { x: tx(bx), y: ty(by), lines }
  })() : null

  // 責任歸屬模式下，內野球/高飛淡出讓外野歸屬色突出
  const dimNonOf = colorMode === 'owner' ? 0.18 : null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', background: 'white' }}>
      {/* ── Controls（同外野主頁：歸屬色切換＋機率範圍）── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '16px',
        padding: '6px 12px', borderBottom: '1px solid var(--slate-200)', flexWrap: 'wrap',
      }}>
        <button onClick={() => { setColorMode(m => m === 'prob' ? 'owner' : 'prob'); setActivePos(null) }} style={{
          padding: '3px 10px', borderRadius: '4px', cursor: 'pointer', fontSize: '12px',
          border: '1px solid #cbd5e1',
          background: colorMode === 'owner' ? '#1e40af' : 'white',
          color: colorMode === 'owner' ? 'white' : '#334155',
        }}>
          {colorMode === 'prob' ? '切換：外野責任歸屬色' : '切換：接殺機率色'}
        </button>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#475569' }}>
          <span>機率範圍</span>
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

    <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{ width: '100%', height: 'auto', display: 'block', background: 'white' }}>
      <defs>
        <linearGradient id="ic-grad" x1="0" y1="1" x2="0" y2="0">
          {RDYLGN.map(([t, [r, g, b]]) => (
            <stop key={t} offset={`${t * 100}%`} stopColor={`rgb(${r},${g},${b})`} />
          ))}
        </linearGradient>
      </defs>
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

      {/* 內野高飛（展示用，不參與優化——顏色＝實證常數接殺機率，畫在最底層） */}
      {popup_balls.filter(() => inRange(POPUP_CATCH)).map((b, i) => {
        const [bx, by] = clampXY(b.x, b.y)
        const isHov = hovered && hovered.kind === 'popup' && hovered.ball === b
        return (
          <circle key={`pu-${i}`}
            cx={tx(bx)} cy={ty(by)}
            r={isHov ? 6 : 3.5}
            fill={rdylgn(POPUP_CATCH)}
            stroke={b.is_out ? '#555' : 'white'} strokeWidth={b.is_out ? 0.9 : 0.6}
            opacity={dimNonOf ?? 0.85}
            onMouseEnter={() => setHovered({ kind: 'popup', ball: b })}
            onMouseLeave={() => setHovered(null)}
            style={{ cursor: 'pointer' }} />
        )
      })}
      {/* 滾地球（顏色 = 最佳化站位下的 P(out)） */}
      {if_balls.filter(b => inRange(b.p_out_opt)).map((b, i) => {
        const [bx, by] = clampXY(b.x, b.y)
        const isHov = hovered && hovered.kind === 'if' && hovered.ball === b
        return (
          <circle key={`if-${i}`}
            cx={tx(bx)} cy={ty(by)}
            r={isHov ? 6 : 4}
            fill={rdylgn(b.p_out_opt)}
            stroke={b.is_out ? '#555' : 'white'} strokeWidth={b.is_out ? 1.1 : 0.6}
            opacity={dimNonOf ?? 0.85}
            onMouseEnter={() => setHovered({ kind: 'if', ball: b })}
            onMouseLeave={() => setHovered(null)}
            style={{ cursor: 'pointer' }} />
        )
      })}
      {/* 外野球（顏色 = 接殺機率或責任歸屬；打牆球另畫橘星） */}
      {of_balls.map((b, i) => {
        if (b.is_wall_ball || !inRange(b.catch_prob)) return null
        const [bx, by] = clampXY(b.x, b.y)
        const isHov = hovered && hovered.kind === 'of' && hovered.ball === b
        const owner = ballOwner?.[i]
        const mine = !activePos || owner === activePos
        const fill = colorMode === 'owner'
          ? (OWNER_COLORS[owner] ?? OWNER_COLORS.null)
          : rdylgn(b.catch_prob)
        return (
          <circle key={`of-${i}`}
            cx={tx(bx)} cy={ty(by)}
            r={isHov ? 6 : 4}
            fill={fill}
            fillOpacity={mine ? 0.88 : 0.10}
            stroke={mine ? 'white' : 'gray'} strokeWidth={mine ? 0.6 : 0.2}
            onMouseEnter={() => setHovered({ kind: 'of', ball: b })}
            onMouseLeave={() => setHovered(null)}
            style={{ cursor: 'pointer' }} />
        )
      })}
      {/* 球場牆線與打牆球（同外野主頁慣例：綠線＋橘星） */}
      {park_boundary && (
        <polyline fill="none" stroke="#00CC55" strokeWidth="2.2" opacity="0.9"
          points={park_boundary.map(p => `${tx(p.x).toFixed(1)},${ty(p.y).toFixed(1)}`).join(' ')} />
      )}
      {of_balls.filter(b => b.is_wall_ball).map((b, i) => {
        const [bx, by] = clampXY(b.x, b.y)
        return (
          <polygon key={`wb-${i}`} points={starPts(tx(bx), ty(by), 7)}
            fill="#FF6B00" stroke="black" strokeWidth="0.4" opacity="0.9" />
        )
      })}

      {/* 七人站位（最佳化紫星；外野三人可點擊看責任歸屬） */}
      {OF_POSITIONS.map(p => (
        <PosMarker key={`o-${p}`} cx={tx(optimized.positions[p].x)} cy={ty(optimized.positions[p].y)}
          code={p} isActive={activePos === p}
          onClick={() => setActivePos(a => a === p ? null : p)} />
      ))}
      {IF_POSITIONS.map(p => (
        <PosMarker key={`o-${p}`} cx={tx(optimized.positions[p].x)} cy={ty(optimized.positions[p].y)}
          code={p} />
      ))}

      {/* Legend */}
      <LegendItem color="#7B2FBE" shape="star" label="最佳化站位" y={PT + 8} />
      {colorMode === 'prob' ? (() => {
        // 接殺機率色階條
        const cbX = SVG_W - PR + 10, cbY = PT + 30, cbW = 12, cbH = 96
        return (
          <g>
            <rect x={cbX} y={cbY} width={cbW} height={cbH}
              fill="url(#ic-grad)" stroke="#bbb" strokeWidth="0.5" />
            {[0, 0.5, 1].map(t => (
              <g key={t}>
                <line x1={cbX + cbW} y1={cbY + cbH * (1 - t)} x2={cbX + cbW + 4}
                  y2={cbY + cbH * (1 - t)} stroke="#666" strokeWidth="0.8" />
                <text x={cbX + cbW + 6} y={cbY + cbH * (1 - t) + 3} fontSize="8.5" fill="#555">
                  {(t * 100).toFixed(0)}%
                </text>
              </g>
            ))}
            <text x={cbX} y={cbY + cbH + 14} fontSize="8.5" fill="#555">接殺/出局機率</text>
          </g>
        )
      })() : (() => {
        // 責任歸屬色說明
        return (
          <g>
            {[['LF', OWNER_COLORS.LF], ['CF', OWNER_COLORS.CF], ['RF', OWNER_COLORS.RF],
              ['其他', OWNER_COLORS.null]].map(([label, color], i) => (
              <g key={label} transform={`translate(${SVG_W - PR + 10},${PT + 30 + i * 20})`}>
                <rect width={12} height={12} rx="2.5" fill={color} />
                <text x={17} y={9.5} fontSize="9" fill="#555">{label}</text>
              </g>
            ))}
            <text x={SVG_W - PR + 10} y={PT + 124} fontSize="8" fill="#aaa">
              <tspan x={SVG_W - PR + 10} dy="0">點外野星標</tspan>
              <tspan x={SVG_W - PR + 10} dy="11">高亮其責任球</tspan>
            </text>
          </g>
        )
      })()}
      {popup_balls.length > 0 && colorMode === 'prob' &&
        <LegendItem color={rdylgn(POPUP_CATCH)} shape="circle" label="內野高飛" y={PT + 152} />}
      {nWall > 0 &&
        <LegendItem color="#FF6B00" shape="star" label={`打牆球 (${nWall})`} y={PT + 172} />}
      <text x={SVG_W - PR + 8} y={PT + 196} fontSize="8.5" fill="#999">
        <tspan x={SVG_W - PR + 8} dy="0">外野 {of_balls.length} 球</tspan>
        <tspan x={SVG_W - PR + 8} dy="12">滾地 {if_balls.length} 球</tspan>
        {popup_balls.length > 0 &&
          <tspan x={SVG_W - PR + 8} dy="12">高飛 {popup_balls.length} 球</tspan>}
      </text>
      <text x={SVG_W - PR + 8} y={PT + 244} fontSize="8" fill="#aaa">
        <tspan x={SVG_W - PR + 8} dy="0">球點＝紀錄座標</tspan>
        <tspan x={SVG_W - PR + 8} dy="11">（出局≈處理位置</tspan>
        <tspan x={SVG_W - PR + 8} dy="11">　安打≈撿球位置）</tspan>
        {popup_balls.length > 0 && <>
          <tspan x={SVG_W - PR + 8} dy="14">高飛≈99% 接殺</tspan>
          <tspan x={SVG_W - PR + 8} dy="11">（聯盟實證），</tspan>
          <tspan x={SVG_W - PR + 8} dy="11">不參與站位優化</tspan>
        </>}
      </text>

      {/* Tooltip */}
      {tip && (
        <g pointerEvents="none">
          <rect x={tip.x - 96} y={tip.y - 34 - tip.lines.length * 13} width={192}
            height={8 + tip.lines.length * 13} rx="5" fill="rgba(15,23,42,0.92)" />
          {tip.lines.map((ln, i) => (
            <text key={i} x={tip.x} y={tip.y - 26 - (tip.lines.length - 1 - i) * 13}
              textAnchor="middle" fill="white" fontSize="9.5">{ln}</text>
          ))}
        </g>
      )}
    </svg>
    </div>
  )
}
