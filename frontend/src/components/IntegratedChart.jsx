import { useEffect, useMemo, useState } from 'react'

// ── Layout（1 SVG unit ≈ 1 ft，本壘原點，+x 朝一壘側）────────────
// 視野涵蓋全場：內野土到外野深處（同 InfieldChart 的座標慣例，範圍放大）。
// 邊界縮到最小讓球場本身佔滿版面（圖例/色階條疊在圖內右下角）
const PL = 10, PT = 12, PW = 560, PR = 12, PB = 10
// Y0=-60：本壘後方界外 popup 99% 落在 -49 內（precomputed_batter_popups 實測），
// 再深的夾回下緣；Y1=425 給 400 呎弧與最深牆線（~420）留邊
const X0 = -262, X1 = 262, Y0 = -60, Y1 = 425
const PH = PW * (Y1 - Y0) / (X1 - X0)          // 等比例，不變形
const SVG_W = PL + PW + PR
const SVG_H = PT + PH + PB
const MAX_R = 412   // 超出視野的深球夾回邊緣（沿同方向）

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
// 責任歸屬色：外野同外野主頁；內野另配四色（避開星標紫與場地綠）
const OWNER_COLORS = {
  LF: '#4472C4', CF: '#27AE60', RF: '#E67E22',
  '1B': '#D81B60', '2B': '#00ACC1', '3B': '#6D4C41', SS: '#F9A825',
  null: '#aaa',
}

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
// 界外過深/過寬的球夾回視野內
function clampXY(x, y) {
  const r = Math.hypot(x, y)
  const k = r > MAX_R ? MAX_R / r : 1
  return [Math.max(X0 + 6, Math.min(X1 - 6, x * k)),
          Math.max(y * k, Y0 + 6)]
}

const OF_POSITIONS = ['LF', 'CF', 'RF']
const IF_POSITIONS = ['1B', '2B', '3B', 'SS']

// ── Marching squares：從網格 KDE 取一條等值線的線段集（SVG path 字串）──
function marchingSquares(grid, gw, gh, level, cell) {
  const parts = []
  const val = (ix, iy) => grid[iy * gw + ix]
  for (let iy = 0; iy < gh - 1; iy++) {
    for (let ix = 0; ix < gw - 1; ix++) {
      const tl = val(ix, iy), tr = val(ix + 1, iy)
      const br = val(ix + 1, iy + 1), bl = val(ix, iy + 1)
      let c = 0
      if (tl >= level) c |= 8
      if (tr >= level) c |= 4
      if (br >= level) c |= 2
      if (bl >= level) c |= 1
      if (c === 0 || c === 15) continue
      const x = ix * cell, y = iy * cell
      const t = (a, b) => (level - a) / (b - a)
      const top    = [x + cell * t(tl, tr), y]
      const right  = [x + cell, y + cell * t(tr, br)]
      const bottom = [x + cell * t(bl, br), y + cell]
      const left   = [x, y + cell * t(tl, bl)]
      const SEGS = {
        1: [[left, bottom]], 2: [[bottom, right]], 3: [[left, right]],
        4: [[top, right]], 5: [[top, right], [left, bottom]],
        6: [[top, bottom]], 7: [[left, top]], 8: [[left, top]],
        9: [[top, bottom]], 10: [[top, left], [bottom, right]],
        11: [[top, right]], 12: [[left, right]], 13: [[bottom, right]],
        14: [[left, bottom]],
      }
      for (const [[x1, y1], [x2, y2]] of SEGS[c])
        parts.push(`M${x1.toFixed(1)},${y1.toFixed(1)}L${x2.toFixed(1)},${y2.toFixed(1)}`)
    }
  }
  return parts.join('')
}

// popup 接殺機率＝聯盟實證常數（2025 例行賽 98.5% 出局；站哪都接得到，
// 所以不參與優化）。外野模型算 popup 是 OOD、校準比常數差，勿改回模型算。
const POPUP_CATCH = 0.985

export default function IntegratedChart({ data }) {
  const [hovered, setHovered] = useState(null)   // { kind: 'of'|'if'|'popup', ball }
  const [activePos, setActivePos] = useState(null)   // 七位置之一或 null
  const [colorMode, setColorMode] = useState('prob') // 'prob' | 'owner'
  const [probMin, setProbMin] = useState(0)          // 0–100 integer
  const [probMax, setProbMax] = useState(100)
  // 球種篩選（複選）：滾地→內野球、飛球/平飛→外野球（真實 bb_type 標籤）、高飛→popup
  const [showTypes, setShowTypes] = useState(
    { ground_ball: true, fly_ball: true, line_drive: true, popup: true })
  // 落點密度模式：把目前可見的球（球種勾選＋機率範圍過濾後）算成網格 KDE，
  // 同一份網格同時產生藍色填色層與等高線（同外野頁 matplotlib KDE 的呈現）
  const [showDensity, setShowDensity] = useState(false)
  const [densitySrc, setDensitySrc] = useState(null)
  const [contours, setContours] = useState(null)

  useEffect(() => {
    if (!showDensity || !data) { setDensitySrc(null); setContours(null); return }
    const within = (p) => p * 100 >= probMin && p * 100 <= probMax
    const pts = []
    if (showTypes.ground_ball)
      for (const b of data.if_balls) if (within(b.p_out_opt)) pts.push(clampXY(b.x, b.y))
    for (const b of data.of_balls) {
      if (b.is_wall_ball || (b.bb_type && !showTypes[b.bb_type]) || !within(b.catch_prob)) continue
      pts.push(clampXY(b.x, b.y))
    }
    if (showTypes.popup && within(POPUP_CATCH))
      for (const b of (data.popup_balls || [])) pts.push(clampXY(b.x, b.y))
    if (pts.length === 0) { setDensitySrc(null); setContours(null); return }

    // 網格 KDE（高斯核）
    const CELL = 4, SIGMA = 3.4                     // 格 4px、頻寬 ~14px
    const GW = Math.ceil(PW / CELL) + 1
    const GH = Math.ceil(PH / CELL) + 1
    const grid = new Float32Array(GW * GH)
    const win = Math.ceil(SIGMA * 3)
    for (const [x, y] of pts) {
      const gx = ((x - X0) / (X1 - X0) * PW) / CELL
      const gy = ((Y1 - y) / (Y1 - Y0) * PH) / CELL
      const ix0 = Math.max(0, Math.floor(gx - win)), ix1 = Math.min(GW - 1, Math.ceil(gx + win))
      const iy0 = Math.max(0, Math.floor(gy - win)), iy1 = Math.min(GH - 1, Math.ceil(gy + win))
      for (let iy = iy0; iy <= iy1; iy++)
        for (let ix = ix0; ix <= ix1; ix++) {
          const d2 = (ix - gx) ** 2 + (iy - gy) ** 2
          grid[iy * GW + ix] += Math.exp(-d2 / (2 * SIGMA * SIGMA))
        }
    }
    let maxV = 0
    for (let i = 0; i < grid.length; i++) if (grid[i] > maxV) maxV = grid[i]

    // 填色層：小畫布逐格上 alpha，再平滑放大
    const small = document.createElement('canvas')
    small.width = GW
    small.height = GH
    const sctx = small.getContext('2d')
    const img = sctx.createImageData(GW, GH)
    for (let i = 0; i < grid.length; i++) {
      const v = grid[i] / maxV
      img.data[i * 4] = 30
      img.data[i * 4 + 1] = 64
      img.data[i * 4 + 2] = 175
      img.data[i * 4 + 3] = Math.round(255 * 0.5 * Math.pow(v, 0.75))
    }
    sctx.putImageData(img, 0, 0)
    const big = document.createElement('canvas')
    big.width = PW
    big.height = PH
    const bctx = big.getContext('2d')
    bctx.imageSmoothingEnabled = true
    bctx.imageSmoothingQuality = 'high'
    bctx.drawImage(small, 0, 0, PW, PH)
    setDensitySrc(big.toDataURL('image/png'))

    // 等高線：同一份網格取 5 條等值線
    const levels = [0.15, 0.3, 0.45, 0.6, 0.8]
    setContours(levels.map(t => marchingSquares(grid, GW, GH, t * maxV, CELL)))
  }, [showDensity, data, showTypes, probMin, probMax])

  // 責任歸屬：距最佳化站位最近者（同外野主頁的前端 fallback 演算法）。
  // 外野球在 LF/CF/RF 之間分、滾地球在 1B/2B/3B/SS 之間分（球種已定守備側）
  const optPositions = data?.optimized?.positions
  const of_balls_all = data?.of_balls
  const if_balls_all = data?.if_balls
  const nearestOf = (b, codes) => codes.reduce((best, code) => {
    const dx = b.x - optPositions[code].x, dy = b.y - optPositions[code].y
    const d = dx * dx + dy * dy
    return d < best.d ? { code, d } : best
  }, { code: null, d: Infinity }).code
  const ballOwner = useMemo(() => {
    if (!optPositions || !of_balls_all) return null
    return of_balls_all.map(b =>
      (b.is_wall_ball || b.catch_prob < 0.05) ? null : nearestOf(b, OF_POSITIONS))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [optPositions, of_balls_all])
  const ifBallOwner = useMemo(() => {
    if (!optPositions || !if_balls_all) return null
    return if_balls_all.map(b => nearestOf(b, IF_POSITIONS))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [optPositions, if_balls_all])

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
    const ofLabel = ball.bb_type === 'line_drive' ? '平飛球'
      : ball.bb_type === 'fly_ball' ? '飛球' : '外野球'
    const lines = kind === 'of'
      ? [`${ofLabel}　接殺機率 ${(ball.catch_prob * 100).toFixed(0)}%`
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

  // 高飛不參與優化、不屬於任何野手：責任歸屬模式或點選野手高亮時一律淡出，
  // 否則會看起來像被歸進選中野手的責任球
  const dimPopup = (colorMode === 'owner' || activePos) ? 0.15 : null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', background: 'white' }}>
      {/* ── Controls（同外野主頁：歸屬色切換＋機率範圍）── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '16px',
        padding: '6px 12px', borderBottom: '1px solid var(--slate-200)', flexWrap: 'wrap',
      }}>
        {!showDensity && (
          <button onClick={() => { setColorMode(m => m === 'prob' ? 'owner' : 'prob'); setActivePos(null) }} style={{
            padding: '3px 10px', borderRadius: '4px', cursor: 'pointer', fontSize: '12px',
            border: '1px solid #cbd5e1',
            background: colorMode === 'owner' ? '#1e40af' : 'white',
            color: colorMode === 'owner' ? 'white' : '#334155',
          }}>
            {colorMode === 'prob' ? '切換：責任歸屬色' : '切換：接殺機率色'}
          </button>
        )}
        <button onClick={() => { setShowDensity(v => !v); setActivePos(null) }} style={{
          padding: '3px 10px', borderRadius: '4px', cursor: 'pointer', fontSize: '12px',
          border: '1px solid #cbd5e1',
          background: showDensity ? '#1e40af' : 'white',
          color: showDensity ? 'white' : '#334155',
        }}>
          {showDensity ? '切換：球點' : '切換：落點密度'}
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
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '12px', color: '#475569' }}>
          <span>球種</span>
          {[['ground_ball', '滾地球'], ['fly_ball', '飛球'],
            ['line_drive', '平飛球'], ['popup', '內野高飛']].map(([key, label]) => (
            <label key={key} style={{ display: 'flex', alignItems: 'center', gap: '3px', cursor: 'pointer' }}>
              <input type="checkbox" checked={showTypes[key]}
                onChange={e => setShowTypes(s => ({ ...s, [key]: e.target.checked }))}
                style={{ accentColor: '#4472C4', cursor: 'pointer' }} />
              {label}
            </label>
          ))}
        </div>
      </div>

    <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{ width: '100%', height: 'auto', display: 'block', background: 'white' }}>
      <defs>
        <linearGradient id="ic-grad-h" x1="0" y1="0" x2="1" y2="0">
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

      {/* 落點密度層＋等高線（取代球點；輸入吃球種勾選＋機率範圍過濾） */}
      {showDensity && densitySrc && (
        <image x={PL} y={PT} width={PW} height={PH} href={densitySrc} />
      )}
      {showDensity && contours && (
        <g transform={`translate(${PL},${PT})`}>
          {contours.map((d, i) => d && (
            <path key={i} d={d} fill="none" stroke="#1e3a8a"
              strokeWidth="0.9" opacity={0.28 + i * 0.13} />
          ))}
        </g>
      )}
      {/* 內野高飛（展示用，不參與優化——顏色＝實證常數接殺機率，畫在最底層） */}
      {!showDensity && popup_balls.filter(() => showTypes.popup && inRange(POPUP_CATCH)).map((b, i) => {
        const [bx, by] = clampXY(b.x, b.y)
        const isHov = hovered && hovered.kind === 'popup' && hovered.ball === b
        return (
          <circle key={`pu-${i}`}
            cx={tx(bx)} cy={ty(by)}
            r={isHov ? 6 : 3.5}
            fill={rdylgn(POPUP_CATCH)}
            stroke={b.is_out ? '#555' : 'white'} strokeWidth={b.is_out ? 0.9 : 0.6}
            opacity={dimPopup ?? 0.85}
            onMouseEnter={() => setHovered({ kind: 'popup', ball: b })}
            onMouseLeave={() => setHovered(null)}
            style={{ cursor: 'pointer' }} />
        )
      })}
      {/* 滾地球（顏色 = P(out) 或責任歸屬） */}
      {!showDensity && if_balls.map((b, i) => {
        if (!showTypes.ground_ball || !inRange(b.p_out_opt)) return null
        const [bx, by] = clampXY(b.x, b.y)
        const isHov = hovered && hovered.kind === 'if' && hovered.ball === b
        const owner = ifBallOwner?.[i]
        const mine = !activePos || owner === activePos
        const fill = colorMode === 'owner'
          ? (OWNER_COLORS[owner] ?? OWNER_COLORS.null)
          : rdylgn(b.p_out_opt)
        return (
          <circle key={`if-${i}`}
            cx={tx(bx)} cy={ty(by)}
            r={isHov ? 6 : 4}
            fill={fill}
            fillOpacity={mine ? 0.88 : 0.10}
            stroke={mine ? (b.is_out ? '#555' : 'white') : 'gray'}
            strokeWidth={mine ? (b.is_out ? 1.1 : 0.6) : 0.2}
            onMouseEnter={() => setHovered({ kind: 'if', ball: b })}
            onMouseLeave={() => setHovered(null)}
            style={{ cursor: 'pointer' }} />
        )
      })}
      {/* 外野球（顏色 = 接殺機率或責任歸屬；打牆球另畫橘星） */}
      {!showDensity && of_balls.map((b, i) => {
        if (b.is_wall_ball || !inRange(b.catch_prob)) return null
        if (b.bb_type && !showTypes[b.bb_type]) return null
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
      {/* 球場牆線與打牆球（同外野主頁慣例：綠線＋橘星）；
          通用球場改畫 400 呎虛線弧當距離參考（同外野主頁） */}
      {park_boundary ? (
        <polyline fill="none" stroke="#00CC55" strokeWidth="2.2" opacity="0.9"
          points={park_boundary.map(p => `${tx(p.x).toFixed(1)},${ty(p.y).toFixed(1)}`).join(' ')} />
      ) : (() => {
        const arcY = Math.sqrt(400 ** 2 - X1 ** 2)          // 弧與視野左右緣的交點
        const rPx = 400 * PW / (X1 - X0)                    // 400 呎換算像素半徑
        return (
          <path d={`M ${tx(X1).toFixed(1)},${ty(arcY).toFixed(1)} A ${rPx.toFixed(1)} ${rPx.toFixed(1)} 0 0 0 ${tx(X0).toFixed(1)},${ty(arcY).toFixed(1)}`}
            fill="none" stroke="#999" strokeWidth="2" strokeDasharray="8 5" opacity="0.8" />
        )
      })()}
      {of_balls.filter(b => b.is_wall_ball && (!b.bb_type || showTypes[b.bb_type])).map((b, i) => {
        const [bx, by] = clampXY(b.x, b.y)
        return (
          <polygon key={`wb-${i}`} points={starPts(tx(bx), ty(by), 7)}
            fill="#FF6B00" stroke="black" strokeWidth="0.4" opacity="0.9" />
        )
      })}

      {/* 七人站位（最佳化紫星；點任一人高亮其責任球） */}
      {[...OF_POSITIONS, ...IF_POSITIONS].map(p => (
        <PosMarker key={`o-${p}`} cx={tx(optimized.positions[p].x)} cy={ty(optimized.positions[p].y)}
          code={p} isActive={activePos === p}
          onClick={() => setActivePos(a => a === p ? null : p)} />
      ))}

      {/* Legend（疊在圖內右下角的半透明卡片） */}
      {(() => {
        const nFly = of_balls.filter(b => b.bb_type === 'fly_ball').length
        const nLd = of_balls.filter(b => b.bb_type === 'line_drive').length
        const hasType = nFly + nLd === of_balls.length
        const countLines = [
          hasType ? `飛球 ${nFly}・平飛 ${nLd}` : `外野 ${of_balls.length} 球`,
          `滾地 ${if_balls.length}` + (popup_balls.length > 0 ? `・高飛 ${popup_balls.length}` : ''),
        ]
        const isProb = colorMode === 'prob' || showDensity
        const H = (isProb ? 100 : 144) + (nWall > 0 ? 16 : 0)
        const W = 178
        const LX = PL + PW - W - 8
        const LY = PT + PH - H - 8
        let y = 14
        const rows = []
        // 最佳化站位
        rows.push(
          <g key="star">
            <polygon points={starPts(16, y, 7)} fill="#7B2FBE" stroke="white" strokeWidth="1" />
            <text x={28} y={y + 3.5} fontSize="9.5" fill="#555">最佳化站位</text>
          </g>)
        y += 14
        if (showDensity) {
          rows.push(
            <text key="dn" x={10} y={y + 8} fontSize="8.5" fill="#555">
              藍色越深＝落點越密集
            </text>)
          y += 26
        } else if (isProb) {
          // 水平色階條
          rows.push(
            <g key="cb">
              <rect x={10} y={y} width={104} height={9}
                fill="url(#ic-grad-h)" stroke="#bbb" strokeWidth="0.5" />
              <text x={10} y={y + 20} fontSize="8" fill="#555">0%</text>
              <text x={114} y={y + 20} fontSize="8" fill="#555" textAnchor="end">100%</text>
              <text x={120} y={y + 8} fontSize="8" fill="#555">接殺/出局</text>
            </g>)
          y += 26
        } else {
          // 七人歸屬色塊（兩欄）
          rows.push(
            <g key="own">
              {[...OF_POSITIONS, ...IF_POSITIONS, '其他'].map((label, i) => (
                <g key={label} transform={`translate(${10 + (i % 2) * 88},${y + Math.floor(i / 2) * 17})`}>
                  <rect width={10} height={10} rx="2" fill={OWNER_COLORS[label] ?? OWNER_COLORS.null} />
                  <text x={14} y={8.5} fontSize="8.5" fill="#555">{label}</text>
                </g>
              ))}
              <text x={10} y={y + 4 * 17 + 8} fontSize="8" fill="#999">點星標高亮其責任球</text>
            </g>)
          y += 4 * 17 + 16
        }
        rows.push(
          <text key="counts" x={10} y={y + 8} fontSize="8.5" fill="#777">
            <tspan x={10} dy="0">{countLines[0]}</tspan>
            <tspan x={10} dy="12">{countLines[1]}</tspan>
          </text>)
        y += 30
        if (nWall > 0) {
          rows.push(
            <g key="wall">
              <polygon points={starPts(16, y, 6)} fill="#FF6B00" stroke="black" strokeWidth="0.4" />
              <text x={26} y={y + 3} fontSize="8.5" fill="#555">打牆球 ({nWall})</text>
            </g>)
        }
        return (
          <g transform={`translate(${LX},${LY})`}>
            <rect width={W} height={H} rx="6" fill="white" fillOpacity="0.88"
              stroke="#ccc" strokeWidth="0.8" />
            {rows}
          </g>
        )
      })()}

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
