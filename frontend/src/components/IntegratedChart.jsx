import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { useLanguage } from '../i18n/LanguageContext.jsx'

// ── Layout (1 SVG unit ≈ 1 ft, home plate is the origin, +x toward first base) ────────────
// Viewport covers the whole field: infield dirt to deep outfield (same coordinate convention
// as InfieldChart, just a larger range).
// Margins are kept minimal so the field itself fills the layout (legend/colorbar overlay the bottom-right corner of the chart)
const PL = 10, PT = 12, PW = 560, PR = 12, PB = 10
// Y0=-60: 99% of foul popups behind home plate land within -49 (measured from precomputed_batter_popups),
// deeper ones get clamped to the bottom edge; Y1=425 leaves margin beyond the 400ft arc and the deepest wall (~420)
const X0 = -262, X1 = 262, Y0 = -60, Y1 = 425
const PH = PW * (Y1 - Y0) / (X1 - X0)          // keep aspect ratio, no distortion
const SVG_W = PL + PW + PR
const SVG_H = PT + PH + PB
const MAX_R = 412   // deep balls beyond the viewport get clamped to the edge (along the same direction)

const tx = x => PL + (x - X0) / (X1 - X0) * PW
const ty = y => PT + (Y1 - y) / (Y1 - Y0) * PH

// ── RdYlGn colormap (same as SprayChart / InfieldChart) ─────────
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

// ── Marker shapes (same convention as the site: optimized = purple star; league average isn't drawn, only kept for numeric comparison) ──
function starPts(cx, cy, r) {
  return Array.from({ length: 10 }, (_, i) => {
    const a = (Math.PI * i) / 5 - Math.PI / 2
    const rad = i % 2 === 0 ? r : r * 0.38
    return `${(cx + Math.cos(a) * rad).toFixed(2)},${(cy + Math.sin(a) * rad).toFixed(2)}`
  }).join(' ')
}

const OPT_STYLE = { color: '#7B2FBE', r: 10, dy: +20 }
// Ownership colors: outfield matches the outfield main page; infield uses four separate colors (avoiding the star marker's purple and the field's green)
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

// ── Infield dirt outer edge: arc centered at the pitcher's mound (0, 60.5) with a 95ft radius (same as src/if_optimize.py) ──
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

// Ball markers are plotted at Statcast-recorded coordinates; balls too deep get clamped to the
// viewport edge along the same direction. Foul balls that are too deep/wide are clamped back into the viewport.
function clampXY(x, y) {
  const r = Math.hypot(x, y)
  const k = r > MAX_R ? MAX_R / r : 1
  return [Math.max(X0 + 6, Math.min(X1 - 6, x * k)),
          Math.max(y * k, Y0 + 6)]
}

const OF_POSITIONS = ['LF', 'CF', 'RF']
const IF_POSITIONS = ['1B', '2B', '3B', 'SS']

// ── Density colormap: matplotlib "Blues" (same cmap as the outfield page's seaborn kdeplot) ──
const BLUES = [
  [0.000, [247, 251, 255]], [0.125, [222, 235, 247]], [0.250, [198, 219, 239]],
  [0.375, [158, 202, 225]], [0.500, [107, 174, 214]], [0.625, [66, 146, 198]],
  [0.750, [33, 113, 181]], [0.875, [8, 81, 156]], [1.000, [8, 48, 107]],
]

function bluesColor(t) {
  t = Math.max(0, Math.min(1, t))
  for (let i = 0; i < BLUES.length - 1; i++) {
    const [t0, c0] = BLUES[i], [t1, c1] = BLUES[i + 1]
    if (t <= t1) {
      const k = (t - t0) / (t1 - t0)
      return [Math.round(c0[0] + (c1[0] - c0[0]) * k),
              Math.round(c0[1] + (c1[1] - c0[1]) * k),
              Math.round(c0[2] + (c1[2] - c0[2]) * k)]
    }
  }
  return BLUES[BLUES.length - 1][1]
}

// Same style as the outfield page's seaborn: thresh=0.05 (bottom 5% left unfilled); 10 levels, light fill
const DENS_THRESH = 0.05
const DENS_LEVELS = 10
const DENS_EDGES = Array.from({ length: DENS_LEVELS },
  (_, i) => DENS_THRESH + i * (1 - DENS_THRESH) / (DENS_LEVELS - 1))

// ── Marching squares: extracts one contour's line segments from the grid KDE (as an SVG path string) ──
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

// Popup catch probability = an empirical league constant (98.5% outs in the 2025 regular season;
// catchable from anywhere, so it doesn't factor into optimization). The outfield model treats popups
// as OOD and is less well-calibrated than the constant — don't switch it back to model-computed values.
const POPUP_CATCH = 0.985

export default function IntegratedChart({ data }) {
  const { t } = useLanguage()
  const svgRef = useRef(null)
  // SVG defs ids must be unique per instance: with two charts on the same page in comparison mode,
  // a fixed id would make chart B reference chart A's clipPath/gradient (2026-07-14 user report:
  // chart B's grass boundary was following chart A's ballpark)
  const uid = useId().replace(/:/g, '')   // colons are a compatibility risk in url(#...) references
  const gradId = `ic-grad-h-${uid}`
  const clipId = `ic-field-clip-${uid}`
  const [hovered, setHovered] = useState(null)   // { kind: 'of'|'if'|'popup', ball }
  const [activePos, setActivePos] = useState(null)   // one of the seven positions, or null
  const [colorMode, setColorMode] = useState('prob') // 'prob' | 'owner'
  const [probMin, setProbMin] = useState(0)          // 0–100 integer
  const [probMax, setProbMax] = useState(100)
  // Batted-ball-type filter (multi-select): ground ball → infield ball, fly ball/line drive → outfield ball (actual bb_type label), popup → popup
  const [showTypes, setShowTypes] = useState(
    { ground_ball: true, fly_ball: true, line_drive: true, popup: true })
  // Landing-spot density layer (on by default): the currently visible balls (after batted-ball-type
  // and probability-range filtering) are computed into a grid KDE, rendered as blue fill + contour
  // lines underneath the ball markers (same presentation as the outfield page's matplotlib KDE)
  const [showDensity, setShowDensity] = useState(true)
  const [densitySrc, setDensitySrc] = useState(null)
  const [contours, setContours] = useState(null)

  useEffect(() => {
    if (!showDensity || !data) { setDensitySrc(null); setContours(null); return }
    const within = (p) => p * 100 >= probMin && p * 100 <= probMax
    // Infield and outfield are collected separately and normalized independently — ground ball density
    // peaks are far higher than outfield, so a shared normalization would squash all outfield structure
    // into the lowest color band (2026-07-14 user report)
    const ptsIF = []
    const ptsOF = []
    if (showTypes.ground_ball)
      for (const b of data.if_balls) if (within(b.p_out_opt)) ptsIF.push(clampXY(b.x, b.y))
    if (showTypes.popup && within(POPUP_CATCH))
      for (const b of (data.popup_balls || [])) ptsIF.push(clampXY(b.x, b.y))
    for (const b of data.of_balls) {
      if (b.is_wall_ball || (b.bb_type && !showTypes[b.bb_type]) || !within(b.catch_prob)) continue
      ptsOF.push(clampXY(b.x, b.y))
    }
    if (ptsIF.length + ptsOF.length === 0) { setDensitySrc(null); setContours(null); return }

    // Grid KDE (Gaussian kernel). Bandwidth adapts to the data via Scott's rule (same as seaborn's default) —
    // a fixed narrow bandwidth makes the contours grow into irregular amoeba shapes (2026-07-14 user report)
    const CELL = 4
    const GW = Math.ceil(PW / CELL) + 1
    const GH = Math.ceil(PH / CELL) + 1
    const kde = (pts) => {
      const g = new Float32Array(GW * GH)
      if (pts.length === 0) return [g, 0]
      // Scott's rule: sigma = mean(std_x, std_y) × n^(-1/6), converted to grid cells
      const px = pts.map(([x]) => (x - X0) / (X1 - X0) * PW)
      const py = pts.map(([, y]) => (Y1 - y) / (Y1 - Y0) * PH)
      const std = (a) => {
        const m = a.reduce((s, v) => s + v, 0) / a.length
        return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length)
      }
      const scottPx = 0.5 * (std(px) + std(py)) * Math.pow(pts.length, -1 / 6)
      const sigma = Math.max(4, Math.min(12, scottPx / CELL))
      const win = Math.ceil(sigma * 3)
      for (let k = 0; k < pts.length; k++) {
        const gx = px[k] / CELL
        const gy = py[k] / CELL
        const ix0 = Math.max(0, Math.floor(gx - win)), ix1 = Math.min(GW - 1, Math.ceil(gx + win))
        const iy0 = Math.max(0, Math.floor(gy - win)), iy1 = Math.min(GH - 1, Math.ceil(gy + win))
        for (let iy = iy0; iy <= iy1; iy++)
          for (let ix = ix0; ix <= ix1; ix++) {
            const d2 = (ix - gx) ** 2 + (iy - gy) ** 2
            g[iy * GW + ix] += Math.exp(-d2 / (2 * sigma * sigma))
          }
      }
      let m = 0
      for (let i = 0; i < g.length; i++) if (g[i] > m) m = g[i]
      return [g, m]
    }
    const [gIF, mIF] = kde(ptsIF)
    const [gOF, mOF] = kde(ptsOF)
    // Merge: each cell takes the larger of the two independently-normalized values → range [0,1]
    const grid = new Float32Array(GW * GH)
    for (let i = 0; i < grid.length; i++)
      grid[i] = Math.max(mIF > 0 ? gIF[i] / mIF : 0, mOF > 0 ? gOF[i] / mOF : 0)
    const maxV = 1

    // Fill layer: banded fill (same as seaborn's fill=True discrete color bands). Per pixel, the grid
    // is bilinearly sampled then quantized into a color band, so the band edges look as crisp as matplotlib's
    const big = document.createElement('canvas')
    big.width = PW
    big.height = PH
    const bctx = big.getContext('2d')
    const img = bctx.createImageData(PW, PH)
    const bandW = (1 - DENS_THRESH) / (DENS_LEVELS - 1)
    for (let py = 0; py < PH; py++) {
      const gy = py / CELL
      const iy = Math.min(GH - 2, Math.floor(gy))
      const fy = gy - iy
      for (let px = 0; px < PW; px++) {
        const gx = px / CELL
        const ix = Math.min(GW - 2, Math.floor(gx))
        const fx = gx - ix
        const v = ((grid[iy * GW + ix] * (1 - fx) + grid[iy * GW + ix + 1] * fx) * (1 - fy)
                   + (grid[(iy + 1) * GW + ix] * (1 - fx) + grid[(iy + 1) * GW + ix + 1] * fx) * fy) / maxV
        if (v < DENS_THRESH) continue
        const band = Math.min(DENS_LEVELS - 2, Math.floor((v - DENS_THRESH) / bandW))
        // Colormap capped at 0.85, opacity kept low overall — don't let the darkest band's deep blue overpower the ball markers
        const [r, g, b] = bluesColor(0.85 * (band + 1) / (DENS_LEVELS - 1))
        const o = (py * PW + px) * 4
        img.data[o] = r
        img.data[o + 1] = g
        img.data[o + 2] = b
        img.data[o + 3] = Math.round(255 * (0.18 + band * 0.028))
      }
    }
    bctx.putImageData(img, 0, 0)
    setDensitySrc(big.toDataURL('image/png'))

    // Contours: same as seaborn's levels=10 contour lines
    setContours(DENS_EDGES.map(t => marchingSquares(grid, GW, GH, t * maxV, CELL)))
  }, [showDensity, data, showTypes, probMin, probMax])

  // Ownership: whichever optimized position is nearest (same frontend fallback algorithm as the outfield main page).
  // Outfield balls are assigned among LF/CF/RF, ground balls among 1B/2B/3B/SS (batted-ball type already determines the fielding side)
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
    const ofLabel = ball.bb_type === 'line_drive' ? t('chart.tooltip.lineDrive')
      : ball.bb_type === 'fly_ball' ? t('chart.tooltip.flyBall') : t('chart.tooltip.ofBall')
    const outHit = ball.is_out ? t('chart.tooltip.out') : t('chart.tooltip.hitError')
    const lines = kind === 'of'
      ? [`${ofLabel}　${t('chart.tooltip.catchProb')} ${(ball.catch_prob * 100).toFixed(0)}%`
         + (ballOwner?.[of_balls.indexOf(ball)] ? t('chart.tooltip.owner', { code: ballOwner[of_balls.indexOf(ball)] }) : '')]
      : kind === 'popup'
      ? [
          `${t('chart.tooltip.popupLabel')}　${outHit}　${t('chart.tooltip.popupCatchProb')}`,
          t('chart.tooltip.popupHint'),
        ]
      : [
          `${t('chart.tooltip.gbLabel')}　${outHit}　EV ${ball.launch_speed.toFixed(0)} mph`,
          t('chart.tooltip.avgToOpt', {
            avg: (ball.p_out_league * 100).toFixed(0),
            opt: (ball.p_out_opt * 100).toFixed(0),
          }),
        ]
    return { x: tx(bx), y: ty(by), lines }
  })() : null

  // Popups don't participate in optimization and don't belong to any fielder: they're always dimmed
  // in ownership mode or when a fielder is highlighted by click, otherwise they'd look like they were
  // assigned to the selected fielder's responsibility
  const dimPopup = (colorMode === 'owner' || activePos) ? 0.15 : null

  // Download PNG: serialize the SVG (the density layer is a data URI, so it's carried along) → canvas 2x → download
  const downloadPng = () => {
    const svg = svgRef.current
    if (!svg) return
    const clone = svg.cloneNode(true)
    clone.setAttribute('width', SVG_W)
    clone.setAttribute('height', SVG_H)
    const xml = new XMLSerializer().serializeToString(clone)
    const url = URL.createObjectURL(new Blob([xml], { type: 'image/svg+xml;charset=utf-8' }))
    const img = new Image()
    img.onload = () => {
      const scale = 2
      const c = document.createElement('canvas')
      c.width = SVG_W * scale
      c.height = SVG_H * scale
      const ctx = c.getContext('2d')
      ctx.fillStyle = 'white'
      ctx.fillRect(0, 0, c.width, c.height)
      ctx.drawImage(img, 0, 0, c.width, c.height)
      URL.revokeObjectURL(url)
      const a = document.createElement('a')
      a.href = c.toDataURL('image/png')
      a.download = `positioning_${(data.name || '').replace(/[ ,]+/g, '_')}_${data.year}.png`
      a.click()
    }
    img.src = url
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', background: 'white' }}>
      {/* ── Controls (same as outfield main page: ownership color toggle + probability range) ── */}
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
          {colorMode === 'prob' ? t('chart.toggleOwner') : t('chart.toggleProb')}
        </button>
        <label style={{ display: 'flex', alignItems: 'center', gap: '3px', cursor: 'pointer',
                        fontSize: '12px', color: '#475569' }}>
          <input type="checkbox" checked={showDensity}
            onChange={e => setShowDensity(e.target.checked)}
            style={{ accentColor: '#4472C4', cursor: 'pointer' }} />
          {t('chart.density')}
        </label>
        <button onClick={downloadPng} style={{
          marginLeft: 'auto', padding: '3px 10px', borderRadius: '4px', cursor: 'pointer',
          fontSize: '12px', border: '1px solid #cbd5e1', background: 'white', color: '#334155',
        }}>
          {t('chart.download')}
        </button>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#475569' }}>
          <span>{t('chart.probRange')}</span>
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
          <span>{t('chart.ballType')}</span>
          {[['ground_ball', t('chart.ballTypes.ground_ball')], ['fly_ball', t('chart.ballTypes.fly_ball')],
            ['line_drive', t('chart.ballTypes.line_drive')], ['popup', t('chart.ballTypes.popup')]].map(([key, label]) => (
            <label key={key} style={{ display: 'flex', alignItems: 'center', gap: '3px', cursor: 'pointer' }}>
              <input type="checkbox" checked={showTypes[key]}
                onChange={e => setShowTypes(s => ({ ...s, [key]: e.target.checked }))}
                style={{ accentColor: '#4472C4', cursor: 'pointer' }} />
              {label}
            </label>
          ))}
        </div>
      </div>

    <svg ref={svgRef} viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      style={{ width: '100%', height: 'auto', display: 'block', background: 'white',
               fontFamily: 'system-ui, sans-serif' }}>
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="1" y2="0">
          {RDYLGN.map(([t, [r, g, b]]) => (
            <stop key={t} offset={`${t * 100}%`} stopColor={`rgb(${r},${g},${b})`} />
          ))}
        </linearGradient>
        {/* Clip region for the field fill: with a specific ballpark, the actual wall polygon; generic, a 400ft arc wedge */}
        <clipPath id={clipId}>
          {park_boundary ? (
            <polygon points={park_boundary.map(p => `${tx(p.x).toFixed(1)},${ty(p.y).toFixed(1)}`).join(' ')} />
          ) : (() => {
            const a = 45.2 * Math.PI / 180
            const fx = 400 * Math.sin(a), fy = 400 * Math.cos(a)
            const rPx = 400 * PW / (X1 - X0)
            return (
              <path d={`M ${tx(0)},${ty(0)} L ${tx(-fx).toFixed(1)},${ty(fy).toFixed(1)} A ${rPx.toFixed(1)} ${rPx.toFixed(1)} 0 0 1 ${tx(fx).toFixed(1)},${ty(fy).toFixed(1)} Z`} />
            )
          })()}
        </clipPath>
      </defs>
      {/* Grass wedge + extended foul lines (clipped to the ballpark boundary; nothing filled beyond the wall) */}
      <g clipPath={`url(#${clipId})`}>
        <path d={`M ${tx(0)} ${ty(0)} L ${tx(Y1 * Math.SQRT1_2 * 1.5)} ${ty(Y1 * 1.05)} L ${tx(0)} ${ty(Y1 * 1.4)} L ${tx(-Y1 * Math.SQRT1_2 * 1.5)} ${ty(Y1 * 1.05)} Z`}
          fill="#e8f2e4" />
        <line x1={tx(0)} y1={ty(0)} x2={tx(Y1 * Math.SQRT1_2 * 1.45)} y2={ty(Y1 * 1.02)} stroke="#fff" strokeWidth="2.5" />
        <line x1={tx(0)} y1={ty(0)} x2={tx(-Y1 * Math.SQRT1_2 * 1.45)} y2={ty(Y1 * 1.02)} stroke="#fff" strokeWidth="2.5" />
      </g>
      {/* Infield dirt (home plate to the dirt's outer-edge arc) */}
      <path d={`M ${tx(0)} ${ty(0)} L ${arcPts[0]} L ${arcPts.join(' L ')} Z`} fill="#e7d6bd" />
      {/* Infield grass square */}
      <polygon points={`${tx(0)},${ty(12)} ${tx(B1[0] - 8.5)},${ty(B1[1] + 3.5)} ${tx(0)},${ty(B2[1] - 5)} ${tx(B3[0] + 8.5)},${ty(B3[1] + 3.5)}`}
        fill="#d9ead3" />
      <line x1={tx(B1[0])} y1={ty(B1[1])} x2={tx(B2[0])} y2={ty(B2[1])} stroke="#fff" strokeWidth="2" />
      <line x1={tx(B3[0])} y1={ty(B3[1])} x2={tx(B2[0])} y2={ty(B2[1])} stroke="#fff" strokeWidth="2" />
      {/* Pitcher's mound and bases */}
      <circle cx={tx(0)} cy={ty(MOUND_Y)} r={(tx(9) - tx(0))} fill="#dbc7a9" />
      <polygon points={basePts(...B1)} fill="white" stroke="#bbb" strokeWidth="0.8" />
      <polygon points={basePts(...B2)} fill="white" stroke="#bbb" strokeWidth="0.8" />
      <polygon points={basePts(...B3)} fill="white" stroke="#bbb" strokeWidth="0.8" />
      <polygon points={basePts(0, 0)} fill="white" stroke="#bbb" strokeWidth="0.8" />

      {/* Landing-spot density layer + contours (rendered beneath the ball markers; input respects batted-ball-type and probability-range filters) */}
      {showDensity && densitySrc && (
        <image x={PL} y={PT} width={PW} height={PH} href={densitySrc} />
      )}
      {showDensity && contours && (
        <g transform={`translate(${PL},${PT})`}>
          {contours.map((d, i) => d && (
            <path key={i} d={d} fill="none" stroke="#4a72c4"
              strokeWidth="0.5" opacity="0.3" />
          ))}
        </g>
      )}
      {/* Infield popups (display only, not part of optimization — color = empirical constant catch probability, drawn at the bottom layer) */}
      {popup_balls.filter(() => showTypes.popup && inRange(POPUP_CATCH)).map((b, i) => {
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
      {/* Ground balls (color = P(out) or ownership) */}
      {if_balls.map((b, i) => {
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
      {/* Outfield balls (color = catch probability or ownership; wall balls are drawn separately as orange stars) */}
      {of_balls.map((b, i) => {
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
      {/* Ballpark wall line and wall balls (same convention as the outfield main page: green line + orange star);
          for a generic ballpark, a 400ft dashed arc is drawn instead as a distance reference (same as the outfield main page) */}
      {park_boundary ? (
        <polyline fill="none" stroke="#00CC55" strokeWidth="2.2" opacity="0.9"
          points={park_boundary.map(p => `${tx(p.x).toFixed(1)},${ty(p.y).toFixed(1)}`).join(' ')} />
      ) : (() => {
        const arcY = Math.sqrt(400 ** 2 - X1 ** 2)          // intersection of the arc with the viewport's left/right edges
        const rPx = 400 * PW / (X1 - X0)                    // 400ft converted to a pixel radius
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

      {/* Positions (optimized = purple star; clicking a fielder highlights their responsibility balls). Batters with too few ground balls only get the three outfield positions */}
      {[...OF_POSITIONS, ...IF_POSITIONS].filter(p => optimized.positions[p]).map(p => (
        <PosMarker key={`o-${p}`} cx={tx(optimized.positions[p].x)} cy={ty(optimized.positions[p].y)}
          code={p} isActive={activePos === p}
          onClick={() => setActivePos(a => a === p ? null : p)} />
      ))}

      {/* Legend (semi-transparent card overlaid in the chart's bottom-right corner) */}
      {(() => {
        const nFly = of_balls.filter(b => b.bb_type === 'fly_ball').length
        const nLd = of_balls.filter(b => b.bb_type === 'line_drive').length
        const hasType = nFly + nLd === of_balls.length
        const countLines = [
          hasType ? t('chart.countLine.flyLine', { fly: nFly, line: nLd }) : t('chart.countLine.ofTotal', { n: of_balls.length }),
          t('chart.countLine.gb', { n: if_balls.length }) + (popup_balls.length > 0 ? t('chart.countLine.popup', { n: popup_balls.length }) : ''),
        ]
        const isProb = colorMode === 'prob'
        const W = 140
        let y = 11
        const rows = []
        // Optimized position
        rows.push(
          <g key="star">
            <polygon points={starPts(13, y, 6)} fill="#7B2FBE" stroke="white" strokeWidth="1" />
            <text x={23} y={y + 3} fontSize="8.5" fill="#555">{t('chart.legend.optimizedPositions')}</text>
          </g>)
        y += 11
        if (isProb) {
          // Horizontal colorbar
          rows.push(
            <g key="cb">
              <rect x={8} y={y} width={80} height={8}
                fill={`url(#${gradId})`} stroke="#bbb" strokeWidth="0.5" />
              <text x={8} y={y + 17} fontSize="7.5" fill="#555">0%</text>
              <text x={88} y={y + 17} fontSize="7.5" fill="#555" textAnchor="end">100%</text>
              <text x={92} y={y + 7.5} fontSize="7.5" fill="#555">{t('chart.legend.catchOut')}</text>
            </g>)
          y += 22
        } else {
          // Seven-fielder ownership color swatches (two columns)
          rows.push(
            <g key="own">
              {[...OF_POSITIONS, ...IF_POSITIONS, t('chart.legend.other')].map((label, i) => (
                <g key={label} transform={`translate(${8 + (i % 2) * 66},${y + Math.floor(i / 2) * 14})`}>
                  <rect width={9} height={9} rx="2" fill={OWNER_COLORS[label] ?? OWNER_COLORS.null} />
                  <text x={12.5} y={7.5} fontSize="8" fill="#555">{label}</text>
                </g>
              ))}
              <text x={8} y={y + 4 * 14 + 7} fontSize="7.5" fill="#999">{t('chart.legend.clickHint')}</text>
            </g>)
          y += 4 * 14 + 13
        }
        rows.push(
          <text key="counts" x={8} y={y + 7} fontSize="8" fill="#777">
            <tspan x={8} dy="0">{countLines[0]}</tspan>
            <tspan x={8} dy="11">{countLines[1]}</tspan>
          </text>)
        y += 24
        if (nWall > 0) {
          rows.push(
            <g key="wall">
              <polygon points={starPts(13, y + 2, 5.5)} fill="#FF6B00" stroke="black" strokeWidth="0.4" />
              <text x={22} y={y + 5} fontSize="8" fill="#555">{t('chart.legend.wallBalls', { n: nWall })}</text>
            </g>)
          y += 13
        }
        const H = y + 4
        const LX = PL + PW - W - 8
        const LY = PT + PH - H - 8
        return (
          <g transform={`translate(${LX},${LY})`}>
            <rect width={W} height={H} rx="5" fill="white" fillOpacity="0.88"
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
