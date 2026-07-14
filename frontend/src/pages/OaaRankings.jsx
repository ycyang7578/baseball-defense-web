import { useState } from 'react'
import Rankings from './Rankings'
import InfieldRankings from './InfieldRankings'

// OAA 排名：外野/內野兩張排名表合併在同一頁（頂部切換），
// 各自沿用原本的年份/球隊/門檻控制
export default function OaaRankings() {
  const [side, setSide] = useState('of')
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'center', gap: 6,
                    padding: '14px 0 0', background: 'var(--slate-50)' }}>
        {[['of', '外野手'], ['if', '內野手']].map(([k, label]) => (
          <button key={k} onClick={() => setSide(k)} style={{
            padding: '6px 22px', fontSize: 13, fontWeight: 600, cursor: 'pointer',
            borderRadius: 7, transition: 'all 0.15s',
            background: side === k ? 'var(--blue-600)' : 'white',
            color: side === k ? 'white' : 'var(--slate-600)',
            border: side === k ? '1px solid var(--blue-600)' : '1px solid var(--slate-200)',
          }}>{label}</button>
        ))}
      </div>
      {side === 'of' ? <Rankings /> : <InfieldRankings />}
    </div>
  )
}
