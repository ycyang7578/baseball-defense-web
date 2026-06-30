export default function GameStateForm({ state, onChange }) {
  const { on1b, on2b, on3b, outs } = state

  function toggleBase(key) {
    onChange({ ...state, [key]: state[key] ? 0 : 1 })
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div>
        <label style={labelStyle}>壘上跑者</label>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 4, marginTop: 4 }}>
          <div />
          <BaseBtn label="2B" active={on2b} onClick={() => toggleBase('on2b')} />
          <div />
          <BaseBtn label="3B" active={on3b} onClick={() => toggleBase('on3b')} />
          <div />
          <BaseBtn label="1B" active={on1b} onClick={() => toggleBase('on1b')} />
        </div>
      </div>

      <div>
        <label style={labelStyle}>出局數</label>
        <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
          {[0, 1, 2].map(n => (
            <button
              key={n}
              onClick={() => onChange({ ...state, outs: n })}
              style={{
                ...btnBase,
                flex: 1,
                background: outs === n ? '#f59e0b' : '#f8fafc',
                color: outs === n ? '#1f2937' : '#6b7280',
                fontWeight: outs === n ? 'bold' : 'normal',
                border: outs === n ? '1px solid #f59e0b' : '1px solid #d1d5db',
              }}
            >
              {n} out{n !== 1 ? 's' : ''}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

function BaseBtn({ label, active, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        ...btnBase,
        background: active ? '#f59e0b' : '#f8fafc',
        color:      active ? '#1f2937' : '#9ca3af',
        fontWeight: active ? 'bold' : 'normal',
        border:     active ? '1px solid #f59e0b' : '1px solid #d1d5db',
      }}
    >
      {label}
    </button>
  )
}

const btnBase = {
  padding: '5px 0',
  borderRadius: 5,
  cursor: 'pointer',
  fontSize: 11,
  transition: 'background 0.15s',
}

const labelStyle = {
  fontSize: 10,
  color: '#6b7280',
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
}
