import { useLanguage } from '../i18n/LanguageContext.jsx'

export default function GameStateForm({ state, onChange }) {
  const { on1b, on2b, on3b, outs } = state
  const { t } = useLanguage()
  const outLabels = [t('gameState.out0'), t('gameState.out1'), t('gameState.out2')]

  function toggleBase(key) {
    onChange({ ...state, [key]: state[key] ? 0 : 1 })
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div>
        <label style={labelStyle}>{t('gameState.runnersOnBase')}</label>
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
        <label style={labelStyle}>{t('gameState.outs')}</label>
        <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
          {[0, 1, 2].map(n => (
            <button
              key={n}
              onClick={() => onChange({ ...state, outs: n })}
              style={{
                ...btnBase,
                flex: 1,
                background: outs === n ? 'var(--amber-500)' : 'var(--slate-50)',
                color: outs === n ? '#1f2937' : 'var(--gray-500)',
                fontWeight: outs === n ? 'bold' : 'normal',
                border: outs === n ? '1px solid var(--amber-500)' : '1px solid var(--gray-300)',
              }}
            >
              {outLabels[n]}
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
        background: active ? 'var(--amber-500)' : 'var(--slate-50)',
        color:      active ? '#1f2937' : 'var(--gray-400)',
        fontWeight: active ? 'bold' : 'normal',
        border:     active ? '1px solid var(--amber-500)' : '1px solid var(--gray-300)',
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
  color: 'var(--gray-500)',
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
}
