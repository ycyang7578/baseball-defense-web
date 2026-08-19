import { useState, useRef, useEffect } from 'react'
import { useLanguage } from '../i18n/LanguageContext.jsx'

// Typeahead-searchable dropdown. options: [{ value, label }]
export default function SearchSelect({ options, value, onChange, placeholder }) {
  const { t } = useLanguage()
  const [query, setQuery] = useState('')
  const [open, setOpen]   = useState(false)
  const [hi, setHi]       = useState(0)
  const ref = useRef(null)

  const selected = options.find(o => o.value === value)
  const filtered = (query.trim()
    ? options.filter(o => o.label.toLowerCase().includes(query.trim().toLowerCase()))
    : options
  ).slice(0, 100)

  useEffect(() => {
    function onDoc(e) { if (ref.current && !ref.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [])

  function pick(o) {
    onChange(o.value)
    setQuery('')
    setOpen(false)
  }

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <input
        value={open ? query : (selected ? selected.label : '')}
        placeholder={placeholder}
        onFocus={() => { setOpen(true); setQuery(''); setHi(0) }}
        onChange={e => { setQuery(e.target.value); setOpen(true); setHi(0) }}
        onKeyDown={e => {
          if (e.key === 'ArrowDown') { setHi(h => Math.min(h + 1, filtered.length - 1)); e.preventDefault() }
          else if (e.key === 'ArrowUp') { setHi(h => Math.max(h - 1, 0)); e.preventDefault() }
          else if (e.key === 'Enter') { if (filtered[hi]) pick(filtered[hi]); e.preventDefault() }
          else if (e.key === 'Escape') { setOpen(false) }
        }}
        style={st.input}
      />
      {open && (
        <div style={st.drop}>
          {filtered.length === 0 && <div style={st.empty}>{t('searchSelect.noMatches')}</div>}
          {filtered.map((o, i) => (
            <div
              key={o.value || '__none__'}
              onMouseDown={() => pick(o)}
              onMouseEnter={() => setHi(i)}
              style={{
                ...st.opt,
                background: i === hi ? 'var(--blue-600)' : 'transparent',
                color: i === hi ? '#fff' : (o.value === '' ? 'var(--gray-400)' : 'var(--slate-800)'),
              }}
            >
              {o.label}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const st = {
  input: {
    width: '100%', padding: '6px 8px', background: 'var(--slate-50)',
    color: 'var(--slate-800)', border: '1px solid var(--gray-300)', borderRadius: 5, fontSize: 12,
    outline: 'none',
  },
  drop: {
    position: 'absolute', top: '100%', left: 0, minWidth: '100%', zIndex: 20,
    marginTop: 2, maxHeight: 240, overflowY: 'auto',
    background: 'white', border: '1px solid var(--gray-300)', borderRadius: 5,
    boxShadow: '0 6px 18px rgba(0,0,0,0.1)',
  },
  opt: { padding: '5px 9px', fontSize: 12, cursor: 'pointer', whiteSpace: 'nowrap' },
  empty: { padding: '6px 9px', fontSize: 12, color: 'var(--gray-400)' },
}
