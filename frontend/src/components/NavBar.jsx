import { NavLink } from 'react-router-dom'
import { useLanguage } from '../i18n/LanguageContext.jsx'

export default function NavBar() {
  const { lang, setLang, t } = useLanguage()

  return (
    <nav style={s.nav}>
      <span style={s.brand}>MLB Lab</span>
      <div style={s.links}>
        <NavLink to="/" end style={({ isActive }) => ({ ...s.link, ...(isActive ? s.active : {}) })}>
          {t('nav.positioning')}
        </NavLink>
        <NavLink to="/rankings" style={({ isActive }) => ({ ...s.link, ...(isActive ? s.active : {}) })}>
          {t('nav.rankings')}
        </NavLink>
      </div>
      <div style={s.langToggle} role="group" aria-label="Language">
        <button
          onClick={() => setLang('zh')}
          style={{ ...s.langBtn, ...(lang === 'zh' ? s.langActive : {}) }}
        >
          中文
        </button>
        <button
          onClick={() => setLang('en')}
          style={{ ...s.langBtn, ...(lang === 'en' ? s.langActive : {}) }}
        >
          EN
        </button>
      </div>
    </nav>
  )
}

const s = {
  nav: {
    position: 'sticky',
    top: 0,
    zIndex: 100,
    display: 'flex',
    alignItems: 'center',
    gap: 24,
    padding: '0 24px',
    height: 48,
    background: 'white',
    borderBottom: '1px solid var(--slate-200)',
    boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
  },
  brand: {
    fontWeight: 700,
    fontSize: 15,
    color: 'var(--slate-800)',
    marginRight: 8,
    whiteSpace: 'nowrap',
    flexShrink: 0,
  },
  links: {
    display: 'flex',
    gap: 4,
    overflowX: 'auto',       // scroll horizontally on narrow screens once there are more links, instead of squeezing into a vertical stack
    scrollbarWidth: 'none',
  },
  link: {
    padding: '6px 14px',
    borderRadius: 6,
    fontSize: 13,
    color: 'var(--slate-500)',
    textDecoration: 'none',
    transition: 'background 0.15s',
    whiteSpace: 'nowrap',
    flexShrink: 0,
  },
  active: {
    background: 'var(--slate-100)',
    color: 'var(--slate-800)',
    fontWeight: 600,
  },
  langToggle: {
    display: 'flex',
    gap: 2,
    marginLeft: 'auto',
    flexShrink: 0,
    background: 'var(--slate-100)',
    borderRadius: 6,
    padding: 2,
  },
  langBtn: {
    padding: '4px 10px',
    borderRadius: 4,
    fontSize: 12,
    fontWeight: 600,
    color: 'var(--slate-500)',
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    transition: 'background 0.15s, color 0.15s',
  },
  langActive: {
    background: 'white',
    color: 'var(--slate-800)',
    boxShadow: '0 1px 2px rgba(0,0,0,0.08)',
  },
}
