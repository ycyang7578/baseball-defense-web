import { NavLink } from 'react-router-dom'

export default function NavBar() {
  return (
    <nav style={s.nav}>
      <span style={s.brand}>MLB Lab</span>
      <div style={s.links}>
        <NavLink to="/" end style={({ isActive }) => ({ ...s.link, ...(isActive ? s.active : {}) })}>
          外野手站位最佳化
        </NavLink>
        <NavLink to="/rankings" style={({ isActive }) => ({ ...s.link, ...(isActive ? s.active : {}) })}>
          守備排名
        </NavLink>
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
  },
  links: {
    display: 'flex',
    gap: 4,
  },
  link: {
    padding: '6px 14px',
    borderRadius: 6,
    fontSize: 13,
    color: 'var(--slate-500)',
    textDecoration: 'none',
    transition: 'background 0.15s',
  },
  active: {
    background: 'var(--slate-100)',
    color: 'var(--slate-800)',
    fontWeight: 600,
  },
}
