import NavBar from './NavBar'

export default function Layout({ children }) {
  return (
    <div style={{ minHeight: '100vh', background: 'var(--slate-50)', color: 'var(--slate-800)', fontFamily: 'system-ui, sans-serif' }}>
      <NavBar />
      {children}
    </div>
  )
}
