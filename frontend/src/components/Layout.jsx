import NavBar from './NavBar'

export default function Layout({ children }) {
  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc', color: '#1e293b', fontFamily: 'system-ui, sans-serif' }}>
      <NavBar />
      {children}
    </div>
  )
}
