import { createContext, useContext, useState, useCallback } from 'react'
import { translations } from './translations'

const STORAGE_KEY = 'mlb-lab-lang'
const LanguageContext = createContext(null)

function getInitialLang() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved === 'zh' || saved === 'en') return saved
  } catch { /* localStorage unavailable (private mode etc.) — fall through to default */ }
  return 'zh'
}

// Dot-path lookup, e.g. t('panel.batter') or t('chart.legend.wallBalls', { n: 3 })
function lookup(lang, key) {
  let node = translations[lang]
  for (const part of key.split('.')) {
    if (node == null) return key
    node = node[part]
  }
  return typeof node === 'string' ? node : key
}

function interpolate(str, vars) {
  if (!vars) return str
  return str.replace(/\{(\w+)\}/g, (m, name) => (name in vars ? vars[name] : m))
}

export function LanguageProvider({ children }) {
  const [lang, setLangState] = useState(getInitialLang)

  const setLang = useCallback((next) => {
    setLangState(next)
    try { localStorage.setItem(STORAGE_KEY, next) } catch { /* ignore write failures */ }
  }, [])

  const t = useCallback((key, vars) => interpolate(lookup(lang, key), vars), [lang])

  return (
    <LanguageContext.Provider value={{ lang, setLang, t }}>
      {children}
    </LanguageContext.Provider>
  )
}

export function useLanguage() {
  const ctx = useContext(LanguageContext)
  if (!ctx) throw new Error('useLanguage must be used within a LanguageProvider')
  return ctx
}
