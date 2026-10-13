import { NavLink } from 'react-router-dom'

const LINKS = [
  { to: '/caption', label: 'Caption' },
  { to: '/history', label: 'History' },
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/evaluation', label: 'Evaluation' },
]

export default function NavBar() {
  return (
    <header className="border-b border-border bg-surface">
      <nav aria-label="Main" className="mx-auto flex max-w-6xl flex-wrap items-center gap-2 px-4 py-3">
        <span className="mr-4 text-lg font-bold tracking-tight">Lumina</span>
        <ul className="flex flex-wrap items-center gap-1">
          {LINKS.map((link) => (
            <li key={link.to}>
              <NavLink
                to={link.to}
                className={({ isActive }) =>
                  `rounded-lg px-3 py-2 text-sm font-semibold transition-colors ${
                    isActive
                      ? 'bg-accent text-white'
                      : 'text-fg hover:bg-accent-soft'
                  }`
                }
              >
                {link.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  )
}