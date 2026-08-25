import type { ReactNode } from 'react'

/**
 * Plain-language scaffolding.
 *
 * The page assumes a reader who knows the basics of finance and machine
 * learning and no more — someone entering the job market. Jargon is allowed,
 * but every term gets defined the first time it carries weight. These asides do
 * that without interrupting the argument for a reader who already knows.
 */
export function Explain({
  term,
  children,
}: {
  term: string
  children: ReactNode
}) {
  return (
    <aside className="explain">
      <p className="explain-term">{term}</p>
      <p className="explain-body">{children}</p>
    </aside>
  )
}

/** The same thing, sized for the inside of a dark data panel. */
export function ReadThis({ children }: { children: ReactNode }) {
  return (
    <p className="read-this">
      <span className="read-this-label">How to read this</span>
      {children}
    </p>
  )
}
