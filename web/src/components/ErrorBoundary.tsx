import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
}

/**
 * Without this, an uncaught error anywhere in the render tree unmounts the
 * whole app — React shows nothing, and the visitor sees a blank white page
 * with no indication anything went wrong. This turns that into a message
 * with a way forward, and logs the real error to the console for debugging.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Render error caught by ErrorBoundary:', error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <main className="state">
          <p className="eyebrow">Something broke</p>
          <p style={{ maxWidth: '32rem' }}>
            The page hit an error it could not recover from. This is usually a stale
            cached copy of the data conflicting with newer code.
          </p>
          <button className="reset" onClick={() => window.location.reload()}>
            Reload the page
          </button>
          <p className="mono" style={{ fontSize: '0.72rem', marginTop: '1rem', opacity: 0.6 }}>
            {this.state.error.message}
          </p>
        </main>
      )
    }
    return this.props.children
  }
}
