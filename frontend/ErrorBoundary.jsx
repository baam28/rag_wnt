import React from 'react';

/**
 * ErrorBoundary — catches any uncaught render errors in the component tree
 * and shows a friendly Vietnamese fallback UI instead of a blank screen.
 *
 * Usage:
 *   <ErrorBoundary>
 *     <App />
 *   </ErrorBoundary>
 */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    // Log to console so the developer can see the full stack trace
    console.error('[ErrorBoundary] Uncaught error:', error, info.componentStack);
  }

  handleReload() {
    window.location.reload();
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '100vh',
          padding: '2rem',
          fontFamily: 'Inter, system-ui, sans-serif',
          background: '#f8fafc',
          color: '#334155',
          textAlign: 'center',
        }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>⚠️</div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.5rem' }}>
            Đã xảy ra lỗi
          </h1>
          <p style={{ color: '#64748b', maxWidth: '480px', marginBottom: '1.5rem' }}>
            Ứng dụng gặp sự cố không mong đợi. Vui lòng thử tải lại trang hoặc liên hệ
            quản trị viên nếu sự cố tiếp tục xảy ra.
          </p>
          {this.state.error && (
            <details style={{
              marginBottom: '1.5rem',
              padding: '0.75rem 1rem',
              background: '#fef2f2',
              border: '1px solid #fca5a5',
              borderRadius: '8px',
              maxWidth: '560px',
              textAlign: 'left',
              fontSize: '0.8rem',
              color: '#dc2626',
              cursor: 'pointer',
            }}>
              <summary style={{ fontWeight: 600 }}>Chi tiết lỗi (dành cho kỹ thuật)</summary>
              <pre style={{ marginTop: '0.5rem', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                {this.state.error.toString()}
              </pre>
            </details>
          )}
          <button
            onClick={this.handleReload}
            style={{
              padding: '0.6rem 1.5rem',
              background: '#2563eb',
              color: '#fff',
              border: 'none',
              borderRadius: '8px',
              fontSize: '0.95rem',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Tải lại trang
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
