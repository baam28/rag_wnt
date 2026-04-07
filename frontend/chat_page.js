import React, { useRef, useEffect } from 'react';
import { MessageBubble } from './app.js';

function ChatPage({
  sessions,
  activeSessionId,
  messages,
  onNewSession,
  onSelectSession,
  onDeleteSession,
  onSend,
  onCancel,
  onFeedback,
  question,
  setQuestion,
  isSending,
  askError,
}) {
  const textareaRef = useRef(null);
  const chatBodyRef = useRef(null);
  const [isSidebarOpen, setIsSidebarOpen] = React.useState(true);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 128) + 'px';
  }, [question]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    const el = chatBodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  // Submit on Enter (not Shift+Enter)
  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isSending && question.trim()) onSend(e);
    }
  }

  function triggerCancel(e) {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    onCancel && onCancel();
  }

  function formatDateLabel(iso) {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      return d.toLocaleString("vi-VN", {
        day: "2-digit",
        month: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return iso;
    }
  }

  function renderSidebar() {
    return (
      <>
      <div 
        className={`sidebar-backdrop ${isSidebarOpen ? 'show' : ''}`} 
        onClick={() => setIsSidebarOpen(false)} 
      />
      <aside className={`sidebar ${!isSidebarOpen ? 'collapsed' : ''}`}>
        <div className="sidebar-body" style={{ padding: '1.25rem 1rem' }}>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
            <button
              type="button"
              onClick={() => setIsSidebarOpen(false)}
              style={{
                background: 'rgba(0,0,0,0.05)',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                padding: '5px',
                display: 'flex',
                alignItems: 'center',
                color: 'inherit',
                transition: 'background 0.2s',
              }}
              onMouseOver={(e) => e.currentTarget.style.background = 'rgba(0,0,0,0.1)'}
              onMouseOut={(e) => e.currentTarget.style.background = 'rgba(0,0,0,0.05)'}
              title="Đóng thanh bên"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="3" y1="12" x2="21" y2="12"></line>
                <line x1="3" y1="6" x2="21" y2="6"></line>
                <line x1="3" y1="18" x2="21" y2="18"></line>
              </svg>
            </button>

            <button className="sidebar-new-btn" onClick={onNewSession} style={{ flex: 1, margin: 0 }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 5v14M5 12h14"/>
              </svg>
              Phiên mới
            </button>
          </div>

          <div className="sidebar-title" style={{ marginBottom: '.75rem', paddingLeft: '.25rem' }}>
            Lịch sử
          </div>
          
          <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: "2px", margin: "0 -0.25rem", padding: "0 0.25rem" }}>
            {sessions.length === 0 && (
              <div className="sidebar-empty">
                <div style={{ display: "flex", justifyContent: "center", marginBottom: ".5rem", color: "#64748b" }}>
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M4.5 6.8c0-1.8 1.4-3.3 3.3-3.3h8.4c1.8 0 3.3 1.4 3.3 3.3v5.8c0 1.8-1.4 3.3-3.3 3.3H11l-3.2 2.7c-.5.4-1.3.1-1.3-.6v-2.1H7.8c-1.8 0-3.3-1.4-3.3-3.3V6.8z" />
                    <path d="M10 9.9h5.5" />
                    <path d="M8.8 12.4h4.3" />
                  </svg>
                </div>
                Chưa có cuộc trò chuyện nào.<br />Bắt đầu bằng cách đặt câu hỏi!
              </div>
            )}
            
            {sessions.map((s) => (
              <div
                key={s.id}
                className={"session-item" + (s.id === activeSessionId ? " active" : "")}
                onClick={() => onSelectSession(s.id)}
              >
                <div className="session-icon">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                  </svg>
                </div>
                <div className="session-content">
                  <div className="session-title">{s.title || "Cuộc trò chuyện mới"}</div>
                  <div className="session-meta">{formatDateLabel(s.updatedAt || s.createdAt)}</div>
                </div>
                <button
                  type="button"
                  className="session-delete"
                  onClick={(e) => { e.stopPropagation(); onDeleteSession(s.id); }}
                  title="Xóa"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        </div>
      </aside>
      </>
    );
  }

  function renderMessages() {
    if (!messages.length) {
      return (
        <div className="chat-empty">
          <div className="chat-empty-icon">
            <img
              src={import.meta.env.BASE_URL + "logo.jpg"}
              alt="PharmaAI"
              style={{ width: "100%", height: "100%", objectFit: "cover", borderRadius: "inherit" }}
            />
          </div>
          <div className="chat-empty-title">Chào mừng đến PharmaAI</div>
          <div className="chat-empty-sub">Đặt câu hỏi về dược liệu, tra cứu giá thuốc hoặc pháp lý dược. Tôi sẵn sàng hỗ trợ!</div>
        </div>
      );
    }
    return messages.map((m, idx) => (
      <MessageBubble
        key={idx}
        message={m}
        onFeedback={onFeedback ? (rating, comment) => onFeedback(idx, rating, comment) : undefined}
      />
    ));
  }

  return (
    <>
      {renderSidebar()}

      <section className="main-panel">
        <div className="chat-card">
          {/* Header */}
          <div className="chat-header">
            <div className="chat-header-title">
              {!isSidebarOpen && (
                <button
                  type="button"
                  onClick={() => setIsSidebarOpen(true)}
                  style={{
                    background: 'rgba(0,0,0,0.05)',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    padding: '5px',
                    display: 'flex',
                    alignItems: 'center',
                    color: 'inherit',
                    marginRight: '6px',
                    transition: 'background 0.2s',
                  }}
                  onMouseOver={(e) => e.currentTarget.style.background = 'rgba(0,0,0,0.1)'}
                  onMouseOut={(e) => e.currentTarget.style.background = 'rgba(0,0,0,0.05)'}
                  title="Mở thanh bên"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="3" y1="12" x2="21" y2="12"></line>
                    <line x1="3" y1="6" x2="21" y2="6"></line>
                    <line x1="3" y1="18" x2="21" y2="18"></line>
                  </svg>
                </button>
              )}
              <span className="chat-status-dot" />
              Trợ lý AI
            </div>
            <span className="chat-header-sub">
              {isSending ? "Đang trả lời..." : "Sẵn sàng"}
            </span>
          </div>

          {/* Messages */}
          <div className="chat-body" ref={chatBodyRef}>
            {renderMessages()}
            {isSending && messages.length > 0 && messages[messages.length - 1]?.role === 'user' && (
              <div className="chat-message assistant">
                <div className="typing-indicator">
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                </div>
              </div>
            )}
          </div>

          {/* Footer / Input */}
          <form className="chat-footer" onSubmit={onSend}>
            <div className="chat-input-wrap">
              <textarea
                ref={textareaRef}
                className="chat-input"
                placeholder="Nhập câu hỏi... (Enter để gửi, Shift+Enter xuống dòng)"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isSending}
                rows={1}
              />
            </div>
            {isSending ? (
              <button
                type="button"
                className="chat-cancel-btn"
                onMouseDown={triggerCancel}
                onTouchStart={triggerCancel}
                onClick={triggerCancel}
                title="Hủy"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="4" y="4" width="16" height="16" rx="2"/>
                </svg>
              </button>
            ) : (
              <button
                type="submit"
                className="chat-send-btn"
                disabled={!question.trim()}
                title="Gửi (Enter)"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13"/>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                </svg>
              </button>
            )}
          </form>

          {askError && (
            <div style={{ padding: "0 .8rem .5rem" }}>
              <div className="error-text">⚠ {askError}</div>
            </div>
          )}
        </div>
      </section>
    </>
  );
}

export default ChatPage;
