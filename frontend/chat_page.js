import React from 'react';
import { MessageBubble } from './app.js';

function ChatPage({
  sessions,
  activeSessionId,
  messages,
  onNewSession,
  onSelectSession,
  onDeleteSession,
  onSend,
  onFeedback,
  question,
  setQuestion,
  isSending,
  askError,
}) {
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

  function renderSidebarSessions() {
    return (
      <div className="section-card">
        <div className="section-title">Phiên trò chuyện</div>
        <div className="section-caption">
          Lưu và mở lại các phiên chat trước đó.
        </div>
        <div className="session-actions">
          <button className="btn btn-ghost btn-sm" onClick={onNewSession}>
            + Phiên mới
          </button>
          <span className="badge-small">
            {sessions.length ? `${sessions.length} phiên` : "Chưa có phiên"}
          </span>
        </div>
        <div className="sessions-list">
          {sessions.map((s) => (
            <div
              key={s.id}
              className={
                "session-item" + (s.id === activeSessionId ? " active" : "")
              }
              onClick={() => onSelectSession(s.id)}
            >
              <div className="session-title-row">
                <span className="session-title">{s.title}</span>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteSession(s.id);
                  }}
                >
                  ✕
                </button>
              </div>
              <div className="session-meta">
                {formatDateLabel(s.updatedAt || s.createdAt)}
              </div>
            </div>
          ))}
          {!sessions.length && (
            <div className="session-meta" style={{ marginTop: "0.25rem" }}>
              Chưa có lịch sử. Hãy bắt đầu 1 câu hỏi mới.
            </div>
          )}
        </div>
      </div>
    );
  }

  function renderMessages() {
    if (!messages.length) {
      return (
        <div className="status-text">
          Hãy đặt câu hỏi đầu tiên của bạn về tài liệu.
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
      <aside className="sidebar">{renderSidebarSessions()}</aside>

      <section className="main-panel">
        <div className="chat-card">
          <div className="chat-header">
            <div>
              <div className="chat-header-title">Chatbot</div>
            </div>
            <div className="badge-small">
              {activeSessionId
                ? `Phiên: ${String(activeSessionId).slice(0, 6)}…`
                : "Chưa có phiên"}
            </div>
          </div>

          <div className="chat-body">{renderMessages()}</div>

          <form className="chat-footer" onSubmit={onSend}>
            <input
              className="chat-input"
              placeholder="Nhập câu hỏi ..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={isSending}
            />
            <button
              type="submit"
              className="btn btn-primary"
              disabled={isSending || !question.trim()}
            >
              {isSending ? "Đang trả lời..." : "Gửi"}
            </button>
          </form>
          {askError && (
            <div style={{ padding: "0 0.9rem 0.5rem" }}>
              <div className="error-text">{askError}</div>
            </div>
          )}
        </div>
      </section>
    </>
  );
}

export default ChatPage;

