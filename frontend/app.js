const { useState, useEffect } = React;

const API_BASE = ""; // same origin, e.g. http://localhost:8000

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

function loadSessionsFromStorage() {
  try {
    const raw = window.localStorage.getItem("rag_chat_sessions");
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveSessionsToStorage(list) {
  try {
    window.localStorage.setItem("rag_chat_sessions", JSON.stringify(list));
  } catch {
    // ignore
  }
}

function App() {
  const [sessions, setSessions] = useState(() => loadSessionsFromStorage());
  const [activeSessionId, setActiveSessionId] = useState(
    () => (loadSessionsFromStorage()[0]?.id) || null
  );
  const [messages, setMessages] = useState(() => {
    const first = loadSessionsFromStorage()[0];
    return first?.messages || [];
  });

  const [question, setQuestion] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [askError, setAskError] = useState("");

  const [uploadFile, setUploadFile] = useState(null);
  const [collectionName, setCollectionName] = useState("rag_chatbot");
  const [skipMetadata, setSkipMetadata] = useState(false);
  const [ingestStatus, setIngestStatus] = useState("");
  const [isIngesting, setIsIngesting] = useState(false);

  const [isClearingDb, setIsClearingDb] = useState(false);
  const [clearDbStatus, setClearDbStatus] = useState("");

  const [activePage, setActivePage] = useState("user"); // 'user' | 'admin'

  // Admin state
  const [collections, setCollections] = useState([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);
  const [collectionsError, setCollectionsError] = useState("");
  const [selectedCollection, setSelectedCollection] = useState("");
  const [docs, setDocs] = useState([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [docsError, setDocsError] = useState("");

  // Feedback analytics state
  const [feedbackData, setFeedbackData] = useState(null);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [feedbackTab, setFeedbackTab] = useState("down"); // "down" | "all"

  async function fetchFeedback() {
    setFeedbackLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/admin/feedback`);
      if (!resp.ok) throw new Error("Failed to load feedback");
      const data = await resp.json();
      setFeedbackData(data);
    } catch (err) {
      console.error("Feedback fetch error:", err);
    } finally {
      setFeedbackLoading(false);
    }
  }

  // Persist sessions whenever they change
  useEffect(() => {
    saveSessionsToStorage(sessions);
  }, [sessions]);

  function ensureActiveSession() {
    if (activeSessionId) return activeSessionId;
    const newId = Math.random().toString(36).slice(2, 10);
    const now = new Date().toISOString();
    const newSession = {
      id: newId,
      title: "New chat",
      createdAt: now,
      updatedAt: now,
      messages: [],
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newId);
    setMessages([]);
    return newId;
  }

  function handleSelectSession(id) {
    const s = sessions.find((x) => x.id === id);
    setActiveSessionId(id);
    setMessages(s?.messages || []);
    setAskError("");
  }

  function handleNewSession() {
    const id = Math.random().toString(36).slice(2, 10);
    const now = new Date().toISOString();
    const newSession = {
      id,
      title: "New chat",
      createdAt: now,
      updatedAt: now,
      messages: [],
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(id);
    setMessages([]);
    setAskError("");
  }

  function handleDeleteSession(id) {
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeSessionId === id) {
      const remaining = sessions.filter((s) => s.id !== id);
      const next = remaining[0] || null;
      setActiveSessionId(next ? next.id : null);
      setMessages(next ? next.messages || [] : []);
    }
  }

  function updateActiveSessionMessages(sessionId, nextMessages) {
    setMessages(nextMessages);
    if (!sessionId) return;
    const now = new Date().toISOString();
    const firstUser = nextMessages.find((m) => m.role === "user" && m.content);
    const titleBase = firstUser?.content?.trim() || "New chat";
    const title =
      titleBase.length > 80 ? titleBase.slice(0, 77) + "..." : titleBase;
    setSessions((prev) =>
      prev
        .map((s) =>
          s.id === sessionId
            ? {
                ...s,
                title,
                updatedAt: now,
                messages: nextMessages,
              }
            : s
        )
        .sort((a, b) => (b.updatedAt || b.createdAt || "").localeCompare(a.updatedAt || a.createdAt || ""))
    );
  }

  async function handleSend(e) {
    e.preventDefault();
    if (!question.trim() || isSending) return;
    setAskError("");

    const sid = ensureActiveSession();
    const userMsg = { role: "user", content: question.trim() };
    const nextMessages = [...messages, userMsg];
    updateActiveSessionMessages(sid, nextMessages);
    setQuestion("");

    setIsSending(true);
    try {
      // Build short history window for API (/ask)
      const history = nextMessages
        .slice(0, -1)
        .filter((m) => m.role === "user" || m.role === "assistant")
        .slice(-8)
        .map((m) => ({ role: m.role, content: m.content }));

      const resp = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMsg.content, history }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed with ${resp.status}`);
      }
      const data = await resp.json();
      const fullText = data.answer || "";
      const sources = data.sources || [];

      await animateAnswer(sid, nextMessages, fullText, sources);
    } catch (err) {
      console.error(err);
      setAskError(String(err));
    } finally {
      setIsSending(false);
    }
  }

  function animateAnswer(sessionId, baseMessages, fullText, sources) {
    return new Promise((resolve) => {
      const assistantIndex = baseMessages.length;
      let currentMessages = [
        ...baseMessages,
        { role: "assistant", content: "", sources },
      ];
      updateActiveSessionMessages(sessionId, currentMessages);

      if (!fullText) {
        resolve();
        return;
      }

      let i = 0;
      const total = fullText.length;
      const step = Math.max(1, Math.floor(total / 80));

      function tick() {
        i += step;
        if (i >= total) i = total;
        currentMessages = currentMessages.map((m, idx) =>
          idx === assistantIndex
            ? { ...m, content: fullText.slice(0, i) }
            : m
        );
        updateActiveSessionMessages(sessionId, currentMessages);
        if (i >= total) {
          resolve();
        } else {
          setTimeout(tick, 25);
        }
      }

      tick();
    });
  }

  async function handleIngest(e) {
    e.preventDefault();
    if (!uploadFile) {
      setIngestStatus("Chọn ít nhất một tệp để ingest.");
      return;
    }
    setIsIngesting(true);
    setIngestStatus("Đang tải lên và xử lý...");
    try {
      const form = new FormData();
      form.append("file", uploadFile);
      form.append("collection_name", collectionName || "rag_chatbot");
      form.append("skip_metadata_llm", skipMetadata ? "true" : "false");
      const resp = await fetch(`${API_BASE}/ingest-file`, {
        method: "POST",
        body: form,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed with ${resp.status}`);
      }
      const data = await resp.json();
      if (data.error) {
        setIngestStatus(`Lỗi: ${data.error}`);
      } else {
        setIngestStatus(
          `Đã ingest: ${data.num_parents} parent, ${data.num_children} child → collection ${data.collection_name}`
        );
      }
    } catch (err) {
      console.error(err);
      setIngestStatus(`Lỗi ingest: ${err}`);
    } finally {
      setIsIngesting(false);
    }
  }

  async function handleClearDb() {
    const confirmed = window.confirm(
      "Bạn có chắc muốn xóa toàn bộ database? Toàn bộ collection và tài liệu sẽ bị xóa vĩnh viễn. Hành động này không thể hoàn tác."
    );
    if (!confirmed) return;
    setIsClearingDb(true);
    setClearDbStatus("");
    try {
      const resp = await fetch(`${API_BASE}/db/clear`, { method: "POST" });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed with ${resp.status}`);
      }
      const data = await resp.json();
      setClearDbStatus(data.message || "Đã xóa database, hãy ingest lại tài liệu.");
    } catch (err) {
      console.error(err);
      setClearDbStatus(`Lỗi: ${err}`);
    } finally {
      setIsClearingDb(false);
    }
  }

  async function fetchCollections() {
    setCollectionsLoading(true);
    setCollectionsError("");
    try {
      const resp = await fetch(`${API_BASE}/admin/collections`);
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed with ${resp.status}`);
      }
      const data = await resp.json();
      setCollections(data || []);
    } catch (err) {
      console.error(err);
      setCollectionsError(String(err));
    } finally {
      setCollectionsLoading(false);
    }
  }

  async function fetchDocs(collectionName) {
    if (!collectionName) {
      setDocs([]);
      return;
    }
    setDocsLoading(true);
    setDocsError("");
    try {
      const resp = await fetch(
        `${API_BASE}/admin/docs?collection_name=${encodeURIComponent(
          collectionName
        )}`
      );
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed with ${resp.status}`);
      }
      const data = await resp.json();
      setDocs(data || []);
    } catch (err) {
      console.error(err);
      setDocsError(String(err));
    } finally {
      setDocsLoading(false);
    }
  }

  async function handleDeleteDoc(collectionName, source) {
    if (
      !window.confirm(
        `Xóa toàn bộ nội dung tài liệu '${source}' khỏi collection '${collectionName}'?`
      )
    ) {
      return;
    }
    try {
      const resp = await fetch(`${API_BASE}/admin/docs`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ collection_name: collectionName, source }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed with ${resp.status}`);
      }
      await fetchDocs(collectionName);
    } catch (err) {
      console.error(err);
      setDocsError(String(err));
    }
  }

  async function handleDeleteCollection(name) {
    if (
      !window.confirm(
        `Xóa toàn bộ collection '${name}' (tất cả chunk + metadata)?`
      )
    ) {
      return;
    }
    try {
      const resp = await fetch(
        `${API_BASE}/admin/collections/${encodeURIComponent(name)}`,
        { method: "DELETE" }
      );
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed with ${resp.status}`);
      }
      if (selectedCollection === name) {
        setSelectedCollection("");
        setDocs([]);
      }
      await fetchCollections();
    } catch (err) {
      console.error(err);
      setCollectionsError(String(err));
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
          <button className="btn btn-ghost btn-sm" onClick={handleNewSession}>
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
              onClick={() => handleSelectSession(s.id)}
            >
              <div className="session-title-row">
                <span className="session-title">{s.title}</span>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteSession(s.id);
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

  function handleFeedback(msgIdx, rating, comment) {
    const msg = messages[msgIdx];
    if (!msg || msg.role !== "assistant") return;
    const prevQ = messages.slice(0, msgIdx).reverse().find((m) => m.role === "user");
    const updated = messages.map((m, i) =>
      i === msgIdx ? { ...m, feedback: rating, feedbackComment: comment } : m
    );
    updateActiveSessionMessages(activeSessionId, updated);

    fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: prevQ?.content || "",
        answer: msg.content || "",
        rating,
        comment: comment || "",
        session_id: activeSessionId || "",
      }),
    }).catch((err) => console.error("Feedback submit error:", err));
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
        onFeedback={(rating, comment) => handleFeedback(idx, rating, comment)}
      />
    ));
  }

  return (
    <div className="app-shell">
      <header className="top-nav">
        <div className="top-nav-left">
          <div className="brand-badge">WN</div>
          <div>
            <div className="brand-text-main">Web Nhà Thuốc Chatbot</div>
            <div className="brand-text-sub">
              Trợ lý hỏi đáp tài liệu 
            </div>
          </div>
        </div>
        <div className="top-nav-right">
          <button
            className={
              "btn btn-ghost btn-sm" +
              (activePage === "user" ? " pill" : "")
            }
            onClick={() => setActivePage("user")}
          >
            Người dùng
          </button>
          <button
            className={
              "btn btn-ghost btn-sm" +
              (activePage === "admin" ? " pill" : "")
            }
            onClick={() => {
              setActivePage("admin");
              fetchCollections();
            }}
          >
            Admin
          </button>
        </div>
      </header>

      <main className="layout-main">
        {activePage === "user" ? (
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
                      ? `Phiên: ${activeSessionId.slice(0, 6)}…`
                      : "Chưa có phiên"}
                  </div>
                </div>

                <div className="chat-body">{renderMessages()}</div>

                <form className="chat-footer" onSubmit={handleSend}>
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
        ) : (
          <>
            <aside className="sidebar sidebar--admin">
              <div className="admin-card">
                <h2 className="admin-card-title">Bảng điều khiển Admin</h2>
                <p className="admin-card-caption">
                  Quản lý ingest tài liệu và cơ sở dữ liệu Qdrant.
                </p>
              </div>

              <div className="admin-card">
                <h2 className="admin-card-title">Ingest tài liệu</h2>
                <p className="admin-card-caption">
                  PDF / DOCX để tạo knowledge base.
                </p>
                <form className="admin-form" onSubmit={handleIngest}>
                  <div className="admin-field">
                    <label className="admin-label">Tệp tài liệu</label>
                    <input
                      type="file"
                      accept=".pdf,.doc,.docx"
                      className="admin-input"
                      onChange={(e) =>
                        setUploadFile(e.target.files[0] || null)
                      }
                    />
                  </div>
                  <div className="admin-field">
                    <label className="admin-label">Tên collection</label>
                    <input
                      type="text"
                      className="admin-input"
                      value={collectionName}
                      onChange={(e) => setCollectionName(e.target.value)}
                      placeholder="vd: my_docs"
                    />
                  </div>
                  <div className="admin-field admin-field--row">
                    <input
                      id="skip-meta"
                      type="checkbox"
                      checked={skipMetadata}
                      onChange={(e) => setSkipMetadata(e.target.checked)}
                      className="admin-checkbox"
                    />
                    <label htmlFor="skip-meta" className="admin-label admin-label--inline">
                      Bỏ qua LLM summary (nhanh hơn)
                    </label>
                  </div>
                  <div className="admin-actions">
                    <button
                      type="submit"
                      className="btn btn-primary btn-full"
                      disabled={isIngesting}
                    >
                      {isIngesting ? "Đang ingest..." : "Ingest tài liệu"}
                    </button>
                  </div>
                  {ingestStatus && (
                    <p className="admin-status">{ingestStatus}</p>
                  )}
                </form>
              </div>
            </aside>

            <section className="main-panel main-panel--admin">
              <div className="admin-card admin-card--db">
                <div className="admin-row admin-row--head" style={{ marginBottom: "1rem" }}>
                  <h2 className="admin-card-title" style={{ flex: 1 }}>Quản lý database</h2>
                  <button
                    className="btn btn-ghost btn-sm"
                    type="button"
                    onClick={fetchCollections}
                  >
                    Làm mới
                  </button>
                  <button
                    className="btn btn-danger btn-sm"
                    onClick={handleClearDb}
                    disabled={isClearingDb}
                  >
                    {isClearingDb ? "Đang xóa..." : "Xóa toàn bộ DB"}
                  </button>
                </div>
                {clearDbStatus && (
                  <p className="admin-status" style={{ marginTop: 0, marginBottom: "0.5rem" }}>{clearDbStatus}</p>
                )}
                {collectionsLoading && (
                  <p className="admin-status">Đang tải...</p>
                )}
                {collectionsError && (
                  <p className="admin-error">{collectionsError}</p>
                )}

                <div className="admin-collection-list">
                  {collections.map((c) => (
                    <div key={c.name} className="admin-collection-group">
                      <div
                        className={
                          "admin-collection-header" +
                          (selectedCollection === c.name ? " active" : "")
                        }
                        onClick={() => {
                          if (selectedCollection === c.name) {
                            setSelectedCollection(null);
                            setDocs([]);
                          } else {
                            setSelectedCollection(c.name);
                            fetchDocs(c.name);
                          }
                        }}
                      >
                        <span className="admin-collection-arrow">
                          {selectedCollection === c.name ? "▼" : "▶"}
                        </span>
                        <span className="admin-collection-name">{c.name}</span>
                        <button
                          type="button"
                          className="btn btn-danger btn-sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteCollection(c.name);
                          }}
                        >
                          Xóa
                        </button>
                      </div>

                      {selectedCollection === c.name && (
                        <div className="admin-collection-body">
                          {docsLoading && (
                            <p className="admin-status">Đang tải tài liệu...</p>
                          )}
                          {docsError && (
                            <p className="admin-error">{docsError}</p>
                          )}
                          {!docsLoading && docs.length === 0 && (
                            <p className="admin-status">Chưa có tài liệu nào.</p>
                          )}
                          {!docsLoading && docs.length > 0 && (
                            <ul className="admin-doc-list">
                              {docs.map((d) => (
                                <li key={String(d.source)} className="admin-doc-item">
                                  <span className="admin-doc-name" title={d.source}>
                                    {d.source}
                                  </span>
                                  <span className="admin-doc-meta">
                                    {d.parent_count ?? 0} chunk
                                  </span>
                                  <button
                                    type="button"
                                    className="btn btn-ghost btn-sm"
                                    onClick={() => openDocument(d.source)}
                                    title={`Xem "${d.source}"`}
                                  >
                                    Xem
                                  </button>
                                  <button
                                    type="button"
                                    className="btn btn-danger btn-sm"
                                    onClick={() =>
                                      handleDeleteDoc(selectedCollection, d.source)
                                    }
                                    title={`Xóa "${d.source}" khỏi collection`}
                                  >
                                    Xóa
                                  </button>
                                </li>
                              ))}
                            </ul>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                  {!collections.length && !collectionsLoading && (
                    <p className="admin-empty">Chưa có collection nào.</p>
                  )}
                </div>
              </div>

              <div className="admin-card admin-card--feedback">
                <div className="admin-row admin-row--head" style={{ marginBottom: "1rem" }}>
                  <h2 className="admin-card-title" style={{ flex: 1 }}>Đánh giá người dùng</h2>
                  <button
                    className="btn btn-ghost btn-sm"
                    type="button"
                    onClick={fetchFeedback}
                  >
                    {feedbackLoading ? "Đang tải..." : "Tải dữ liệu"}
                  </button>
                </div>

                {feedbackData && (
                  <>
                    <div className="feedback-stats">
                      <div className="feedback-stat-card feedback-stat--total">
                        <div className="feedback-stat-number">{feedbackData.total}</div>
                        <div className="feedback-stat-label">Tổng đánh giá</div>
                      </div>
                      <div className="feedback-stat-card feedback-stat--up">
                        <div className="feedback-stat-number">{feedbackData.up}</div>
                        <div className="feedback-stat-label">Hữu ích</div>
                      </div>
                      <div className="feedback-stat-card feedback-stat--down">
                        <div className="feedback-stat-number">{feedbackData.down}</div>
                        <div className="feedback-stat-label">Chưa tốt</div>
                      </div>
                      <div className="feedback-stat-card feedback-stat--rate">
                        <div className="feedback-stat-number">
                          {feedbackData.total > 0
                            ? Math.round((feedbackData.up / feedbackData.total) * 100) + "%"
                            : "—"}
                        </div>
                        <div className="feedback-stat-label">Tỉ lệ tốt</div>
                      </div>
                    </div>

                    <div className="feedback-tabs">
                      <button
                        className={"feedback-tab" + (feedbackTab === "down" ? " active" : "")}
                        onClick={() => setFeedbackTab("down")}
                      >
                        Câu đánh giá thấp ({feedbackData.down})
                      </button>
                      <button
                        className={"feedback-tab" + (feedbackTab === "all" ? " active" : "")}
                        onClick={() => setFeedbackTab("all")}
                      >
                        Tất cả
                      </button>
                    </div>

                    <div className="feedback-list">
                      {feedbackTab === "down" && (
                        feedbackData.down_entries.length === 0 ? (
                          <p className="admin-status">Chưa có đánh giá chưa tốt nào.</p>
                        ) : (
                          feedbackData.down_entries.map((entry, i) => (
                            <div key={i} className="feedback-entry feedback-entry--down">
                              <div className="feedback-entry-header">
                                <span className="feedback-entry-rating down">Chưa tốt</span>
                                <span className="feedback-entry-time">{entry.timestamp}</span>
                              </div>
                              <div className="feedback-entry-qa">
                                <div className="feedback-entry-q">
                                  <strong>Câu hỏi:</strong> {entry.question}
                                </div>
                                <div className="feedback-entry-a">
                                  <strong>Trả lời:</strong> {entry.answer || ""}
                                </div>
                              </div>
                              {entry.comment && (
                                <div className="feedback-entry-comment">
                                  <strong>Góp ý:</strong> {entry.comment}
                                </div>
                              )}
                            </div>
                          ))
                        )
                      )}
                      {feedbackTab === "all" && (
                        feedbackData.all_entries.length === 0 ? (
                          <p className="admin-status">Chưa có đánh giá nào.</p>
                        ) : (
                          feedbackData.all_entries.map((entry, i) => (
                            <div key={i} className={"feedback-entry feedback-entry--" + entry.rating}>
                              <div className="feedback-entry-header">
                                <span className={"feedback-entry-rating " + entry.rating}>
                                  {entry.rating === "up" ? "Hữu ích" : "Chưa tốt"}
                                </span>
                                <span className="feedback-entry-time">{entry.timestamp}</span>
                              </div>
                              <div className="feedback-entry-qa">
                                <div className="feedback-entry-q">
                                  <strong>Câu hỏi:</strong> {entry.question}
                                </div>
                                <div className="feedback-entry-a">
                                  <strong>Trả lời:</strong> {entry.answer || ""}
                                </div>
                              </div>
                              {entry.comment && (
                                <div className="feedback-entry-comment">
                                  <strong>Góp ý:</strong> {entry.comment}
                                </div>
                              )}
                            </div>
                          ))
                        )
                      )}
                    </div>
                  </>
                )}

                {!feedbackData && !feedbackLoading && (
                  <p className="admin-status" style={{ textAlign: "center" }}>
                    Nhấn "Tải dữ liệu" để xem thống kê đánh giá.
                  </p>
                )}
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
}

function deduplicateSources(sources) {
  if (!Array.isArray(sources)) return [];
  const seen = new Map();
  for (const s of sources) {
    const key = (s.source || "") + "|" + (s.collection_name || "");
    if (!seen.has(key)) {
      seen.set(key, { ...s, contents: [], pages: [] });
    }
    const entry = seen.get(key);
    if (s.content) entry.contents.push(s.content);
    if (s.page != null && !entry.pages.includes(s.page)) entry.pages.push(s.page);
    if (s.summary && !entry.summary) entry.summary = s.summary;
  }
  return Array.from(seen.values());
}

function isPdf(filename) {
  return /\.pdf$/i.test(filename || "");
}

function openDocument(filename, page) {
  const url = `/uploads/${encodeURIComponent(filename)}`;
  if (isPdf(filename)) {
    const pageParam = page != null ? `#page=${Number(page) + 1}` : "";
    window.open(url + pageParam, "_blank");
  } else {
    window.open(url, "_blank");
  }
}

function DocViewerModal({ source, contents, pages, onClose }) {
  const filename = source || "document";

  return (
    <div className="doc-modal-overlay" onClick={onClose}>
      <div className="doc-modal" onClick={(e) => e.stopPropagation()}>
        <div className="doc-modal-header">
          <div className="doc-modal-title" title={filename}>{filename}</div>
          <div className="doc-modal-actions">
            <a
              className="btn btn-ghost btn-sm"
              href={`/uploads/${encodeURIComponent(filename)}`}
              download
              onClick={(e) => e.stopPropagation()}
            >
              Tải về
            </a>
            <button className="btn btn-ghost btn-sm" onClick={onClose}>✕</button>
          </div>
        </div>
        {pages && pages.length > 0 && (
          <div className="doc-modal-pages">
            Trang tham khảo: {pages.map((p) => Number(p) + 1).join(", ")}
          </div>
        )}
        <div className="doc-modal-body">
          {contents && contents.length > 0 ? (
            contents.map((c, ci) => (
              <div key={ci} className="doc-modal-chunk">
                {contents.length > 1 && (
                  <div className="source-chunk-label">Đoạn {ci + 1}</div>
                )}
                {c}
              </div>
            ))
          ) : (
            <p className="admin-status">Không có nội dung xem trước.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message, onFeedback }) {
  const [showSources, setShowSources] = useState(false);
  const [expandedSource, setExpandedSource] = useState(null);
  const [viewerSource, setViewerSource] = useState(null);
  const [feedbackComment, setFeedbackComment] = useState("");
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [feedbackSending, setFeedbackSending] = useState(false);
  const isUser = message.role === "user";
  const hasSources = Array.isArray(message.sources) && message.sources.length;
  const uniqueSources = hasSources ? deduplicateSources(message.sources) : [];

  const handleRate = (rating) => {
    if (message.feedback === rating) return;
    if (rating === "down") {
      setShowFeedbackForm(true);
      return;
    }
    setShowFeedbackForm(false);
    onFeedback && onFeedback(rating, "");
  };

  const handleSubmitFeedback = () => {
    onFeedback && onFeedback("down", feedbackComment);
    setShowFeedbackForm(false);
  };

  const handleSkipFeedback = () => {
    onFeedback && onFeedback("down", "");
    setShowFeedbackForm(false);
  };

  const formatAssistantText = (text) => {
    if (!text) return "";
    let html = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    html = html.replace(/\s*\[Source\s*\d+\]/gi, "");
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\n/g, "<br/>");
    return html;
  };

  const toggleSource = (idx) => {
    setExpandedSource(expandedSource === idx ? null : idx);
  };

  const handleOpenDoc = (s, e) => {
    e.stopPropagation();
    if (isPdf(s.source)) {
      openDocument(s.source, s.pages && s.pages.length ? s.pages[0] : s.page);
    } else {
      setViewerSource(s);
    }
  };

  return (
    <div className={`chat-message ${isUser ? "user" : "assistant"}`}>
      {isUser ? (
        <div>{message.content}</div>
      ) : (
        <div
          dangerouslySetInnerHTML={{ __html: formatAssistantText(message.content) }}
        />
      )}
      {!isUser && hasSources && (
        <>
          <div
            className="sources-badge"
            onClick={() => setShowSources((prev) => !prev)}
          >
            Nguồn tham khảo ({uniqueSources.length} tài liệu){" "}
            <span>{showSources ? "▲" : "▼"}</span>
          </div>
          {showSources && (
            <div className="sources-panel">
              {uniqueSources.map((s, idx) => (
                <div
                  key={idx}
                  className={"source-card" + (expandedSource === idx ? " expanded" : "")}
                >
                  <div
                    className="source-card-header"
                    onClick={() => toggleSource(idx)}
                  >
                    <span className="source-card-icon">
                      {expandedSource === idx ? "▼" : "▶"}
                    </span>
                    <span
                      className="source-card-name source-card-link"
                      onClick={(e) => handleOpenDoc(s, e)}
                      title="Nhấn để xem tài liệu"
                    >
                      {s.source || "Không rõ nguồn"}
                    </span>
                    {s.collection_name && (
                      <span className="source-card-collection">{s.collection_name}</span>
                    )}
                    <button
                      className="source-card-view-btn"
                      onClick={(e) => handleOpenDoc(s, e)}
                      title="Xem tài liệu"
                    >
                      Xem
                    </button>
                  </div>
                  {s.summary && (
                    <div className="source-card-summary">{s.summary}</div>
                  )}
                  {expandedSource === idx && s.contents && s.contents.length > 0 && (
                    <div className="source-card-content">
                      {s.contents.map((c, ci) => (
                        <div key={ci}>
                          {s.contents.length > 1 && (
                            <div className="source-chunk-label">Đoạn {ci + 1}</div>
                          )}
                          {c}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}
      {!isUser && message.content && (
        <div className="feedback-row">
          <button
            type="button"
            className={"feedback-btn" + (message.feedback === "up" ? " active-up" : "")}
            onClick={() => handleRate("up")}
            title="Hữu ích"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/>
            </svg>
          </button>
          <button
            type="button"
            className={"feedback-btn" + (message.feedback === "down" ? " active-down" : "")}
            onClick={() => handleRate("down")}
            title="Chưa tốt"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10zM17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/>
            </svg>
          </button>
          {message.feedback && (
            <span className="feedback-label">
              {message.feedback === "up" ? "Cảm ơn!" : "Đã ghi nhận"}
            </span>
          )}
        </div>
      )}
      {showFeedbackForm && (
        <div className="feedback-form">
          <textarea
            className="feedback-textarea"
            rows="2"
            placeholder="Góp ý thêm (không bắt buộc)..."
            value={feedbackComment}
            onChange={(e) => setFeedbackComment(e.target.value)}
          />
          <div className="feedback-form-actions">
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={handleSubmitFeedback}
            >
              Gửi góp ý
            </button>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={handleSkipFeedback}
            >
              Bỏ qua
            </button>
          </div>
        </div>
      )}
      {viewerSource && (
        <DocViewerModal
          source={viewerSource.source}
          contents={viewerSource.contents}
          pages={viewerSource.pages}
          onClose={() => setViewerSource(null)}
        />
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

