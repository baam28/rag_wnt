import React, { useState, useEffect, useRef } from 'react';
import LandingPage from './landing.js';
import ChatPage from './chat_page.js';
import AdminPage from './admin_page.js';

const API_BASE = ""; // same origin

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

function getStoredToken() {
  try {
    return window.localStorage.getItem("rag_jwt") || null;
  } catch {
    return null;
  }
}

function App() {
  const [jwtToken, setJwtToken] = useState(() => getStoredToken());
  const [username, setUsername] = useState("");
  const [isAdmin, setIsAdmin] = useState(false);
  const [route, setRoute] = useState(() => {
    const path = window.location.pathname;
    if (path.startsWith("/admin")) return "admin";
    return "chat";
  });

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

  // Admin state
  const [uploadFile, setUploadFile] = useState(null);
  const [collectionName, setCollectionName] = useState("");
  const [newCollectionMode, setNewCollectionMode] = useState(false);
  const [newCollectionName, setNewCollectionName] = useState("");
  const [ingestStatus, setIngestStatus] = useState("");
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestProgress, setIngestProgress] = useState(null);
  const [currentJobId, setCurrentJobId] = useState(null);
  const [skipSummary, setSkipSummary] = useState(false);
  const ingestPollRef = useRef(null);

  const [isClearingDb, setIsClearingDb] = useState(false);
  const [clearDbStatus, setClearDbStatus] = useState("");

  const [collections, setCollections] = useState([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);
  const [collectionsError, setCollectionsError] = useState("");
  const [selectedCollection, setSelectedCollection] = useState("");
  const [docs, setDocs] = useState([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [docsError, setDocsError] = useState("");

  const [feedbackData, setFeedbackData] = useState(null);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [feedbackTab, setFeedbackTab] = useState("down");

  function syncUserFromToken(token) {
    if (!token) {
      setIsAdmin(false);
      setUsername("");
      return;
    }
    try {
      const payload = JSON.parse(atob(token.split(".")[1]));
      setIsAdmin(!!payload.admin);
      setUsername(payload.email || payload.username || "");
    } catch {
      setIsAdmin(false);
      setUsername("");
    }
  }

  useEffect(() => {
    syncUserFromToken(jwtToken);
  }, [jwtToken]);

  useEffect(() => {
    if (!jwtToken) return;
    (async () => {
      try {
        const resp = await authFetch(`${API_BASE}/chat/sessions`);
        if (!resp.ok) return;
        const data = await resp.json();
        setSessions(
          data.map((s) => ({
            id: s.id,
            title: s.title,
            createdAt: s.created_at,
            updatedAt: s.updated_at,
            messages: [],
          }))
        );
        if (data.length > 0) {
          const firstId = data[0].id;
          setActiveSessionId(firstId);
          const mResp = await authFetch(
            `${API_BASE}/chat/sessions/${firstId}/messages`
          );
          if (mResp.ok) {
            const msgs = await mResp.json();
            const fetchedMessages = msgs.map((m) => ({ 
              role: m.role, 
              content: m.content,
              sources: m.sources,
              priceData: m.priceData,
              feedback: m.feedback,
              feedbackComment: m.feedbackComment
            }));
            setMessages(fetchedMessages);
            setSessions(prev => 
              prev.map(s => s.id === firstId ? { ...s, messages: fetchedMessages } : s)
            );
          }
        } else {
          setActiveSessionId(null);
          setMessages([]);
        }
      } catch (e) {
        console.error("Load server sessions error", e);
      }
    })();
  }, [jwtToken]);

  useEffect(() => {
    if (!jwtToken) {
      saveSessionsToStorage(sessions);
    }
  }, [sessions, jwtToken]);

  useEffect(() => {
    // Resume ingest polling if there's a stored jobId
    const storedJobId = window.localStorage.getItem("rag_ingest_job_id");
    if (storedJobId) {
      startIngestPolling(storedJobId);
    }
    return () => {
      if (ingestPollRef.current) clearInterval(ingestPollRef.current);
    };
  }, []);

  async function startIngestPolling(jobId) {
    if (ingestPollRef.current) clearInterval(ingestPollRef.current);
    setIsIngesting(true);
    setCurrentJobId(jobId);
    window.localStorage.setItem("rag_ingest_job_id", jobId);

    const poll = async () => {
      try {
        const r = await authFetch(`${API_BASE}/ingest-jobs/${jobId}`);
        if (!r.ok) {
          if (r.status === 404) stopIngestPolling();
          return;
        }
        const j = await r.json();
        setIngestProgress({
          phase: j.phase || j.status,
          message: j.message || j.status,
          current: j.current ?? 0,
          total: j.total ?? 1,
        });

        if (j.status === "done") {
          stopIngestPolling();
          const res = j.result;
          setIngestStatus(
            `Đã ingest: ${res.num_parents} parent, ${res.num_children} child → collection ${res.collection_name}`
          );
          await fetchCollections();
          setNewCollectionMode(false);
          setNewCollectionName("");
          setCollectionName(res.collection_name);
        } else if (j.status === "error" || j.status === "cancelled") {
          const statusMsg = j.status === "cancelled" ? "Đã hủy." : `Lỗi ingest: ${j.error || "Unknown"}`;
          stopIngestPolling();
          setIngestStatus(statusMsg);
        }
      } catch (e) {
        console.error("Polling error:", e);
      }
    };

    await poll();
    ingestPollRef.current = setInterval(poll, 2000);
  }

  function stopIngestPolling() {
    if (ingestPollRef.current) clearInterval(ingestPollRef.current);
    ingestPollRef.current = null;
    setIsIngesting(false);
    setCurrentJobId(null);
    setIngestProgress(null);
    window.localStorage.removeItem("rag_ingest_job_id");
  }

  function authFetch(url, options = {}) {
    const headers = options.headers ? { ...options.headers } : {};
    if (jwtToken) {
      headers["Authorization"] = `Bearer ${jwtToken}`;
    }
    return fetch(url, { ...options, headers }).then(async (resp) => {
      if (resp.status === 401 || resp.status === 403) {
        // token invalid -> logout
        handleLogout();
      }
      return resp;
    });
  }

  async function fetchFeedback() {
    setFeedbackLoading(true);
    try {
      const resp = await authFetch(`${API_BASE}/admin/feedback`);
      if (!resp.ok) throw new Error("Failed to load feedback");
      const data = await resp.json();
      setFeedbackData(data);
    } catch (err) {
      console.error("Feedback fetch error:", err);
    } finally {
      setFeedbackLoading(false);
    }
  }

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

  async function handleSelectSession(id) {
    const s = sessions.find((x) => x.id === id);
    setActiveSessionId(id);

    if (jwtToken && s && s.messages.length === 0) {
      try {
        const mResp = await authFetch(`${API_BASE}/chat/sessions/${id}/messages`);
        if (mResp.ok) {
          const msgs = await mResp.json();
          const fetchedMessages = msgs.map((m) => ({ 
            role: m.role, 
            content: m.content,
            sources: m.sources,
            priceData: m.priceData,
            feedback: m.feedback,
            feedbackComment: m.feedbackComment
          }));
          setMessages(fetchedMessages);
          setSessions((prev) => 
            prev.map((x) => (x.id === id ? { ...x, messages: fetchedMessages } : x))
          );
          setAskError("");
          return;
        }
      } catch (err) {
        console.error("Failed to fetch messages for session", id, err);
      }
    }

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

      const resp = await authFetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userMsg.content,
          history,
          session_id: sid,
        }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed with ${resp.status}`);
      }
      const data = await resp.json();
      const fullText = data.answer || "";
      const sources = data.sources || [];
      const priceData = data.price_data || null;

      await animateAnswer(sid, nextMessages, fullText, sources, priceData);
    } catch (err) {
      console.error(err);
      setAskError(String(err));
    } finally {
      setIsSending(false);
    }
  }

  function animateAnswer(sessionId, baseMessages, fullText, sources, priceData) {
    return new Promise((resolve) => {
      const assistantIndex = baseMessages.length;
      let currentMessages = [
        ...baseMessages,
        { role: "assistant", content: "", sources, priceData },
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

  useEffect(() => {
    return () => {
      if (ingestPollRef.current) clearInterval(ingestPollRef.current);
    };
  }, []);

  async function handleIngest(e) {
    e.preventDefault();
    if (!uploadFile) {
      setIngestStatus("Chọn ít nhất một tệp để ingest.");
      return;
    }
    const targetCollection = collectionName || "drug";
    if (!targetCollection) {
      setIngestStatus("Vui lòng chọn hoặc tạo một collection.");
      return;
    }
    setIsIngesting(true);
    setIngestStatus("Đang tải lên...");
    setIngestProgress({ phase: "upload", message: "Đang tải lên file...", current: 0, total: 1 });
    try {
      const form = new FormData();
      form.append("file", uploadFile);
      form.append("collection_name", targetCollection);
      form.append("skip_summary", skipSummary ? "true" : "false");
      const resp = await authFetch(`${API_BASE}/ingest-file?async=true`, {
        method: "POST",
        body: form,
      });
      const data = await resp.json().catch(() => ({}));
      if (resp.status === 202 && data.job_id) {
        setIngestStatus("Đang xử lý (job chạy nền)...");
        await startIngestPolling(data.job_id);
      } else if (!resp.ok) {
        throw new Error(data.detail || `Request failed with ${resp.status}`);
      } else {
        if (data.error) {
          setIngestStatus(`Lỗi: ${data.error}`);
        } else {
          setIngestStatus(
            `Đã ingest: ${data.num_parents} parent, ${data.num_children} child → collection ${data.collection_name}`
          );
          await fetchCollections();
          setNewCollectionMode(false);
          setNewCollectionName("");
          setCollectionName(data.collection_name || targetCollection);
        }
        setIngestProgress(null);
        setIsIngesting(false);
      }
    } catch (err) {
      console.error(err);
      setIngestStatus(`Lỗi ingest: ${err}`);
      setIngestProgress(null);
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
      const resp = await authFetch(`${API_BASE}/db/clear`, { method: "POST" });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed with ${resp.status}`);
      }
      const data = await resp.json();
      setClearDbStatus(data.message || "Đã xóa database, hãy ingest lại tài liệu.");
      await fetchCollections();
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
      const resp = await authFetch(`${API_BASE}/admin/collections`);
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed with ${resp.status}`);
      }
      const data = await resp.json();
      setCollections(data || []);
      if (data && data.length > 0) {
        if (!collectionName && !newCollectionMode) {
          setCollectionName(data[0].name);
        }
      } else {
        // No collections: default to "create new" so the name input is visible
        setNewCollectionMode(true);
      }
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
      const resp = await authFetch(
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
      const resp = await authFetch(`${API_BASE}/admin/docs`, {
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
      const resp = await authFetch(
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

  function handleFeedback(msgIdx, rating, comment) {
    const msg = messages[msgIdx];
    if (!msg || msg.role !== "assistant") return;
    const prevQ = messages.slice(0, msgIdx).reverse().find((m) => m.role === "user");
    const updated = messages.map((m, i) =>
      i === msgIdx ? { ...m, feedback: rating, feedbackComment: comment } : m
    );
    updateActiveSessionMessages(activeSessionId, updated);

    authFetch(`${API_BASE}/feedback`, {
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

  function handleLoginSuccess(token, info) {
    setJwtToken(token);
    syncUserFromToken(token);
    const newIsAdmin = info ? info.is_admin : false;
    const dest = newIsAdmin ? "admin" : "chat";
    setRoute(dest);
    const path = dest === "admin" ? "/admin/" : "/app/";
    if (!window.location.pathname.startsWith(path.slice(0, -1))) {
      window.history.replaceState(null, "", path);
    }
  }

  function handleLogout() {
    setJwtToken(null);
    setIsAdmin(false);
    setUsername("");
    try {
      window.localStorage.removeItem("rag_jwt");
    } catch {
      // ignore
    }
    setRoute("chat");
    if (!window.location.pathname.startsWith("/app")) {
      window.history.replaceState(null, "", "/app/");
    }
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
          {jwtToken && !isAdmin && (
            <>
            </>
          )}
          {!jwtToken ? null : (
            <>
              <span className="user-info" style={{ marginLeft: "0.5rem" }}>
                <button
                  className={"btn btn-ghost btn-sm" + (route === "account" ? " pill" : "")}
                  type="button"
                  onClick={() => {
                    setRoute("account");
                    if (!window.location.pathname.startsWith("/account")) {
                      window.history.replaceState(null, "", "/account/");
                    }
                  }}
                  title="Quản lý tài khoản"
                >
                  {username}
                </button>
              </span>
              <button
                className="btn btn-ghost btn-sm"
                type="button"
                onClick={handleLogout}
                style={{ marginLeft: "0.5rem" }}
                title="Đăng xuất"
              >
                Đăng xuất
              </button>
            </>
          )}
        </div>
      </header>

      <main className="layout-main">
        {!jwtToken ? (
          <LandingPage onLoggedIn={handleLoginSuccess} />
        ) : route === "account" ? (
          <UserAccountPage username={username} onLogout={handleLogout} onBack={() => {
            const dest = isAdmin ? "admin" : "chat";
            setRoute(dest);
            const path = dest === "admin" ? "/admin/" : "/app/";
            if (!window.location.pathname.startsWith(path.slice(0, -1))) {
              window.history.replaceState(null, "", path);
            }
          }} />
        ) : route === "admin" && isAdmin ? (
          <AdminPage
            state={{
              collections,
              collectionsLoading,
              collectionsError,
              selectedCollection,
              docs,
              docsLoading,
              docsError,
              uploadFile,
              collectionName,
              newCollectionMode,
              newCollectionName,
              skipSummary,
              ingestStatus,
              ingestProgress,
              isIngesting,
              currentJobId,
              isClearingDb,
              clearDbStatus,
              feedbackData,
              feedbackLoading,
              feedbackTab,
            }}
            handlers={{
              setUploadFile,
              setCollectionName,
              setNewCollectionMode,
              setNewCollectionName,
              setSkipSummary,
              setSelectedCollection,
              setFeedbackTab,
              fetchCollections,
              fetchDocs,
              handleIngest,
              handleClearDb,
              handleDeleteCollection,
              handleDeleteDoc,
              fetchFeedback,
              authFetch,
            }}
          />
        ) : isAdmin ? (
          <AdminPage
            state={{
              collections,
              collectionsLoading,
              collectionsError,
              selectedCollection,
              docs,
              docsLoading,
              docsError,
              uploadFile,
              collectionName,
              newCollectionMode,
              newCollectionName,
              skipSummary,
              ingestStatus,
              ingestProgress,
              isIngesting,
              currentJobId,
              isClearingDb,
              clearDbStatus,
              feedbackData,
              feedbackLoading,
              feedbackTab,
            }}
            handlers={{
              setUploadFile,
              setCollectionName,
              setNewCollectionMode,
              setNewCollectionName,
              setSkipSummary,
              setSelectedCollection,
              setFeedbackTab,
              fetchCollections,
              fetchDocs,
              handleIngest,
              handleClearDb,
              handleDeleteCollection,
              handleDeleteDoc,
              fetchFeedback,
              authFetch,
            }}
          />
        ) : (
          <ChatPage
            sessions={sessions}
            activeSessionId={activeSessionId}
            messages={messages}
            onNewSession={handleNewSession}
            onSelectSession={handleSelectSession}
            onDeleteSession={handleDeleteSession}
            onSend={handleSend}
            question={question}
            setQuestion={setQuestion}
            isSending={isSending}
            askError={askError}
          />
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

export function MessageBubble({ message, onFeedback }) {
  const [showSources, setShowSources] = useState(false);
  const [expandedSource, setExpandedSource] = useState(null);
  const [expandedDrugs, setExpandedDrugs] = useState(new Set());
  const [viewerSource, setViewerSource] = useState(null);
  const [feedbackComment, setFeedbackComment] = useState("");
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [feedbackSending, setFeedbackSending] = useState(false);
  const isUser = message.role === "user";

  const toggleDrugExpand = (drugName) => {
    setExpandedDrugs((prev) => {
      const next = new Set(prev);
      if (next.has(drugName)) next.delete(drugName);
      else next.add(drugName);
      return next;
    });
  };
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

  const priceData = message.priceData || null;

  return (
    <div className={`chat-message ${isUser ? "user" : "assistant"}`}>
      {isUser ? (
        <div>{message.content}</div>
      ) : (
        <div
          dangerouslySetInnerHTML={{ __html: formatAssistantText(message.content) }}
        />
      )}
      {!isUser && priceData && (priceData.drugs?.length > 0 || (priceData.prices?.length > 0 && !priceData.is_prescription)) && (() => {
        const drugs = priceData.drugs?.length > 0
          ? priceData.drugs
          : (() => {
              const byName = {};
              for (const p of priceData.prices || []) {
                const name = p.drug_name || "—";
                if (!byName[name]) byName[name] = { drug_name: name, options: [], price_raw: null };
                const raw = p.price_raw != null ? p.price_raw : parseInt(String(p.price).replace(/[\s.]/g, ""), 10) || 0;
                byName[name].options.push({ unit: p.unit, price: p.price, price_raw: raw, source_name: p.source_name, source_url: p.source_url });
              }
              return Object.values(byName).map((d) => {
                d.options.sort((a, b) => (a.price_raw || 0) - (b.price_raw || 0));
                const c = d.options[0];
                return { drug_name: d.drug_name, options: d.options, cheapest: { unit: c.unit, price: c.price, source_name: c.source_name, source_url: c.source_url } };
              });
            })();
        return (
          <div className="price-list">
            <div className="price-list-header">
              <span className="price-list-title">{priceData.drug_name || "Giá thuốc"}</span>
              {priceData.is_prescription && <span className="price-list-rx">Rx</span>}
            </div>
            {priceData.price_range && (
              <div className="price-list-range">{priceData.price_range}</div>
            )}
            <ul className="price-list-drugs">
              {drugs.map((d, di) => (
                <li key={di} className="price-list-drug">
                  <div className="price-list-drug-row">
                    <span className="price-list-drug-name">{d.drug_name}</span>
                    <span className="price-list-drug-cheapest">
                      {d.cheapest.price}/{d.cheapest.unit}
                    </span>
                    {d.cheapest.source_url && (
                      <a className="price-list-link" href={d.cheapest.source_url} target="_blank" rel="noopener noreferrer" title="Xem tại nhà thuốc">↗</a>
                    )}
                    {d.options.length > 1 && (
                      <button type="button" className="price-list-expand-btn" onClick={() => toggleDrugExpand(d.drug_name)} aria-expanded={expandedDrugs.has(d.drug_name)}>
                        {expandedDrugs.has(d.drug_name) ? "Thu gọn" : "Xem thêm"}
                      </button>
                    )}
                  </div>
                  {expandedDrugs.has(d.drug_name) && d.options.length > 1 && (
                    <ul className="price-list-options">
                      {d.options.map((opt, oi) => (
                        <li key={oi} className="price-list-option">
                          <span className="price-list-option-price">{opt.price}/{opt.unit}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </li>
              ))}
            </ul>
            {priceData.disclaimer && (
              <div className="price-list-disclaimer">{priceData.disclaimer}</div>
            )}
          </div>
        );
      })()}
      {!isUser && priceData && priceData.is_prescription && (!priceData.prices || priceData.prices.length === 0) && (
        <div className="price-card price-card--rx">
          <div className="price-card-header">
            <span className="price-card-title">{priceData.drug_name || "Thuốc kê đơn"}</span>
            <span className="price-card-rx">Rx</span>
          </div>
          <div className="price-card-range">{priceData.notes || "Thuốc kê đơn – giá không niêm yết."}</div>
          {priceData.disclaimer && (
            <div className="price-card-disclaimer">{priceData.disclaimer}</div>
          )}
        </div>
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

export { UserAccountPage };
export default App;

function UserAccountPage({ username, onLogout, onBack }) {
  const [oldPassword, setOldPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleChangePassword(e) {
    e.preventDefault();
    if (!oldPassword || !newPassword) {
      setError("Vui lòng nhập đầy đủ mật khẩu cũ và mới.");
      return;
    }
    if (newPassword.length < 6) {
      setError("Mật khẩu mới phải có ít nhất 6 ký tự.");
      return;
    }

    setError("");
    setMessage("");
    setIsLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/auth/password`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${getStoredToken()}`,
        },
        body: JSON.stringify({
          old_password: oldPassword,
          new_password: newPassword,
        }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || "Đổi mật khẩu thất bại");
      }
      setMessage("Đổi mật khẩu thành công!");
      setOldPassword("");
      setNewPassword("");
    } catch (err) {
      console.error(err);
      setError(String(err.message));
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="account-page">
      <div className="account-container">
        {onBack && (
          <button 
            type="button" 
            className="btn btn-ghost btn-sm" 
            onClick={onBack}
            style={{ position: "absolute", top: "1.5rem", left: "1.5rem" }}
          >
            ❮ Chat
          </button>
        )}
        <h2 style={{ textAlign: "center", marginTop: onBack ? "0.5rem" : "0" }}>Tài khoản của tôi</h2>
        <div className="account-info" style={{ textAlign: "center", marginBottom: "0.5rem" }}>
          <p><strong>Tên đăng nhập:</strong> {username}</p>
        </div>

        <form onSubmit={handleChangePassword} className="password-form">
          <h3>Đổi mật khẩu</h3>
          <div className="form-group">
            <label>Mật khẩu hiện tại</label>
            <input
              type="password"
              value={oldPassword}
              onChange={(e) => setOldPassword(e.target.value)}
              placeholder="Nhập mật khẩu hiện tại"
            />
          </div>
          <div className="form-group">
            <label>Mật khẩu mới</label>
            <input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              placeholder="Nhập mật khẩu mới (ít nhất 6 ký tự)"
            />
          </div>
          {error && <div className="text-danger" style={{ fontSize: "0.85rem", marginTop: "-0.5rem" }}>{error}</div>}
          {message && <div className="text-success" style={{ fontSize: "0.85rem", marginTop: "-0.5rem", color: "#16a34a" }}>{message}</div>}
          
          <button type="submit" className="btn btn-primary" disabled={isLoading}>
            {isLoading ? "Đang xử lý..." : "Cập nhật mật khẩu"}
          </button>
        </form>

        <div className="account-actions" style={{ marginTop: "2rem", paddingTop: "1.5rem", borderTop: "1px solid #e2e8f0" }}>
          <button type="button" className="btn btn-ghost btn-sm" onClick={onLogout} style={{ color: "#ef4444", borderColor: "#fca5a5", backgroundColor: "#fef2f2" }}>
            Đăng xuất
          </button>
        </div>
      </div>
    </div>
  );
}

