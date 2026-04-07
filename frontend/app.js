import React, { useState, useEffect, useRef } from 'react';
import LandingPage from './landing.js';
import ChatPage from './chat_page.js';
import AdminPage from './admin_page.js';

const API_BASE = ""; // same origin
const HISTORY_MAX_TURNS = 4;
const HISTORY_MAX_MESSAGES = HISTORY_MAX_TURNS * 2;
const HISTORY_MAX_CHARS = 800;

function generateUUID() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback for older browsers
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
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

function getStoredToken() {
  try {
    return window.localStorage.getItem("rag_jwt") || null;
  } catch {
    return null;
  }
}

function sanitizeHistoryForApi(messageList) {
  return (messageList || [])
    .filter((m) => (m.role === "user" || m.role === "assistant") && m.content)
    .slice(-HISTORY_MAX_MESSAGES)
    .map((m) => ({
      role: m.role,
      content: String(m.content).slice(0, HISTORY_MAX_CHARS),
    }));
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
  const [uploadFiles, setUploadFiles] = useState([]);
  const [collectionName, setCollectionName] = useState("");
  const [newCollectionMode, setNewCollectionMode] = useState(false);
  const [newCollectionName, setNewCollectionName] = useState("");
  const [ingestStatus, setIngestStatus] = useState("");
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestProgress, setIngestProgress] = useState(null);
  const [currentJobId, setCurrentJobId] = useState(null);

  const ingestPollRef = useRef(null);
  const abortControllerRef = useRef(null);
  const inFlightControllersRef = useRef(new Map());
  const sendLockRef = useRef(false);
  const activeRequestIdRef = useRef(0);
  const pendingQuestionRef = useRef("");
  const pendingBaseMessagesRef = useRef([]);
  const pendingSessionIdRef = useRef(null);
  const cancelAnimRef = useRef(false);

  const [collections, setCollections] = useState([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);
  const [collectionsError, setCollectionsError] = useState("");
  const [selectedCollection, setSelectedCollection] = useState("");
  const [docs, setDocs] = useState([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [docsError, setDocsError] = useState("");
  const [docsQuery, setDocsQuery] = useState({
    search: "",
    sort_by: "document_date",
    sort_order: "desc",
    year_from: "",
    year_to: "",
  });

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
      setUsername(payload.username || payload.email || "");
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
          const res = j.result || {};
          const fileCount = Number(res.file_count || (Array.isArray(res.files) ? res.files.length : 0) || 1);
          setIngestStatus(
            fileCount > 1
              ? `Đã ingest ${fileCount} tệp: ${res.num_parents || 0} parent, ${res.num_children || 0} child → collection ${res.collection_name || collectionName || "drug"}`
              : `Đã ingest: ${res.num_parents || 0} parent, ${res.num_children || 0} child → collection ${res.collection_name || collectionName || "drug"}`
          );
          await fetchCollections();
          setNewCollectionMode(false);
          setNewCollectionName("");
          setCollectionName(res.collection_name || collectionName || "drug");
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
    const newId = generateUUID();
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
    cancelInFlight({ restoreInput: false });
    setActiveSessionId(id);
    setMessages(s?.messages || []);
    setQuestion("");
    setAskError("");

    // Always try to fetch fresh messages from server when switching sessions
    if (jwtToken) {
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
            feedbackComment: m.feedbackComment,
          }));
          setMessages(fetchedMessages);
          setSessions((prev) =>
            prev.map((x) => (x.id === id ? { ...x, messages: fetchedMessages } : x))
          );
          return;
        }
      } catch (err) {
        console.error("Failed to fetch messages for session", id, err);
      }
    }

    // Fallback to in-memory messages
    setMessages(s?.messages || []);
  }

  function handleNewSession() {
    cancelInFlight({ restoreInput: false });
    const id = generateUUID();
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

  async function handleDeleteSession(id) {
    if (activeSessionId === id) {
      cancelInFlight({ restoreInput: false });
    }
    // Optimistic UI update first
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeSessionId === id) {
      const remaining = sessions.filter((s) => s.id !== id);
      const next = remaining[0] || null;
      setActiveSessionId(next ? next.id : null);
      setMessages(next ? next.messages || [] : []);
    }
    // Persist deletion on the server
    if (jwtToken) {
      try {
        await authFetch(`${API_BASE}/chat/sessions/${id}`, { method: "DELETE" });
      } catch (err) {
        console.error("Failed to delete session on server:", err);
      }
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

  function cancelInFlight({ restoreInput = false } = {}) {
    activeRequestIdRef.current += 1;
    cancelAnimRef.current = true;
    for (const [, ctrl] of inFlightControllersRef.current) {
      try {
        ctrl.abort();
      } catch {}
    }
    inFlightControllersRef.current.clear();
    sendLockRef.current = false;
    abortControllerRef.current = null;

    const q = pendingQuestionRef.current || "";
    const base = Array.isArray(pendingBaseMessagesRef.current) ? pendingBaseMessagesRef.current : [];
    const pendingSessionId = pendingSessionIdRef.current;
    pendingQuestionRef.current = "";
    pendingBaseMessagesRef.current = [];
    pendingSessionIdRef.current = null;
    setAskError("");
    setIsSending(false);

    if (restoreInput) {
      setQuestion(q);
      updateActiveSessionMessages(pendingSessionId || activeSessionId, base);
    } else {
      setQuestion("");
    }
  }

  async function handleSend(e) {
    e.preventDefault();
    if (!question.trim() || isSending || abortControllerRef.current || sendLockRef.current) return;
    sendLockRef.current = true;
    setAskError("");

    const requestId = ++activeRequestIdRef.current;
    const originalQuestion = question.trim();
    pendingQuestionRef.current = originalQuestion;
    pendingBaseMessagesRef.current = [...messages];
    cancelAnimRef.current = false;

    const sid = ensureActiveSession();
    pendingSessionIdRef.current = sid;
    const userMsg = { role: "user", content: originalQuestion };
    const nextMessages = [...messages, userMsg];
    updateActiveSessionMessages(sid, nextMessages);
    setQuestion("");

    const controller = new AbortController();
    abortControllerRef.current = controller;
    inFlightControllersRef.current.set(requestId, controller);
    setIsSending(true);
    try {
      // Build short history window for API (/ask)
      const history = sanitizeHistoryForApi(nextMessages.slice(0, -1));

      const resp = await authFetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userMsg.content,
          history,
          session_id: sid,
        }),
        signal: controller.signal,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed with ${resp.status}`);
      }
      if (requestId !== activeRequestIdRef.current) return;
      const data = await resp.json();
      if (requestId !== activeRequestIdRef.current) return;
      const fullText = data.answer || "";
      const sources = data.sources || [];
      const priceData = data.price_data || null;

      // Sync to the server-assigned session ID (handles any mismatch)
      const serverSessionId = data.session_id || sid;
      if (serverSessionId !== sid) {
        setActiveSessionId(serverSessionId);
        setSessions((prev) =>
          prev.map((s) => (s.id === sid ? { ...s, id: serverSessionId } : s))
        );
        pendingSessionIdRef.current = serverSessionId;
      }
      const effectiveSid = serverSessionId;

      if (!cancelAnimRef.current && requestId === activeRequestIdRef.current) {
        await animateAnswer(effectiveSid, nextMessages, fullText, sources, priceData);
      }
    } catch (err) {
      if (requestId !== activeRequestIdRef.current) return;
      if (err.name === "AbortError" || controller.signal.aborted) {
        // cancelled — question already restored by handleCancel
        return;
      }
      console.error(err);
      setAskError(String(err));
    } finally {
      inFlightControllersRef.current.delete(requestId);
      if (requestId === activeRequestIdRef.current && abortControllerRef.current === controller) {
        abortControllerRef.current = null;
        setIsSending(false);
      }
      if (requestId === activeRequestIdRef.current) {
        sendLockRef.current = false;
      }
    }
  }

  function handleCancel(e) {
    if (e) {
      e.preventDefault?.();
      e.stopPropagation?.();
    }
    cancelInFlight({ restoreInput: true });
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
        if (cancelAnimRef.current) {
          resolve();
          return;
        }
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
    if (!uploadFiles || uploadFiles.length === 0) {
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
    setIngestProgress({ phase: "upload", message: "Đang tải lên file...", current: 0, total: uploadFiles.length });
    try {
      const form = new FormData();
      uploadFiles.forEach((file) => form.append("file", file));
      form.append("collection_name", targetCollection);
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
          const fileCount = Number(data.file_count || (Array.isArray(data.files) ? data.files.length : 0) || 1);
          setIngestStatus(
            fileCount > 1
              ? `Đã ingest ${fileCount} tệp: ${data.num_parents || 0} parent, ${data.num_children || 0} child → collection ${data.collection_name || targetCollection}`
              : `Đã ingest: ${data.num_parents || 0} parent, ${data.num_children || 0} child → collection ${data.collection_name || targetCollection}`
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

  async function fetchDocs(collectionName, overrides = {}) {
    if (!collectionName) {
      setDocs([]);
      return;
    }
    const effectiveQuery = {
      ...docsQuery,
      ...(overrides || {}),
    };
    if (overrides && Object.keys(overrides).length > 0) {
      setDocsQuery(effectiveQuery);
    }

    const params = new URLSearchParams({
      collection_name: collectionName,
      sort_by: effectiveQuery.sort_by || "document_date",
      sort_order: effectiveQuery.sort_order || "desc",
    });
    if (effectiveQuery.search) {
      params.set("search", String(effectiveQuery.search));
    }
    if (effectiveQuery.year_from !== "" && effectiveQuery.year_from != null) {
      params.set("year_from", String(effectiveQuery.year_from));
    }
    if (effectiveQuery.year_to !== "" && effectiveQuery.year_to != null) {
      params.set("year_to", String(effectiveQuery.year_to));
    }

    setDocsLoading(true);
    setDocsError("");
    try {
      const resp = await authFetch(`${API_BASE}/admin/docs?${params.toString()}`);
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

  async function handleDeleteDocs(collectionName, sources = []) {
    const uniqueSources = Array.from(new Set((sources || []).filter(Boolean)));
    if (!collectionName || uniqueSources.length === 0) {
      return;
    }
    if (
      !window.confirm(
        `Xóa ${uniqueSources.length} tài liệu đã chọn khỏi collection '${collectionName}'?`
      )
    ) {
      return;
    }

    setDocsLoading(true);
    setDocsError("");
    try {
      const failures = [];
      for (const source of uniqueSources) {
        const resp = await authFetch(`${API_BASE}/admin/docs`, {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ collection_name: collectionName, source }),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          failures.push(err.detail || source);
        }
      }
      if (failures.length > 0) {
        throw new Error(`Không thể xóa một số tài liệu: ${failures.slice(0, 3).join(", ")}`);
      }
      await fetchDocs(collectionName);
    } catch (err) {
      console.error(err);
      setDocsError(String(err));
    } finally {
      setDocsLoading(false);
    }
  }

  async function handleDeleteCollection(name) {
    if (
      !window.confirm(
        `Làm trống toàn bộ dữ liệu trong collection '${name}' (tất cả chunk + metadata)? Collection sẽ được giữ lại.`
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
      await fetchCollections();
      if (selectedCollection === name) {
        await fetchDocs(name);
      }
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
          <div className="brand-badge">
            <img src={import.meta.env.BASE_URL + "logo.jpg"} alt="PharmaAI" />
          </div>
          <div>
            <div className="brand-text-main">PharmaAI</div>
            <div className="brand-text-sub">Trợ lý dược thông minh</div>
          </div>
        </div>
        <div className="top-nav-right">
          {jwtToken && (
            <div className="nav-user-wrap">
              <button
                className="nav-user-btn"
                type="button"
                onClick={() => {
                  setRoute("account");
                  window.history.replaceState(null, "", "/account/");
                }}
                title="Quản lý tài khoản"
              >
                <span className="nav-user-avatar">
                  {(username || "U").charAt(0).toUpperCase()}
                </span>
                <span className="nav-user-name">{username}</span>
              </button>
              <button
                className="nav-logout-btn"
                type="button"
                onClick={handleLogout}
                title="Đăng xuất"
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
                  <polyline points="16 17 21 12 16 7"/>
                  <line x1="21" y1="12" x2="9" y2="12"/>
                </svg>
              </button>
            </div>
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
              docsQuery,
              uploadFiles,
              collectionName,
              newCollectionMode,
              newCollectionName,
              ingestStatus,
              ingestProgress,
              isIngesting,
              currentJobId,
              feedbackData,
              feedbackLoading,
              feedbackTab,
            }}
            handlers={{
              setUploadFiles,
              setCollectionName,
              setNewCollectionMode,
              setNewCollectionName,
              setSelectedCollection,
              setDocsQuery,
              setFeedbackTab,
              fetchCollections,
              fetchDocs,
              handleIngest,
              handleDeleteCollection,
              handleDeleteDoc,
              handleDeleteDocs,
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
              docsQuery,
              uploadFiles,
              collectionName,
              newCollectionMode,
              newCollectionName,
              ingestStatus,
              ingestProgress,
              isIngesting,
              currentJobId,
              feedbackData,
              feedbackLoading,
              feedbackTab,
            }}
            handlers={{
              setUploadFiles,
              setCollectionName,
              setNewCollectionMode,
              setNewCollectionName,
              setSelectedCollection,
              setDocsQuery,
              setFeedbackTab,
              fetchCollections,
              fetchDocs,
              handleIngest,
              handleDeleteCollection,
              handleDeleteDoc,
              handleDeleteDocs,
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
            onCancel={handleCancel}
            onFeedback={handleFeedback}
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

const DISLIKE_REASONS = [
  "Thông tin không chính xác",
  "Không liên quan",
  "Thiếu thông tin",
  "Câu trả lời khó hiểu",
  "Khác",
];

export function MessageBubble({ message, onFeedback }) {
  const [showSources, setShowSources] = useState(false);
  const [expandedSource, setExpandedSource] = useState(null);
  const [expandedDrugs, setExpandedDrugs] = useState(new Set());
  const [viewerSource, setViewerSource] = useState(null);
  const [feedbackComment, setFeedbackComment] = useState("");
  const [feedbackReason, setFeedbackReason] = useState("");
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [feedbackSending, setFeedbackSending] = useState(false);
  const [copied, setCopied] = useState(false);
  const [localFeedback, setLocalFeedback] = useState(message.feedback || null);
  const isUser = message.role === "user";

  // Sync if parent prop changes (e.g. after session reload)
  const prevFeedbackRef = React.useRef(message.feedback);
  if (message.feedback !== prevFeedbackRef.current) {
    prevFeedbackRef.current = message.feedback;
    if (message.feedback && message.feedback !== localFeedback) {
      setLocalFeedback(message.feedback);
    }
  }

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
    if (localFeedback === rating) return;
    if (rating === "down") {
      setFeedbackReason("");
      setFeedbackComment("");
      setShowFeedbackForm(true);
      return;
    }
    setShowFeedbackForm(false);
    setLocalFeedback(rating);
    onFeedback && onFeedback(rating, "");
  };

  const handleCopy = () => {
    const text = message.content || "";
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => {});
  };

  const handleSubmitFeedback = () => {
    const combined = [feedbackReason, feedbackComment].filter(Boolean).join(" — ");
    setLocalFeedback("down");
    onFeedback && onFeedback("down", combined);
    setShowFeedbackForm(false);
  };

  const handleSkipFeedback = () => {
    setLocalFeedback("down");
    onFeedback && onFeedback("down", "");
    setShowFeedbackForm(false);
  };

  const formatAssistantText = (rawText) => {
    if (!rawText) return "";

    const esc = (s) =>
      s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    // Render [Citation name] as a styled inline chip (not a link, since no URL)
    const renderCitations = (s) =>
      s.replace(/\[([^\]]+)\]/g, '<cite class="chat-cite">$1</cite>');

    const inline = (s) =>
      renderCitations(
        s
          .replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>")
          .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
          .replace(/\*([^*\n]+?)\*/g, "<em>$1</em>")
          .replace(/`([^`]+)`/g, '<code class="chat-icode">$1</code>')
      );

    // No longer strip [Source N] — citations now use doc names
    const text = rawText.trimEnd();
    const lines = text.split("\n");
    const out = [];
    let i = 0;

    while (i < lines.length) {
      const raw = lines[i];
      const trimmed = raw.trim();

      // Fenced code block
      if (trimmed.startsWith("```")) {
        const lang = trimmed.slice(3).trim();
        i++;
        const codeLines = [];
        while (i < lines.length && !lines[i].trim().startsWith("```")) {
          codeLines.push(esc(lines[i]));
          i++;
        }
        if (i < lines.length) i++;
        const langAttr = lang ? ` data-lang="${esc(lang)}"` : "";
        out.push(`<pre class="chat-code"${langAttr}><code>${codeLines.join("\n")}</code></pre>`);
        continue;
      }

      // Horizontal rule
      if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
        out.push("<hr/>");
        i++;
        continue;
      }

      // Headings
      const hm = raw.match(/^(#{1,3}) (.+)/);
      if (hm) {
        const lvl = hm[1].length;
        out.push(`<h${lvl} class="chat-h${lvl}">${inline(esc(hm[2].trim()))}</h${lvl}>`);
        i++;
        continue;
      }

      // Blockquote
      if (/^> ?/.test(raw)) {
        const qLines = [];
        while (i < lines.length && /^> ?/.test(lines[i])) {
          qLines.push(inline(esc(lines[i].replace(/^> ?/, ""))));
          i++;
        }
        out.push(`<blockquote class="chat-bq">${qLines.join("<br/>")}</blockquote>`);
        continue;
      }

      // Unordered list
      if (/^[ \t]*[-*+] /.test(raw)) {
        const items = [];
        while (i < lines.length && /^[ \t]*[-*+] /.test(lines[i])) {
          items.push(`<li>${inline(esc(lines[i].replace(/^[ \t]*[-*+] /, "")))}</li>`);
          i++;
        }
        out.push(`<ul class="chat-ul">${items.join("")}</ul>`);
        continue;
      }

      // Ordered list
      if (/^[ \t]*\d+[.)]\s/.test(raw)) {
        const items = [];
        while (i < lines.length && /^[ \t]*\d+[.)]\s/.test(lines[i])) {
          items.push(`<li>${inline(esc(lines[i].replace(/^[ \t]*\d+[.)]\s+/, "")))}</li>`);
          i++;
        }
        out.push(`<ol class="chat-ol">${items.join("")}</ol>`);
        continue;
      }

      // Table
      if (/^\|/.test(raw)) {
        const tLines = [];
        while (i < lines.length && /^\|/.test(lines[i])) {
          tLines.push(lines[i]);
          i++;
        }
        if (tLines.length >= 2) {
          const parseRow = (row) => row.split("|").slice(1, -1).map((c) => c.trim());
          const hCells = parseRow(tLines[0]).map((c) => `<th>${inline(esc(c))}</th>`).join("");
          const bodyRows = tLines
            .slice(2)
            .map((r) => `<tr>${parseRow(r).map((c) => `<td>${inline(esc(c))}</td>`).join("")}</tr>`)
            .join("");
          out.push(`<table class="chat-table"><thead><tr>${hCells}</tr></thead><tbody>${bodyRows}</tbody></table>`);
        } else {
          tLines.forEach((l) => out.push(`<p>${inline(esc(l))}</p>`));
        }
        continue;
      }

      // Empty line
      if (trimmed === "") { i++; continue; }

      // Paragraph: group consecutive plain lines
      const pLines = [];
      while (
        i < lines.length &&
        lines[i].trim() !== "" &&
        !/^[ \t]*```/.test(lines[i]) &&
        !/^(#{1,3}) /.test(lines[i]) &&
        !/^> ?/.test(lines[i]) &&
        !/^[ \t]*[-*+] /.test(lines[i]) &&
        !/^[ \t]*\d+[.)]\s/.test(lines[i]) &&
        !/^\|/.test(lines[i]) &&
        !/^(-{3,}|\*{3,}|_{3,})$/.test(lines[i].trim())
      ) {
        pLines.push(inline(esc(lines[i])));
        i++;
      }
      if (pLines.length) out.push(`<p>${pLines.join("<br/>")}</p>`);
    }

    return out.join("");
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
          className="chat-prose"
          dangerouslySetInnerHTML={{ __html: formatAssistantText(message.content) }}
        />
      )}

      {!isUser && hasSources && (
        <>
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
                    {s.legal_type && (
                      <span className="source-card-legal-type">{s.legal_type}</span>
                    )}
                    {s.issuance_date && (
                      <span className="source-card-date">{s.issuance_date}</span>
                    )}
                    <button
                      className="source-card-view-btn"
                      onClick={(e) => handleOpenDoc(s, e)}
                      title="Xem tài liệu"
                    >
                      Xem
                    </button>
                  </div>
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
            className={"feedback-action-btn copy-btn" + (copied ? " copied" : "")}
            onClick={handleCopy}
            title="Sao chép câu trả lời"
          >
            {copied ? (
              <>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12"/>
                </svg>
                Đã sao chép
              </>
            ) : (
              <>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                </svg>
                Sao chép
              </>
            )}
          </button>
          <div className="feedback-divider" />
          {localFeedback ? (
            <div className={"feedback-rated " + (localFeedback === "up" ? "feedback-rated--up" : "feedback-rated--down")}>
              {localFeedback === "up" ? (
                <>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                    <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/>
                  </svg>
                  Hữu ích
                </>
              ) : (
                <>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                    <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10zM17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/>
                  </svg>
                  Đã ghi nhận
                </>
              )}
            </div>
          ) : (
            <>
              <button
                type="button"
                className="feedback-action-btn"
                onClick={() => handleRate("up")}
                title="Hữu ích"
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/>
                </svg>
                Hữu ích
              </button>
              <button
                type="button"
                className="feedback-action-btn down-btn"
                onClick={() => handleRate("down")}
                title="Chưa tốt"
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10zM17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/>
                </svg>
                Chưa tốt
              </button>
            </>
          )}
        </div>
      )}
      {showFeedbackForm && (
        <div className="feedback-form">
          <div className="feedback-form-header">
            <span className="feedback-form-title">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10zM17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/>
              </svg>
              Điều gì chưa tốt?
            </span>
            <button type="button" className="feedback-form-close" onClick={handleSkipFeedback} title="Đóng">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </button>
          </div>
          <div className="feedback-reasons">
            {DISLIKE_REASONS.map((r) => (
              <button
                key={r}
                type="button"
                className={"feedback-reason-chip" + (feedbackReason === r ? " selected" : "")}
                onClick={() => setFeedbackReason(feedbackReason === r ? "" : r)}
              >
                {r}
              </button>
            ))}
          </div>
          <textarea
            className="feedback-textarea"
            rows="2"
            placeholder="Mô tả thêm (không bắt buộc)..."
            value={feedbackComment}
            onChange={(e) => setFeedbackComment(e.target.value)}
          />
          <div className="feedback-form-actions">
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={handleSubmitFeedback}
            >
              Gửi phản hồi
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
