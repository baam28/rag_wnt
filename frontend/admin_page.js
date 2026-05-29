import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';

const ANALYTICS_METRICS = [
  { key: 'users', label: 'Tổng số User', dataKey: 'users_per_day', title: 'Số user theo ngày', color: '#2563eb' },
  { key: 'sessions', label: 'Số cuộc hội thoại', dataKey: 'sessions_per_day', title: 'Số cuộc hội thoại theo ngày', color: '#16a34a' },
  { key: 'messages', label: 'Tổng tin nhắn', dataKey: 'messages_per_day', title: 'Tin nhắn theo ngày', color: '#4f46e5' },
  { key: 'feedback', label: 'Đánh giá thu thập được', dataKey: 'feedback_per_day', title: 'Đánh giá theo ngày', color: '#d97706' },
  { key: 'api_usage', label: 'API usage (tokens)', dataKey: 'api_usage', title: 'API usage (tokens) theo ngày', color: '#0d9488' },
];

const COLLECTION_LABELS = {
  drug: "Dược phẩm",
  legal: "Văn bản pháp lý",
};

const DOCS_DEFAULT_QUERY = {
  search: "",
  sort_by: "document_date",
  sort_order: "desc",
  year_from: "",
  year_to: "",
};

const DOC_SORT_OPTIONS = [
  { value: "document_date:desc", label: "Năm ban hành (mới → cũ)" },
  { value: "document_date:asc", label: "Năm ban hành (cũ → mới)" },
  { value: "source:asc", label: "Tên A → Z" },
  { value: "source:desc", label: "Tên Z → A" },
];

function getCollectionLabel(name) {
  return COLLECTION_LABELS[name] || name;
}

function AdminPage({

  state,
  handlers,
}) {
  const {
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
  } = state;

  const {
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
  } = handlers;

  const [activeTab, setActiveTab] = React.useState("analytics");
  const [analyticsData, setAnalyticsData] = React.useState(null);
  const [analyticsLoading, setAnalyticsLoading] = React.useState(false);
  const [selectedMetric, setSelectedMetric] = React.useState('messages');
  const [users, setUsers] = React.useState([]);
  const [usersLoading, setUsersLoading] = React.useState(false);

  const [docFilters, setDocFilters] = React.useState(docsQuery || { ...DOCS_DEFAULT_QUERY });
  const [selectedDocs, setSelectedDocs] = React.useState([]);

  const [pwdForm, setPwdForm] = React.useState({ userId: null, value: "" });

  // API settings state
  const [apiSettings, setApiSettings] = React.useState(null);
  const [apiSettingsLoading, setApiSettingsLoading] = React.useState(false);
  const [apiSettingsSaving, setApiSettingsSaving] = React.useState(false);
  const [apiSettingsMsg, setApiSettingsMsg] = React.useState(null); // { type: "success"|"error", text }
  const [apiForm, setApiForm] = React.useState({
    provider: "openai",
    model: "gpt-4o-mini",
  });

  const OPENAI_MODELS = [
    { value: "gpt-4o", label: "GPT-4o" },
    { value: "gpt-4o-mini", label: "GPT-4o Mini" },
    { value: "gpt-4-turbo", label: "GPT-4 Turbo" },
    { value: "gpt-3.5-turbo", label: "GPT-3.5 Turbo" },
  ];
  const ANTHROPIC_MODELS = [
    { value: "claude-opus-4-6", label: "Claude Opus 4.6" },
    { value: "claude-sonnet-4-6", label: "Claude Sonnet 4.6" },
    { value: "claude-haiku-4-5-20251001", label: "Claude Haiku 4.5" },
  ];

  async function fetchApiSettings() {
    setApiSettingsLoading(true);
    setApiSettingsMsg(null);
    try {
      const resp = await authFetch("/admin/api-settings");
      if (resp.ok) {
        const data = await resp.json();
        setApiSettings(data);
        setApiForm({ provider: data.provider || "openai", model: data.model || "gpt-4o-mini" });
      } else {
        setApiSettingsMsg({ type: "error", text: "Không tải được cấu hình API." });
      }
    } catch {
      setApiSettingsMsg({ type: "error", text: "Lỗi kết nối." });
    }
    setApiSettingsLoading(false);
  }

  async function handleSaveApiSettings(e) {
    e.preventDefault();
    setApiSettingsSaving(true);
    setApiSettingsMsg(null);
    try {
      const resp = await authFetch("/admin/api-settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: apiForm.provider, model: apiForm.model }),
      });
      if (resp.ok) {
        const data = await resp.json();
        setApiSettings(data);
        setApiSettingsMsg({ type: "success", text: "Đã lưu cấu hình API." });
      } else {
        const err = await resp.json();
        setApiSettingsMsg({ type: "error", text: "Lỗi: " + (err.detail || "Không thể lưu.") });
      }
    } catch {
      setApiSettingsMsg({ type: "error", text: "Lỗi kết nối." });
    }
    setApiSettingsSaving(false);
  }

  React.useEffect(() => {
    setDocFilters(docsQuery || { ...DOCS_DEFAULT_QUERY });
  }, [docsQuery]);

  function applyDocFilters(nextFilters = docFilters) {
    const normalized = {
      search: String(nextFilters.search || "").trim(),
      sort_by: nextFilters.sort_by || "document_date",
      sort_order: nextFilters.sort_order || "desc",
      year_from: nextFilters.year_from === "" ? "" : String(nextFilters.year_from).trim(),
      year_to: nextFilters.year_to === "" ? "" : String(nextFilters.year_to).trim(),
    };
    setDocsQuery(normalized);
    if (selectedCollection) {
      fetchDocs(selectedCollection, normalized);
    }
  }

  const hasActiveDocFilters = Boolean(
    (docFilters.search || "").trim() ||
    (docFilters.year_from || "").toString().trim() ||
    (docFilters.year_to || "").toString().trim() ||
    `${docFilters.sort_by || "document_date"}:${docFilters.sort_order || "desc"}` !== "document_date:desc"
  );

  function resetDocFilters() {
    const resetValue = { ...DOCS_DEFAULT_QUERY };
    setDocFilters(resetValue);
    setDocsQuery(resetValue);
    if (selectedCollection) {
      fetchDocs(selectedCollection, resetValue);
    }
  }

  React.useEffect(() => {
    setSelectedDocs((prev) => prev.filter((source) => docs.some((d) => d.source === source)));
  }, [docs]);

  React.useEffect(() => {
    setSelectedDocs([]);
  }, [selectedCollection]);

  React.useEffect(() => {
    if (activeTab === "analytics" && !analyticsData) {
      setAnalyticsLoading(true);
      authFetch("/admin/analytics")
        .then(res => res.json())
        .then(data => { setAnalyticsData(data); setAnalyticsLoading(false); })
        .catch(() => setAnalyticsLoading(false));
    } else if (activeTab === "users" && users.length === 0) {
      fetchUsers();
    } else if (activeTab === "docs") {
      if (collections.length === 0) fetchCollections();
    } else if (activeTab === "feedback" && !feedbackData) {
      fetchFeedback();
    } else if (activeTab === "api" && !apiSettings) {
      fetchApiSettings();
    }
  }, [activeTab]);

  async function fetchUsers() {
    setUsersLoading(true);
    try {
      const resp = await authFetch("/admin/users");
      if (resp.ok) setUsers(await resp.json());
    } catch {}
    setUsersLoading(false);
  }

  async function handleToggleAdmin(userId, currentAdmin) {
    try {
      const resp = await authFetch(`/admin/users/${userId}/role`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_admin: !currentAdmin })
      });
      if (resp.ok) fetchUsers();
      else alert("Không thể thay đổi quyền (có thể là chính bạn).");
    } catch {}
  }

  async function handleDeleteUser(userId) {
    if (!window.confirm("Xóa tài khoản này? (bao gồm toàn bộ lịch sử hội thoại)")) return;
    try {
      const resp = await authFetch(`/admin/users/${userId}`, { method: "DELETE" });
      if (resp.ok) fetchUsers();
      else alert("Lỗi khi xóa tài khoản.");
    } catch {}
  }

  async function handleSetPassword(userId) {
    if (!pwdForm.value || pwdForm.value.length < 6) {
      alert("Mật khẩu phải từ 6 ký tự.");
      return;
    }
    try {
      const resp = await authFetch(`/admin/users/${userId}/password`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_password: pwdForm.value })
      });
      if (resp.ok) {
        alert("Đã đổi mật khẩu thành công.");
        setPwdForm({ userId: null, value: "" });
      } else {
        const err = await resp.json();
        alert("Lỗi: " + (err.detail || "Không thể đổi mật khẩu"));
      }
    } catch (e) {
      alert("Lỗi kết nối.");
    }
  }

  return (
    <>

      <section className="main-panel main-panel--admin" style={{ display: "flex", flexDirection: "column" }}>
        <div className="admin-tabs" style={{ flexShrink: 0 }}>
          <button className={"admin-tab" + (activeTab === "analytics" ? " active" : "")} onClick={() => setActiveTab("analytics")}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            Thống kê
          </button>
          <button className={"admin-tab" + (activeTab === "users" ? " active" : "")} onClick={() => setActiveTab("users")}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
            Quản lý User
          </button>
          <button className={"admin-tab" + (activeTab === "docs" ? " active" : "")} onClick={() => setActiveTab("docs")}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
            Tài liệu & DB
          </button>
          <button className={"admin-tab" + (activeTab === "feedback" ? " active" : "")} onClick={() => setActiveTab("feedback")}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
            Đánh giá
          </button>
          <button className={"admin-tab" + (activeTab === "api" ? " active" : "")} onClick={() => setActiveTab("api")}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/><path d="M7 8h.01M11 8h6"/></svg>
            Cấu hình API
          </button>
        </div>

        <div style={{ flex: 1, overflowY: "auto", paddingRight: "0.5rem", paddingBottom: "2rem" }}>
        
        {activeTab === "analytics" && (
          <div className="admin-analytics">
            <div className="admin-card">
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.5rem", flexWrap: "wrap", gap: "0.5rem" }}>
                <h2 className="admin-card-title" style={{ margin: 0 }}>Thống kê tổng quan</h2>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  onClick={() => {
                    setAnalyticsData(null);
                    setAnalyticsLoading(true);
                    authFetch("/admin/analytics").then((res) => res.json()).then((data) => { setAnalyticsData(data); setAnalyticsLoading(false); }).catch(() => setAnalyticsLoading(false));
                  }}
                  disabled={analyticsLoading}
                >
                  {analyticsLoading ? "Đang tải..." : "Làm mới"}
                </button>
              </div>
              {analyticsLoading && !analyticsData ? <p className="admin-status">Đang tải...</p> : analyticsData ? (
                <>
                  <div className="feedback-stats admin-stat-cards">
                    {ANALYTICS_METRICS.map((m) => {
                      const total =
                        m.key === 'users' ? analyticsData.total_users
                        : m.key === 'sessions' ? analyticsData.total_sessions
                        : m.key === 'messages' ? analyticsData.total_messages
                        : m.key === 'feedback' ? (analyticsData.feedback?.total ?? 0)
                        : (analyticsData.llm_usage?.total?.prompt_tokens ?? 0) + (analyticsData.llm_usage?.total?.completion_tokens ?? 0);
                      const isSelected = selectedMetric === m.key;
                      return (
                        <button
                          key={m.key}
                          type="button"
                          className={`feedback-stat-card admin-stat-card ${isSelected ? 'admin-stat-card--selected' : ''}`}
                          style={{ cursor: 'pointer', textAlign: 'left', border: '2px solid transparent' }}
                          onClick={() => setSelectedMetric(m.key)}
                          aria-pressed={isSelected}
                          aria-label={`${m.label}: ${total}. Nhấn để xem biểu đồ`}
                        >
                          <div className="feedback-stat-number">{typeof total === 'number' ? total.toLocaleString() : total}</div>
                          <div className="feedback-stat-label">{m.label}</div>
                        </button>
                      );
                    })}
                  </div>

                  <div className="admin-chart-area" style={{ marginTop: '1.5rem' }} aria-label={`Biểu đồ: ${ANALYTICS_METRICS.find((m) => m.key === selectedMetric)?.title ?? selectedMetric}`}>
                    {(() => {
                      const metric = ANALYTICS_METRICS.find((m) => m.key === selectedMetric);
                      let chartData = [];
                      let dataKey = 'count';
                      if (metric) {
                        if (metric.key === 'api_usage' && analyticsData.llm_usage?.daily) {
                          chartData = analyticsData.llm_usage.daily.map((d) => ({ ...d, value: d.count ?? (d.prompt_tokens || 0) + (d.completion_tokens || 0) }));
                          dataKey = 'value';
                        } else {
                          chartData = analyticsData[metric.dataKey] ?? [];
                        }
                      }
                      const hasData = chartData.length > 0 && chartData.some((d) => (d[dataKey] ?? 0) > 0);
                      if (!hasData) {
                        return (
                          <p className="admin-status" style={{ padding: '2rem 0' }}>
                            Chưa có dữ liệu trong khoảng thời gian này.
                          </p>
                        );
                      }
                      return (
                        <>
                          <h3 className="admin-chart-title">{metric?.title}</h3>
                          <p className="admin-chart-subtitle">30 ngày gần nhất</p>
                          <div className="admin-recharts-wrap">
                            <ResponsiveContainer width="100%" height={280}>
                              <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                <XAxis
                                  dataKey="date"
                                  tick={{ fontSize: 11, fill: '#6b7280' }}
                                  tickFormatter={(v) => (v && v.length >= 10 ? v.slice(5) : v)}
                                />
                                <YAxis tick={{ fontSize: 11, fill: '#6b7280' }} allowDecimals={false} />
                                <Tooltip
                                  labelFormatter={(v) => v}
                                  formatter={(val) => [typeof val === 'number' ? val.toLocaleString() : val, metric?.label]}
                                  contentStyle={{ fontSize: '12px', borderRadius: '8px' }}
                                />
                                <Line
                                  type="monotone"
                                  dataKey={dataKey}
                                  stroke={metric?.color ?? '#4f46e5'}
                                  strokeWidth={2}
                                  dot={{ r: 3, fill: metric?.color }}
                                  activeDot={{ r: 5 }}
                                />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </>
                      );
                    })()}
                  </div>
                </>
              ) : <p className="admin-status">Không có dữ liệu.</p>}
            </div>
          </div>
        )}

        {activeTab === "users" && (
          <div className="admin-card">
            <div className="admin-row admin-row--head" style={{ marginBottom: "1.5rem" }}>
              <h2 className="admin-card-title" style={{ flex: 1 }}>Quản lý User</h2>
              <button className="btn btn-ghost btn-sm" type="button" onClick={fetchUsers}>Làm mới</button>
            </div>
            {usersLoading ? <p className="admin-status">Đang tải...</p> : (
              <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", fontSize: "0.9rem" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid #e5e7eb", color: "#64748b" }}>
                    <th style={{ padding: "0.5rem 0.5rem 0.5rem 0", fontWeight: 600 }}>Tên đăng nhập</th>
                    <th style={{ padding: "0.5rem", fontWeight: 600 }}>Vai trò</th>
                    <th style={{ padding: "0.5rem 0 0.5rem 0.5rem", textAlign: "right", fontWeight: 600 }}>Thao tác</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map(u => (
                    <tr key={u.id} style={{ borderBottom: "1px solid #f3f4f6" }}>
                      <td style={{ padding: "0.75rem 0.5rem 0.75rem 0", fontWeight: 500, color: "#0f172a" }}>{u.username}</td>
                      <td style={{ padding: "0.75rem 0.5rem" }}>
                        <span style={{ 
                          padding: "0.25rem 0.6rem", 
                          borderRadius: "999px", 
                          fontSize: "0.75rem", 
                          fontWeight: 500,
                          background: u.is_admin ? "#eff6ff" : "#f1f5f9", 
                          color: u.is_admin ? "#2563eb" : "#475569" 
                        }}>
                          {u.is_admin ? "Admin" : "User"}
                        </span>
                      </td>
                      <td style={{ padding: "0.75rem 0 0.75rem 0.5rem", textAlign: "right", whiteSpace: "nowrap" }}>
                        {pwdForm.userId === u.id ? (
                          <div style={{ display: "inline-flex", gap: "0.5rem", marginRight: "0.5rem" }}>
                            <input 
                              type="password" 
                              className="admin-input" 
                              style={{ width: "120px", padding: "0.25rem 0.5rem", fontSize: "0.8rem", height: "1.75rem" }}
                              placeholder="MK mới"
                              value={pwdForm.value}
                              onChange={(e) => setPwdForm({ ...pwdForm, value: e.target.value })}
                            />
                            <button className="btn btn-primary btn-sm" onClick={() => handleSetPassword(u.id)}>Lưu</button>
                            <button className="btn btn-ghost btn-sm" onClick={() => setPwdForm({ userId: null, value: "" })}>Hủy</button>
                          </div>
                        ) : (
                          <button className="btn btn-ghost btn-sm" style={{ marginRight: "0.5rem" }} onClick={() => setPwdForm({ userId: u.id, value: "" })}>
                            Đổi MK
                          </button>
                        )}
                        {u.username !== "admin" && (
                          <>
                            <button className="btn btn-ghost btn-sm" style={{ marginRight: "0.5rem" }} onClick={() => handleToggleAdmin(u.id, u.is_admin)}>
                              {u.is_admin ? "Gỡ Admin" : "Cấp Admin"}
                            </button>
                            <button className="btn btn-danger btn-sm" onClick={() => handleDeleteUser(u.id)}>Xóa</button>
                          </>
                        )}
                      </td>
                    </tr>
                  ))}
                  {users.length === 0 && (
                    <tr>
                      <td colSpan="3" style={{ padding: "1rem 0", textAlign: "center", color: "#64748b" }}>Chưa có user nào.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            )}
          </div>
        )}

        {activeTab === "docs" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
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
                  multiple
                  className="admin-input"
                  onChange={(e) => setUploadFiles(Array.from(e.target.files || []))}
                />
                {Array.isArray(uploadFiles) && uploadFiles.length > 0 && (
                  <p className="admin-card-caption" style={{ marginTop: "0.5rem" }}>
                    Đã chọn {uploadFiles.length} tệp: {uploadFiles.map((f) => f.name).join(", ")}
                  </p>
                )}
              </div>
              <div className="admin-field">
                <label className="admin-label">Collection</label>
                <select
                  className="admin-input"
                  value={collectionName || "drug"}
                  onChange={(e) => setCollectionName(e.target.value)}
                  aria-label="Chọn collection"
                >
                  {Object.entries(COLLECTION_LABELS).map(([value, label]) => (
                    <option key={value} value={value}>{label}</option>
                  ))}
                </select>
              </div>

              <div
                className="admin-actions"
                style={{ flexWrap: "wrap", gap: "0.5rem" }}
              >
                <button
                  type="submit"
                  className="btn btn-primary btn-full"
                  disabled={isIngesting}
                  style={{ flex: "1 1 auto" }}
                >
                  {isIngesting ? "Đang ingest..." : "Ingest tài liệu"}
                </button>
                {isIngesting && currentJobId && (
                  <button
                    type="button"
                    className="btn btn-danger"
                    onClick={async () => {
                      try {
                        const r = await fetch(
                          `/ingest-jobs/${currentJobId}/cancel`,
                          { method: "POST" }
                        );
                        if (r.ok) {
                          // status will be picked up by polling
                        }
                      } catch (e) {
                        console.error(e);
                      }
                    }}
                  >
                    Hủy ingest
                  </button>
                )}
              </div>
              {ingestProgress && (
                <div
                  className="admin-status"
                  style={{ marginTop: "0.5rem" }}
                >
                  <p style={{ marginBottom: "0.25rem" }}>
                    {ingestProgress.message}
                  </p>
                  {ingestProgress.total > 0 && (
                    <progress
                      max={ingestProgress.total}
                      value={ingestProgress.current}
                      style={{ width: "100%", height: "6px" }}
                    />
                  )}
                </div>
              )}
              {ingestStatus && (
                <p className="admin-status">{ingestStatus}</p>
              )}
            </form>
          </div>

          <div className="admin-card admin-card--db">
            <div
              className="admin-row admin-row--head"
              style={{ marginBottom: "1rem" }}
            >
              <h2 className="admin-card-title" style={{ flex: 1 }}>
                Quản lý database
              </h2>
            </div>
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
                      } else {
                        setSelectedCollection(c.name);
                        fetchDocs(c.name);
                      }
                    }}
                  >
                    <div className="admin-collection-name">
                      {getCollectionLabel(c.name)}
                    </div>
                  </div>
                  {selectedCollection === c.name && (
                    <div className="admin-collection-body">
                      <form
                        className="admin-doc-toolbar"
                        onSubmit={(e) => {
                          e.preventDefault();
                          applyDocFilters();
                        }}
                      >
                        <div className="admin-doc-toolbar-head">
                          <div>
                            <p className="admin-doc-toolbar-title">Tìm kiếm & sắp xếp</p>
                            <p className="admin-doc-toolbar-caption">
                              {docsLoading
                                ? "Đang cập nhật danh sách..."
                                : `${docs.length} tài liệu${hasActiveDocFilters ? " · đang áp dụng bộ lọc" : ""}`}
                            </p>
                          </div>
                          <div className="admin-doc-toolbar-actions">
                            <button className="btn btn-primary btn-sm" type="submit">Áp dụng</button>
                            <button className="btn btn-ghost btn-sm" type="button" onClick={resetDocFilters}>Đặt lại</button>
                          </div>
                        </div>

                        <div className="admin-doc-filters-row">
                          <div className="admin-doc-field">
                            <label className="admin-label" htmlFor={`doc-search-${c.name}`}>Tìm tài liệu</label>
                            <input
                              id={`doc-search-${c.name}`}
                              type="text"
                              className="admin-input admin-doc-search"
                              placeholder="Nhập tên văn bản..."
                              value={docFilters.search || ""}
                              onChange={(e) => setDocFilters((prev) => ({ ...prev, search: e.target.value }))}
                            />
                          </div>
                          <div className="admin-doc-field">
                            <label className="admin-label" htmlFor={`doc-sort-${c.name}`}>Sắp xếp</label>
                            <select
                              id={`doc-sort-${c.name}`}
                              className="admin-input"
                              value={`${docFilters.sort_by || "document_date"}:${docFilters.sort_order || "desc"}`}
                              onChange={(e) => {
                                const [sortBy, sortOrder] = String(e.target.value).split(":");
                                setDocFilters((prev) => ({
                                  ...prev,
                                  sort_by: sortBy || "document_date",
                                  sort_order: sortOrder || "desc",
                                }));
                              }}
                            >
                              {DOC_SORT_OPTIONS.map((opt) => (
                                <option key={opt.value} value={opt.value}>{opt.label}</option>
                              ))}
                            </select>
                          </div>
                        </div>

                        <div className="admin-doc-filters-row admin-doc-filters-row--years">
                          <div className="admin-doc-field">
                            <label className="admin-label" htmlFor={`doc-year-from-${c.name}`}>Từ năm</label>
                            <input
                              id={`doc-year-from-${c.name}`}
                              type="number"
                              min="1900"
                              max="2099"
                              step="1"
                              className="admin-input"
                              placeholder="Ví dụ: 2016"
                              value={docFilters.year_from || ""}
                              onChange={(e) => setDocFilters((prev) => ({ ...prev, year_from: e.target.value }))}
                            />
                          </div>
                          <div className="admin-doc-field">
                            <label className="admin-label" htmlFor={`doc-year-to-${c.name}`}>Đến năm</label>
                            <input
                              id={`doc-year-to-${c.name}`}
                              type="number"
                              min="1900"
                              max="2099"
                              step="1"
                              className="admin-input"
                              placeholder="Ví dụ: 2026"
                              value={docFilters.year_to || ""}
                              onChange={(e) => setDocFilters((prev) => ({ ...prev, year_to: e.target.value }))}
                            />
                          </div>
                        </div>
                      </form>
                      <div className="admin-doc-bulk-bar">
                        <label className="admin-doc-check-all">
                          <input
                            type="checkbox"
                            checked={docs.length > 0 && selectedDocs.length === docs.length}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedDocs(docs.map((d) => d.source));
                              } else {
                                setSelectedDocs([]);
                              }
                            }}
                          />
                          <span>Chọn tất cả</span>
                        </label>
                        <div className="admin-doc-bulk-actions">
                          <span className="admin-doc-selected-count">
                            {selectedDocs.length > 0 ? `Đã chọn ${selectedDocs.length}` : "Chưa chọn tài liệu"}
                          </span>
                          <button
                            className="btn btn-danger btn-sm"
                            type="button"
                            disabled={selectedDocs.length === 0 || docsLoading}
                            onClick={async () => {
                              await handleDeleteDocs(selectedCollection, selectedDocs);
                              setSelectedDocs([]);
                            }}
                          >
                            Xóa đã chọn
                          </button>
                          <button
                            className="btn btn-danger btn-sm"
                            onClick={() => handleDeleteCollection(c.name)}
                            title={`Xóa toàn bộ dữ liệu trong collection '${c.name}'`}
                          >
                            Xoá tất cả văn bản
                          </button>
                        </div>
                      </div>
                      {docsLoading && (
                        <p className="admin-status">Đang tải...</p>
                      )}
                      {docsError && (
                        <p className="admin-error">{docsError}</p>
                      )}
                      {!docsLoading && docs.length === 0 && (
                        <p className="admin-status">Không có tài liệu phù hợp bộ lọc hiện tại.</p>
                      )}
                      {docs.length > 0 && (
                        <ul className="admin-doc-list">
                          {docs.map((d) => (
                            <li
                              key={d.source}
                              className="admin-doc-item"
                            >
                              <label className="admin-doc-check">
                                <input
                                  type="checkbox"
                                  checked={selectedDocs.includes(d.source)}
                                  onChange={(e) => {
                                    if (e.target.checked) {
                                      setSelectedDocs((prev) => (prev.includes(d.source) ? prev : [...prev, d.source]));
                                    } else {
                                      setSelectedDocs((prev) => prev.filter((source) => source !== d.source));
                                    }
                                  }}
                                />
                              </label>
                              <div className="admin-doc-info">
                                <span className="admin-doc-source">
                                  {d.source}
                                </span>
                                {d.document_year ? (
                                  <span className="admin-doc-year">Năm: {d.document_year}</span>
                                ) : null}
                              </div>
                              <button
                                className="btn btn-ghost btn-sm"
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
        </div>
        )}

        {activeTab === "feedback" && (
        <div className="admin-card admin-card--feedback">
          <div
            className="admin-row admin-row--head"
            style={{ marginBottom: "1rem" }}
          >
            <h2 className="admin-card-title" style={{ flex: 1 }}>
              Đánh giá người dùng
            </h2>
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
                  <div className="feedback-stat-number">
                    {feedbackData.total}
                  </div>
                  <div className="feedback-stat-label">Tổng đánh giá</div>
                </div>
                <div className="feedback-stat-card feedback-stat--up">
                  <div className="feedback-stat-number">
                    {feedbackData.up}
                  </div>
                  <div className="feedback-stat-label">Hữu ích</div>
                </div>
                <div className="feedback-stat-card feedback-stat--down">
                  <div className="feedback-stat-number">
                    {feedbackData.down}
                  </div>
                  <div className="feedback-stat-label">Chưa tốt</div>
                </div>
                <div className="feedback-stat-card feedback-stat--rate">
                  <div className="feedback-stat-number">
                    {feedbackData.total > 0
                      ? Math.round(
                          (feedbackData.up / feedbackData.total) * 100
                        ) + "%"
                      : "—"}
                  </div>
                  <div className="feedback-stat-label">Tỉ lệ tốt</div>
                </div>
              </div>

              <div className="feedback-tabs">
                <button
                  className={
                    "feedback-tab" +
                    (feedbackTab === "down" ? " active" : "")
                  }
                  onClick={() => setFeedbackTab("down")}
                >
                  Câu đánh giá thấp ({feedbackData.down})
                </button>
                <button
                  className={
                    "feedback-tab" + (feedbackTab === "all" ? " active" : "")
                  }
                  onClick={() => setFeedbackTab("all")}
                >
                  Tất cả
                </button>
              </div>

              <div className="feedback-list">
                {feedbackTab === "down" &&
                  (feedbackData.down_entries.length === 0 ? (
                    <p className="admin-status">
                      Chưa có đánh giá chưa tốt nào.
                    </p>
                  ) : (
                    feedbackData.down_entries.map((entry, i) => (
                      <div
                        key={i}
                        className="feedback-entry feedback-entry--down"
                      >
                        <div className="feedback-entry-header">
                          <span className="feedback-entry-rating down">
                            Chưa tốt
                          </span>
                          <span className="feedback-entry-time">
                            {entry.timestamp}
                          </span>
                        </div>
                        <div className="feedback-entry-qa">
                          <div className="feedback-entry-q">
                            <strong>Câu hỏi:</strong> {entry.question}
                          </div>
                          <div className="feedback-entry-a">
                            <strong>Trả lời:</strong>{" "}
                            {entry.answer || ""}
                          </div>
                        </div>
                        {entry.comment && (
                          <div className="feedback-entry-comment">
                            <strong>Góp ý:</strong> {entry.comment}
                          </div>
                        )}
                      </div>
                    ))
                  ))}
                {feedbackTab === "all" &&
                  (feedbackData.all_entries.length === 0 ? (
                    <p className="admin-status">
                      Chưa có đánh giá nào.
                    </p>
                  ) : (
                    feedbackData.all_entries.map((entry, i) => (
                      <div
                        key={i}
                        className={
                          "feedback-entry feedback-entry--" +
                          entry.rating
                        }
                      >
                        <div className="feedback-entry-header">
                          <span
                            className={
                              "feedback-entry-rating " + entry.rating
                            }
                          >
                            {entry.rating === "up"
                              ? "Hữu ích"
                              : "Chưa tốt"}
                          </span>
                          <span className="feedback-entry-time">
                            {entry.timestamp}
                          </span>
                        </div>
                        <div className="feedback-entry-qa">
                          <div className="feedback-entry-q">
                            <strong>Câu hỏi:</strong> {entry.question}
                          </div>
                          <div className="feedback-entry-a">
                            <strong>Trả lời:</strong>{" "}
                            {entry.answer || ""}
                          </div>
                        </div>
                        {entry.comment && (
                          <div className="feedback-entry-comment">
                            <strong>Góp ý:</strong> {entry.comment}
                          </div>
                        )}
                      </div>
                    ))
                  ))}
              </div>
            </>
          )}

          {!feedbackData && !feedbackLoading && (
            <p
              className="admin-status"
              style={{ textAlign: "center" }}
            >
              Nhấn "Tải dữ liệu" để xem thống kê đánh giá.
            </p>
          )}
        </div>
        )}

        {activeTab === "api" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>

            {/* ── Active model summary ──────────────────────────────── */}
            {apiSettings && (
              <div style={{
                display: "flex", gap: "0.75rem", flexWrap: "wrap",
                padding: "0.75rem 1rem", borderRadius: "10px",
                background: "#f8fafc", border: "1px solid #e2e8f0", fontSize: "0.85rem",
              }}>
                <span style={{ color: "#64748b" }}>Đang dùng:</span>
                <span style={{ fontWeight: 600, color: "#0f172a" }}>
                  {apiSettings.model}
                  <span style={{
                    marginLeft: "0.4rem", padding: "0.1rem 0.5rem", borderRadius: "999px",
                    fontSize: "0.72rem", fontWeight: 500,
                    background: apiSettings.provider === "anthropic" ? "#fdf4ff" : "#eff6ff",
                    color: apiSettings.provider === "anthropic" ? "#7c3aed" : "#2563eb",
                  }}>
                    {apiSettings.provider === "anthropic" ? "Anthropic" : "OpenAI"}
                  </span>
                </span>
              </div>
            )}

            <div className="admin-card">
              <div className="admin-row admin-row--head" style={{ marginBottom: "1.5rem" }}>
                <h2 className="admin-card-title" style={{ flex: 1 }}>Cấu hình API</h2>
                <button className="btn btn-ghost btn-sm" type="button" onClick={fetchApiSettings} disabled={apiSettingsLoading}>
                  {apiSettingsLoading ? "Đang tải..." : "Làm mới"}
                </button>
              </div>

              {apiSettingsLoading && !apiSettings && <p className="admin-status">Đang tải...</p>}

              {apiSettings && (
                <form onSubmit={handleSaveApiSettings} style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>

                  {/* ── LLM (Output) ──────────────────────────────────── */}
                  <div>
                    <p style={{ fontSize: "0.8rem", fontWeight: 600, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "1rem" }}>
                      LLM — Model trả lời
                    </p>

                    {/* Provider */}
                    <div className="admin-field" style={{ marginBottom: "1rem" }}>
                      <label className="admin-label">Provider</label>
                      <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
                        {[{ value: "openai", label: "OpenAI" }, { value: "anthropic", label: "Anthropic" }].map(p => (
                          <label key={p.value} style={{
                            display: "flex", alignItems: "center", gap: "0.4rem",
                            padding: "0.5rem 1rem", borderRadius: "8px", cursor: "pointer",
                            border: `2px solid ${apiForm.provider === p.value ? "#2563eb" : "#e5e7eb"}`,
                            background: apiForm.provider === p.value ? "#eff6ff" : "#fff",
                            fontWeight: apiForm.provider === p.value ? 600 : 400,
                            fontSize: "0.9rem", transition: "all 0.15s",
                          }}>
                            <input type="radio" name="llm_provider" value={p.value}
                              checked={apiForm.provider === p.value}
                              onChange={() => {
                                const def = p.value === "anthropic" ? "claude-sonnet-4-6" : "gpt-4o-mini";
                                setApiForm(f => ({ ...f, provider: p.value, model: def, openai_api_key: "", anthropic_api_key: "" }));
                              }}
                              style={{ accentColor: "#2563eb" }} />
                            {p.label}
                          </label>
                        ))}
                      </div>
                    </div>

                    {/* Model */}
                    <div className="admin-field" style={{ marginBottom: "1rem" }}>
                      <label className="admin-label">Model</label>
                      <select className="admin-input" value={apiForm.model}
                        onChange={e => setApiForm(f => ({ ...f, model: e.target.value }))}
                        style={{ maxWidth: "320px" }}>
                        {(apiForm.provider === "anthropic" ? ANTHROPIC_MODELS : OPENAI_MODELS).map(m => (
                          <option key={m.value} value={m.value}>{m.label}</option>
                        ))}
                      </select>
                    </div>

                    <p style={{ fontSize: "0.78rem", color: "#94a3b8", marginTop: "0.5rem" }}>
                      API keys are configured via environment variables on the server.
                    </p>
                  </div>

                  {apiSettingsMsg && (
                    <p style={{
                      padding: "0.5rem 0.75rem", borderRadius: "6px", fontSize: "0.875rem",
                      background: apiSettingsMsg.type === "success" ? "#f0fdf4" : "#fef2f2",
                      color: apiSettingsMsg.type === "success" ? "#16a34a" : "#dc2626",
                      border: `1px solid ${apiSettingsMsg.type === "success" ? "#bbf7d0" : "#fecaca"}`,
                      margin: 0,
                    }}>
                      {apiSettingsMsg.text}
                    </p>
                  )}

                  <div>
                    <button type="submit" className="btn btn-primary" disabled={apiSettingsSaving} style={{ minWidth: "120px" }}>
                      {apiSettingsSaving ? "Đang lưu..." : "Lưu cấu hình"}
                    </button>
                  </div>
                </form>
              )}
            </div>
          </div>
        )}

        </div>
      </section>
    </>
  );
}

export default AdminPage;
