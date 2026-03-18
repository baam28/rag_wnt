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
  } = state;

  const {
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
  } = handlers;

  const [activeTab, setActiveTab] = React.useState("analytics");
  const [analyticsData, setAnalyticsData] = React.useState(null);
  const [analyticsLoading, setAnalyticsLoading] = React.useState(false);
  const [selectedMetric, setSelectedMetric] = React.useState('messages');
  const [users, setUsers] = React.useState([]);
  const [usersLoading, setUsersLoading] = React.useState(false);

  const [pwdForm, setPwdForm] = React.useState({ userId: null, value: "" });

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
                  className="admin-input"
                  onChange={(e) => setUploadFile(e.target.files[0] || null)}
                />
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
              <div className="admin-field">
                <label
                  className="admin-label"
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "0.5rem",
                    cursor: "pointer",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={skipSummary}
                    onChange={(e) => setSkipSummary(e.target.checked)}
                  />
                  Bỏ qua tóm tắt (nhanh hơn, không gọi LLM)
                </label>
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
              <button
                className="btn btn-danger btn-sm"
                onClick={handleClearDb}
                disabled={isClearingDb}
              >
                {isClearingDb ? "Đang xóa..." : "Xóa toàn bộ DB"}
              </button>
            </div>
            {clearDbStatus && (
              <p
                className="admin-status"
                style={{ marginTop: 0, marginBottom: "0.5rem" }}
              >
                {clearDbStatus}
              </p>
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
                      {docsLoading && (
                        <p className="admin-status">Đang tải...</p>
                      )}
                      {docsError && (
                        <p className="admin-error">{docsError}</p>
                      )}
                      {!docsLoading && docs.length === 0 && (
                        <p className="admin-status">Trống.</p>
                      )}
                      {docs.length > 0 && (
                        <ul className="admin-doc-list">
                          {docs.map((d) => (
                            <li
                              key={d.source}
                              className="admin-doc-item"
                            >
                              <div className="admin-doc-info">
                                <span className="admin-doc-source">
                                  {d.source}
                                </span>
                                <span className="admin-doc-chunks">
                                  ({d.parent_count} tài liệu gốc)
                                </span>
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

        </div>
      </section>
    </>
  );
}

export default AdminPage;
