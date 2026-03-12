import React from 'react';

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
        <div className="admin-tabs" style={{ display: "flex", gap: "1rem", marginBottom: "1.5rem", borderBottom: "1px solid #e5e7eb", paddingBottom: "0.5rem", flexShrink: 0 }}>
          <button className={`btn ${activeTab === "analytics" ? "btn-primary" : "btn-ghost"}`} onClick={() => setActiveTab("analytics")}>Thống kê</button>
          <button className={`btn ${activeTab === "users" ? "btn-primary" : "btn-ghost"}`} onClick={() => setActiveTab("users")}>Quản lý User</button>
          <button className={`btn ${activeTab === "docs" ? "btn-primary" : "btn-ghost"}`} onClick={() => setActiveTab("docs")}>Tài liệu & DB</button>
          <button className={`btn ${activeTab === "feedback" ? "btn-primary" : "btn-ghost"}`} onClick={() => setActiveTab("feedback")}>Đánh giá</button>
        </div>

        <div style={{ flex: 1, overflowY: "auto", paddingRight: "0.5rem", paddingBottom: "2rem" }}>
        
        {activeTab === "analytics" && (
          <div className="admin-card">
            <h2 className="admin-card-title" style={{ marginBottom: "1.5rem" }}>Thống kê tổng quan</h2>
            {analyticsLoading ? <p className="admin-status">Đang tải...</p> : analyticsData ? (
              <div className="feedback-stats">
                <div className="feedback-stat-card feedback-stat--total">
                  <div className="feedback-stat-number">{analyticsData.total_users}</div>
                  <div className="feedback-stat-label">Tổng số User</div>
                </div>
                <div className="feedback-stat-card feedback-stat--up">
                  <div className="feedback-stat-number">{analyticsData.total_sessions}</div>
                  <div className="feedback-stat-label">Số cuộc hội thoại</div>
                </div>
                <div className="feedback-stat-card feedback-stat--down">
                  <div className="feedback-stat-number">{analyticsData.total_messages}</div>
                  <div className="feedback-stat-label">Tổng tin nhắn</div>
                </div>
                <div className="feedback-stat-card feedback-stat--rate">
                  <div className="feedback-stat-number">{analyticsData.feedback.total}</div>
                  <div className="feedback-stat-label">Đánh giá thu thập được</div>
                </div>
              </div>
            ) : <p className="admin-status">Không có dữ liệu.</p>}
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
                  <option value="drug">drug — Thông tin dược phẩm</option>
                  <option value="legal">legal — Văn bản pháp lý</option>
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
                      {c.name}
                    </div>
                    <button
                      className="btn btn-danger btn-sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteCollection(c.name);
                      }}
                      title={`Xóa collection "${c.name}"`}
                    >
                      Xóa
                    </button>
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

