import React, { useState } from 'react';

function LandingPage({ onLoggedIn }) {
  const [mode, setMode] = useState("login"); // 'login' | 'register'
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showPwd, setShowPwd] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!username.trim() || !password) {
      setError("Vui lòng nhập tên đăng nhập và mật khẩu.");
      return;
    }
    if (mode === "register" && !email.trim()) {
      setError("Vui lòng nhập địa chỉ email.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const endpoint = mode === "register" ? "/auth/register" : "/auth/login";
      const resp = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          mode === "register"
            ? { username: username.trim(), email: email.trim(), password }
            : { username: username.trim(), password }
        ),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(data.detail || "Không thể " + (mode === "register" ? "đăng ký" : "đăng nhập"));
      }
      const token = data.token;
      if (!token) {
        throw new Error("Thiếu token phản hồi từ server.");
      }
      try {
        window.localStorage.setItem("rag_jwt", token);
      } catch {
        // ignore
      }
      onLoggedIn(token, {
        username: data.user?.username || username.trim(),
        isAdmin: !!data.user?.is_admin,
      });
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="landing-main">
      {/* Left branding panel */}
      <div className="landing-panel-left">
        <div className="landing-brand-badge">
          <img src={import.meta.env.BASE_URL + "logo.jpg"} alt="PharmaAI" />
        </div>
        <h1 className="landing-panel-title">PharmaAI</h1>
        <p className="landing-panel-sub">
          Trợ lý dược thông minh — hỏi đáp dược liệu, tra cứu thuốc<br />và tư vấn pháp lý ngành dược tức thì.
        </p>
        <ul className="landing-features">
          <li className="landing-feature">
            <span className="landing-feature-icon">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2v-4M9 21H5a2 2 0 0 1-2-2v-4m0 0h18"/>
              </svg>
            </span>
            <span>Tra cứu giá thuốc thời gian thực</span>
          </li>
          <li className="landing-feature">
            <span className="landing-feature-icon">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              </svg>
            </span>
            <span>Hỏi đáp quy định pháp lý dược</span>
          </li>
          <li className="landing-feature">
            <span className="landing-feature-icon">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                <line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/>
              </svg>
            </span>
            <span>Tra cứu thông tin các hoạt chất</span>
          </li>
        </ul>
      </div>

      {/* Right auth panel */}
      <div className="landing-panel-right">
        <div className="landing-auth-card">
          <h2 className="landing-auth-title">
            {mode === "login" ? "Đăng nhập" : "Tạo tài khoản"}
          </h2>
          <p className="landing-auth-sub">
            {mode === "login"
              ? "Chào mừng trở lại! Vui lòng đăng nhập để tiếp tục."
              : "Tạo tài khoản miễn phí để bắt đầu sử dụng."}
          </p>

          <div className="landing-tabs">
            <button
              type="button"
              className={"landing-tab" + (mode === "login" ? " active" : "")}
              onClick={() => { setMode("login"); setError(""); setEmail(""); setUsername(""); setPassword(""); }}
            >
              Đăng nhập
            </button>
            <button
              type="button"
              className={"landing-tab" + (mode === "register" ? " active" : "")}
              onClick={() => { setMode("register"); setError(""); setEmail(""); setUsername(""); setPassword(""); }}
            >
              Đăng ký
            </button>
          </div>

          <form className="landing-form" onSubmit={handleSubmit}>
            <div className="landing-field">
              <label className="landing-label">Tên đăng nhập</label>
              <input
                type="text"
                className="landing-input"
                placeholder="vd: nhanvien1"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
              />
            </div>
            {mode === "register" && (
              <div className="landing-field">
                <label className="landing-label">Email</label>
                <input
                  type="email"
                  className="landing-input"
                  placeholder="vd: nhanvien1@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoComplete="email"
                />
              </div>
            )}
            <div className="landing-field">
              <label className="landing-label">Mật khẩu</label>
              <div style={{ position: "relative" }}>
                <input
                  type={showPwd ? "text" : "password"}
                  className="landing-input landing-input--password"
                  placeholder={mode === "register" ? "Ít nhất 6 ký tự" : "Mật khẩu"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete={mode === "login" ? "current-password" : "new-password"}
                  style={{ paddingRight: "2.5rem" }}
                />
                <button
                  type="button"
                  onClick={() => setShowPwd(p => !p)}
                  className="landing-password-toggle"
                  aria-label={showPwd ? "Ẩn mật khẩu" : "Hiện mật khẩu"}
                  aria-pressed={showPwd}
                  tabIndex={-1}
                >
                  {showPwd ? (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <circle cx="12" cy="12" r="6" />
                      <line x1="6" y1="18" x2="18" y2="6" />
                    </svg>
                  ) : (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <circle cx="12" cy="12" r="6" />
                      <circle cx="12" cy="12" r="2" />
                    </svg>
                  )}
                </button>
              </div>
            </div>

            {error && <div className="landing-error">{error}</div>}

            <button
              type="submit"
              className="btn btn-primary btn-full landing-submit"
              disabled={loading}
              style={{ minHeight: "2.75rem", fontSize: ".9375rem", fontWeight: 700, marginTop: ".5rem" }}
            >
              {loading ? (
                <span style={{ display: "flex", alignItems: "center", gap: ".5rem" }}>
                  <span className="spinner" />
                  {mode === "register" ? "Đang đăng ký..." : "Đang đăng nhập..."}
                </span>
              ) : mode === "register" ? "Tạo tài khoản" : "Đăng nhập →"}
            </button>
          </form>

          <p className="landing-note">
            {mode === "login"
              ? "Chưa có tài khoản? Chuyển sang tab Đăng ký."
              : "Tài khoản sẽ được lưu và dùng để truy cập hệ thống."}
          </p>
        </div>
      </div>
    </section>
  );
}

export default LandingPage;
