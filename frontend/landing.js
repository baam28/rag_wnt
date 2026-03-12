import React, { useState } from 'react';

function LandingPage({ onLoggedIn }) {
  const [mode, setMode] = useState("login"); // 'login' | 'register'
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    if (!username.trim() || !password) {
      setError("Vui lòng nhập tên đăng nhập và mật khẩu.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const endpoint = mode === "register" ? "/auth/register" : "/auth/login";
      const resp = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username.trim(), password }),
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
      <div className="landing-hero landing-hero--auth">
        <h1 className="landing-title">Trợ lý Web Nhà Thuốc AI</h1>
        <p className="landing-subtitle">
          Đăng nhập để sử dụng chatbot hỏi đáp tài liệu nội bộ, quy trình và pháp lý nhà thuốc.
        </p>

        <div className="landing-tabs">
          <button
            type="button"
            className={"landing-tab" + (mode === "login" ? " active" : "")}
            onClick={() => setMode("login")}
          >
            Đăng nhập
          </button>
          <button
            type="button"
            className={"landing-tab" + (mode === "register" ? " active" : "")}
            onClick={() => setMode("register")}
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
          <div className="landing-field">
            <label className="landing-label">Mật khẩu</label>
            <input
              type="password"
              className="landing-input"
              placeholder="Mật khẩu từ 6 ký tự"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete={mode === "login" ? "current-password" : "new-password"}
            />
          </div>
          {error && <div className="landing-error">{error}</div>}
          <button
            type="submit"
            className="btn btn-primary btn-full landing-submit"
            disabled={loading}
          >
            {loading
              ? mode === "register"
                ? "Đang đăng ký..."
                : "Đang đăng nhập..."
              : mode === "register"
              ? "Đăng ký"
              : "Đăng nhập"}
          </button>
        </form>

        <p className="landing-note">
          Tài khoản đăng ký mới sẽ được lưu trong hệ thống và sử dụng để truy cập chatbot.
        </p>
      </div>
    </section>
  );
}

export default LandingPage;

