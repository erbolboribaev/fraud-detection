function statusOf(score) {
  if (score >= 0.85) return { cls: "alert", label: "fraud" };
  if (score >= 0.3) return { cls: "watch", label: "watch" };
  return { cls: "clear", label: "clear" };
}

function fmtTime(iso) {
  return new Date(iso).toTimeString().slice(0, 8);
}

export default function LiveFeed({ transactions }) {
  // Defensive: older log entries (written before a schema change, e.g.) may
  // be missing fields the current schema always includes. Skip those rather
  // than crashing the whole panel.
  transactions = transactions.filter((t) => typeof t.amount === "number" && typeof t.score === "number");
  return (
    <div className="col">
      <div className="panel-title">
        Live transaction stream <span className="count">{transactions.length} rows</span>
      </div>
      <div className="feed-head">
        <div>time</div><div>card</div><div>category</div>
        <div style={{ textAlign: "right" }}>amount</div>
        <div>score</div><div></div>
        <div style={{ textAlign: "right" }}>status</div>
      </div>
      <div className="feed-body">
        {transactions.map((t) => {
          const st = statusOf(t.score);
          const color =
            st.cls === "alert" ? "var(--alert)" : st.cls === "watch" ? "var(--watch)" : "var(--clear)";
          return (
            <div className="feed-row" key={t.transaction_id}>
              <div className="time mono">{fmtTime(t.timestamp)}</div>
              <div className="card mono">{t.card_id}</div>
              <div className="cat">{t.merchant_category}</div>
              <div className="amt mono">${t.amount.toFixed(2)}</div>
              <div className="score-bar-wrap">
                <div className="score-track">
                  <div className="score-fill" style={{ width: `${t.score * 100}%`, background: color }} />
                </div>
              </div>
              <div className="mono" style={{ color: "var(--text-muted)" }}>{t.score.toFixed(3)}</div>
              <div style={{ textAlign: "right" }}>
                <span className={`tag ${st.cls}`}>{st.label}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
