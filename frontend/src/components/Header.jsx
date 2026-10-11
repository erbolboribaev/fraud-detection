export default function Header({ stats, connected }) {
  const rate = stats ? (stats.fraud_rate * 100).toFixed(2) : "0.00";
  return (
    <header>
      <div className="brand">
        <span className={`dot ${connected ? "" : "dot-off"}`}></span>
        Fraud Console
      </div>
      <div className="status">
        {connected ? "stream connected \u00b7 scoring live" : "reconnecting\u2026"}
      </div>
      <div className="metric-strip">
        <div className="metric">
          <div className="value mono">{stats ? stats.total_today : "\u2014"}</div>
          <div className="label">scored today</div>
        </div>
        <div className="metric alert-metric">
          <div className="value mono">{stats ? stats.fraud_today : "\u2014"}</div>
          <div className="label">flagged</div>
        </div>
        <div className="metric">
          <div className="value mono">{rate}%</div>
          <div className="label">fraud rate</div>
        </div>
      </div>
    </header>
  );
}
