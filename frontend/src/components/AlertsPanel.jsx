function fmtTime(iso) {
  return new Date(iso).toTimeString().slice(0, 8);
}

export default function AlertsPanel({ alerts, selectedId, onSelect }) {
  return (
    <div className="panel">
      <div className="panel-title">
        Active alerts <span className="count">{alerts.length}</span>
      </div>
      <div>
        {alerts.map((a) => (
          <div
            key={a.transaction_id}
            className={`alert-item ${selectedId === a.transaction_id ? "selected" : ""}`}
            onClick={() => onSelect(a.transaction_id)}
          >
            <div className="row1">
              <span className="card-id mono">{a.card_id}</span>
              <span className="score mono">{a.score.toFixed(3)}</span>
            </div>
            <div className="row2">
              <span>{a.merchant_category} {'\u00b7'} ${a.amount.toFixed(2)}</span>
              <span>{fmtTime(a.timestamp)}</span>
            </div>
          </div>
        ))}
        {alerts.length === 0 && (
          <div style={{ padding: "16px 20px", color: "var(--text-dim)", fontSize: 12 }}>
            No alerts yet.
          </div>
        )}
      </div>
    </div>
  );
}
