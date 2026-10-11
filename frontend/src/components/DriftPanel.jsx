export default function DriftPanel({ drift, onOpenReport }) {
  if (!drift || !drift.available) {
    return (
      <div className="panel" style={{ borderBottom: "none" }}>
        <div className="panel-title">
          Data drift <span className="count">vs. training baseline</span>
        </div>
        <div className="drift-body">
          <div style={{ color: "var(--text-dim)", fontSize: 12 }}>
            No drift report yet. Run <code>python -m src.drift_report</code>.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="panel" style={{ borderBottom: "none" }}>
      <div className="panel-title">
        Data drift <span className="count">vs. training baseline</span>
      </div>
      <div className="drift-body">
        <div className="drift-summary">
          <span className="frac">{drift.drifted_count}/{drift.total_count}</span>
          <span className="desc">features drifted from training baseline</span>
        </div>
        {drift.features.map((f) => (
          <div className="drift-feature" key={f.name}>
            <div className={`indicator ${f.drifted ? "on" : "off"}`} />
            <div className="name">{f.name}</div>
            <div className="dscore">{f.score.toFixed(2)}</div>
          </div>
        ))}
        <div className="drift-link" onClick={onOpenReport}>Open full report</div>
      </div>
    </div>
  );
}
