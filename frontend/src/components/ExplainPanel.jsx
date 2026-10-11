export default function ExplainPanel({ data, onFeedback }) {
  if (!data) {
    return (
      <div className="panel">
        <div className="panel-title">Explain selected transaction</div>
        <div className="explain-body">
          <div style={{ color: "var(--text-dim)", fontSize: 12, padding: "8px 0" }}>
            Select an alert to see its breakdown.
          </div>
        </div>
      </div>
    );
  }

  const max = Math.max(...data.contributions.map((c) => Math.abs(c.value)));

  return (
    <div className="panel">
      <div className="panel-title">Explain selected transaction</div>
      <div className="explain-body">
        <div className="explain-txn">
          <span className="id mono">{data.transaction_id}</span>{' '}{'\u00b7'}{' '}
          {data.card_id}{' '}{'\u00b7'}{' '}${data.amount?.toFixed(2)}
        </div>
        {data.contributions.map((c) => {
          const pct = (Math.abs(c.value) / max) * 48;
          const cls = c.value > 0 ? "pos" : "neg";
          return (
            <div className="contrib-row" key={c.feature}>
              <div className="contrib-name">{c.feature}</div>
              <div className="contrib-track">
                <div className="contrib-mid" />
                <div className={`contrib-fill ${cls}`} style={{ width: `${pct}%` }} />
              </div>
              <div className={`contrib-val ${cls}`}>
                {c.value > 0 ? "+" : ""}
                {c.value.toFixed(2)}
              </div>
            </div>
          );
        })}
        <div className="explain-total">
          <span className="label">final probability</span>
          <span className="value">{data.probability.toFixed(3)}</span>
        </div>
        <div className="feedback-row">
          <button className="fb-btn fb-fraud" onClick={() => onFeedback(data.transaction_id, true)}>
            Confirm fraud
          </button>
          <button className="fb-btn fb-legit" onClick={() => onFeedback(data.transaction_id, false)}>
            Mark legitimate
          </button>
        </div>
      </div>
    </div>
  );
}
