const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function getJSON(path) {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const api = {
  stats: () => getJSON("/stats"),
  recentTransactions: (limit = 40) => getJSON(`/transactions/recent?limit=${limit}`),
  recentAlerts: (limit = 8) => getJSON(`/alerts/recent?limit=${limit}`),
  explain: (transactionId) => getJSON(`/explain/${encodeURIComponent(transactionId)}`),
  driftSummary: () => getJSON("/drift/summary"),
  sendFeedback: async (transactionId, actualIsFraud) => {
    const res = await fetch(`${BASE_URL}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transaction_id: transactionId, actual_is_fraud: actualIsFraud }),
    });
    if (!res.ok) throw new Error(`feedback -> ${res.status}`);
    return res.json();
  },
};
