import { useEffect, useState, useCallback, useRef } from "react";
import { api } from "./api";
import Header from "./components/Header.jsx";
import LiveFeed from "./components/LiveFeed.jsx";
import AlertsPanel from "./components/AlertsPanel.jsx";
import ExplainPanel from "./components/ExplainPanel.jsx";
import DriftPanel from "./components/DriftPanel.jsx";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [stats, setStats] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [explainData, setExplainData] = useState(null);
  const [drift, setDrift] = useState(null);
  const [connected, setConnected] = useState(true);
  const hasAutoSelected = useRef(false);

  const pollFast = useCallback(async () => {
    try {
      const [s, t, a] = await Promise.all([
        api.stats(),
        api.recentTransactions(40),
        api.recentAlerts(8),
      ]);
      setStats(s);
      setTransactions(t);
      setAlerts(a);
      setConnected(true);

      if (!hasAutoSelected.current && a.length > 0) {
        hasAutoSelected.current = true;
        setSelectedId(a[0].transaction_id);
      }
    } catch {
      setConnected(false);
    }
  }, []);

  const pollDrift = useCallback(async () => {
    try {
      setDrift(await api.driftSummary());
    } catch {
      /* drift panel just stays on its last known state */
    }
  }, []);

  useEffect(() => {
    pollFast();
    pollDrift();
    const fastTimer = setInterval(pollFast, 2000);
    const driftTimer = setInterval(pollDrift, 15000);
    return () => {
      clearInterval(fastTimer);
      clearInterval(driftTimer);
    };
  }, [pollFast, pollDrift]);

  useEffect(() => {
    if (!selectedId) {
      setExplainData(null);
      return;
    }
    let cancelled = false;
    api
      .explain(selectedId)
      .then((data) => {
        if (!cancelled) setExplainData(data);
      })
      .catch(() => {
        if (!cancelled) setExplainData(null);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  const handleFeedback = useCallback((transactionId, actualIsFraud) => {
    api.sendFeedback(transactionId, actualIsFraud).catch((e) => console.error(e));
  }, []);

  return (
    <>
      <Header stats={stats} connected={connected} />
      <div className="layout">
        <LiveFeed transactions={transactions} />
        <div className="col">
          <AlertsPanel alerts={alerts} selectedId={selectedId} onSelect={setSelectedId} />
          <ExplainPanel data={explainData} onFeedback={handleFeedback} />
          <DriftPanel
            drift={drift}
            onOpenReport={() => window.open(`${API_URL}/reports/drift_report.html`, "_blank")}
          />
        </div>
      </div>
    </>
  );
}
