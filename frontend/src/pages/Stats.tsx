import { useEffect, useState } from 'react'
import { getStats } from '../api'
import PageHeader from '../components/PageHeader'

const LEADERBOARD = [
  ['SVM-RBF (Tuned)',    '0.5748','0.5701','—',     'tuned'],
  ['LDA',                '0.5748','0.5696','0.7577','baseline'],
  ['MLP',                '0.5681','0.5657','0.7659','baseline'],
  ['Logistic Regression','0.5648','0.5631','0.7565','baseline'],
  ['Elastic Net',        '0.5581','0.5539','0.7584','baseline'],
  ['RF (Tuned)',         '0.5615','0.5475','—',     'tuned'],
  ['SVM-Linear',         '0.5515','0.5441','0.7587','baseline'],
  ['Gradient Boosting',  '0.5382','0.5384','0.7402','baseline'],
  ['Random Forest',      '0.5482','0.5317','0.7572','baseline'],
  ['AdaBoost',           '0.5150','0.5123','0.6963','baseline'],
  ['Decision Tree',      '0.5083','0.5017','0.6924','baseline'],
  ['Naive Bayes',        '0.4186','0.4081','0.6840','baseline'],
  ['XGBoost (Tuned)',    '0.3555','0.3537','—',     'tuned'],
  ['XGBoost',            '0.3455','0.3428','0.5243','baseline'],
]

export default function Stats() {
  const [data, setData] = useState<any>(null)
  const [tab,  setTab]  = useState<'models'|'stats'|'rag'>('models')

  useEffect(() => { getStats().then(setData) }, [])

  if (!data) return <div className="p-6 text-gray-500 text-sm">Loading stats…</div>

  const kpis = [
    ['2,000',  'Papers'],
    ['27 → 80','Features'],
    ['12',     'ML Models'],
    ['57.5%',  'Best Acc.'],
    ['0.5701', 'Best F1'],
    ['0.7577', 'Best AUC'],
    [String(data.rag.chunks), 'Chunks'],
    [data.rag.eval_pass, 'Eval Pass'],
  ]

  return (
    <div className="p-6 max-w-5xl">
      <PageHeader icon="📊" title="Project Statistics"
        sub="Verified ML pipeline results, statistical analysis, RAG system metrics" />

      {/* KPI row */}
      <div className="grid grid-cols-4 md:grid-cols-8 gap-3 mb-6">
        {kpis.map(([val, lbl]) => (
          <div key={lbl} className="card text-center py-4">
            <div className="text-lg font-bold text-blue-400">{val}</div>
            <div className="text-xs text-gray-500 uppercase tracking-wide mt-0.5">{lbl}</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-4 border-b border-gray-800 pb-3">
        {(['models','stats','rag'] as const).map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              tab === t ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-gray-200'}`}>
            {t === 'models' ? '🏆 Leaderboard'
              : t === 'stats' ? '📈 Statistical Analysis'
              : '🤖 RAG & LLM'}
          </button>
        ))}
      </div>

      {tab === 'models' && (
        <div className="card overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 border-b border-gray-800 text-xs uppercase tracking-wide">
                <th className="text-left pb-3 pr-4">Model</th>
                <th className="text-right pb-3 pr-4">Accuracy</th>
                <th className="text-right pb-3 pr-4">Macro-F1</th>
                <th className="text-right pb-3 pr-4">ROC-AUC</th>
                <th className="text-left pb-3">Type</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {LEADERBOARD.map(([model, acc, f1, auc, type], i) => (
                <tr key={i} className={`${i === 0 ? 'text-blue-300' : 'text-gray-300'}`}>
                  <td className="py-2.5 pr-4 font-medium">
                    {i === 0 && <span className="text-yellow-400 mr-1.5">★</span>}
                    {model}
                  </td>
                  <td className="py-2.5 pr-4 text-right tabular-nums">{acc}</td>
                  <td className="py-2.5 pr-4 text-right tabular-nums font-semibold">{f1}</td>
                  <td className="py-2.5 pr-4 text-right tabular-nums text-gray-400">{auc}</td>
                  <td className="py-2.5">
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      type === 'tuned'
                        ? 'bg-amber-900/40 text-amber-400 border border-amber-800'
                        : 'bg-gray-800 text-gray-500'}`}>
                      {type}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-gray-500 mt-3">
            Test set n=301 · 3-class OA prediction · R/caret pipeline
          </p>
        </div>
      )}

      {tab === 'stats' && (
        <div className="grid grid-cols-2 gap-4">
          <div className="card space-y-3">
            <h2 className="font-semibold text-white">Friedman Non-Parametric Test</h2>
            <Stat label="H₀" value="All 12 models share the same median CV F1" />
            <Stat label="χ²(11)" value={String(data.ml.friedman_chi2)} highlight />
            <Stat label="p-value" value={data.ml.friedman_p.toExponential(0)} highlight />
            <div className="bg-emerald-900/30 border border-emerald-800 rounded-lg px-4 py-3">
              <span className="text-emerald-400 font-bold">✅ REJECT H₀</span>
              <span className="text-emerald-600 text-sm ml-2">performance differences are real</span>
            </div>
          </div>
          <div className="card space-y-3">
            <h2 className="font-semibold text-white">Nemenyi Post-hoc Analysis</h2>
            <Stat label="Total pairs" value="66" />
            <Stat label="Significant pairs" value="7" highlight />
            <Stat label="α" value="0.05" />
            <div className="bg-gray-950 rounded-lg p-3 text-sm text-gray-400 leading-relaxed">
              XGBoost and Naive Bayes are provably worse than the top group.
              Top models (LDA, RF, Elastic Net) are statistically indistinguishable.
            </div>
            <Stat label="Top feature (Gini)" value="pub_ieee = 43.78" />
          </div>
        </div>
      )}

      {tab === 'rag' && (
        <div className="grid grid-cols-2 gap-4">
          <div className="card space-y-3">
            <h2 className="font-semibold text-white">RAG Pipeline</h2>
            <Stat label="Embedding model"  value={data.rag.embed_model} />
            <Stat label="Dimensions"       value="384" />
            <Stat label="Vector store"     value="ChromaDB 1.5.9" />
            <Stat label="Chunk size"       value={`${data.rag.chunk_size} chars`} />
            <Stat label="Top-k retrieval"  value={String(data.rag.top_k)} />
            <Stat label="Chunks indexed"   value={String(data.rag.chunks)} highlight />
            <Stat label="Eval pass rate"   value={data.rag.eval_pass} highlight />
          </div>
          <div className="card space-y-3">
            <h2 className="font-semibold text-white">Fine-tuned LLM</h2>
            <Stat label="Base model"   value="Qwen2.5-1.5B-Instruct" />
            <Stat label="Method"       value="QLoRA 4-bit NF4" />
            <Stat label="Examples"     value="8,343" />
            <Stat label="Train loss"   value="2.84 → 0.49" highlight />
            <Stat label="Export"       value="GGUF Q4_K_M (934 MB)" />
            <Stat label="Deployment"   value="Ollama (local, no cloud)" />
            <Stat label="TTFT"         value="~10 s on CPU" />
          </div>
        </div>
      )}
    </div>
  )
}

function Stat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex justify-between items-center py-1 border-b border-gray-800 last:border-0">
      <span className="text-xs text-gray-500">{label}</span>
      <span className={`text-sm font-semibold ${highlight ? 'text-blue-400' : 'text-gray-200'}`}>
        {value}
      </span>
    </div>
  )
}
