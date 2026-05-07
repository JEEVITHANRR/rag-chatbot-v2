import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const FILE_ICON = (
  <svg width="16" height="16" fill="none" stroke="#4f7fff" strokeWidth="1.5" viewBox="0 0 24 24">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <polyline points="14 2 14 8 20 8"/>
  </svg>
)

export default function Dashboard({ stats, documents, messages }) {
  const userMsgs = messages.filter(m => m.role === 'user')

  const activityData = (() => {
    if (userMsgs.length === 0) return []
    const map = {}
    userMsgs.forEach(m => {
      const key = m.time || '00:00'
      map[key] = (map[key] || 0) + 1
    })
    return Object.entries(map).slice(-8).map(([time, count]) => ({ time, count }))
  })()

  const docChartData = documents.map(d => ({
    name: d.name.length > 12 ? d.name.slice(0, 12) + '…' : d.name,
    chunks: d.chunks,
  }))

  return (
    <>
      <div className="page-header">
        <div className="page-title">Dashboard</div>
        <div className="page-sub">Overview of your document intelligence session</div>
      </div>

      <div className="stats-row">
        <div className="stat-card">
          <div className="stat-label">Documents loaded</div>
          <div className="stat-val">{stats.docs}</div>
          <div className="stat-sub">PDF / TXT / MD files</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Questions asked</div>
          <div className="stat-val">{stats.messages}</div>
          <div className="stat-sub">This session</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Total chunks</div>
          <div className="stat-val">{stats.chunks}</div>
          <div className="stat-sub">Vector embeddings stored</div>
        </div>
      </div>

      <div className="dashboard">
        {activityData.length > 0 && (
          <div className="chart-section">
            <div className="section-title">Query activity</div>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={activityData} barSize={14}>
                <XAxis dataKey="time" tick={{ fill: '#555b72', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#555b72', fontSize: 11 }} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip
                  contentStyle={{ background: '#22263a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 6, fontSize: 12 }}
                  labelStyle={{ color: '#8b90a4' }}
                  itemStyle={{ color: '#e8eaf0' }}
                />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {activityData.map((_, i) => <Cell key={i} fill="#4f7fff" />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {docChartData.length > 0 && (
          <div className="chart-section">
            <div className="section-title">Chunks per document</div>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={docChartData} barSize={14} layout="vertical">
                <XAxis type="number" tick={{ fill: '#555b72', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" tick={{ fill: '#8b90a4', fontSize: 11 }} axisLine={false} tickLine={false} width={100} />
                <Tooltip
                  contentStyle={{ background: '#22263a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 6, fontSize: 12 }}
                  labelStyle={{ color: '#8b90a4' }}
                  itemStyle={{ color: '#e8eaf0' }}
                />
                <Bar dataKey="chunks" radius={[0, 3, 3, 0]} fill="#22c55e" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        <div className="doc-list">
          <div className="doc-list-header">Loaded documents</div>
          {documents.length === 0 ? (
            <div className="empty-state">No documents loaded yet. Go to Documents to upload.</div>
          ) : (
            documents.map((doc, i) => (
              <div className="doc-row" key={i}>
                <div className="doc-icon">{FILE_ICON}</div>
                <div className="doc-name">{doc.name}</div>
                <div className="doc-meta">{doc.pages} page{doc.pages !== 1 ? 's' : ''}</div>
                <span className="chunk-badge">{doc.chunks} chunks</span>
              </div>
            ))
          )}
        </div>
      </div>
    </>
  )
}
