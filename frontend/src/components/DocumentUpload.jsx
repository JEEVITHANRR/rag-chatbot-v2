import { useState, useRef } from 'react'

const UPLOAD_ICON = (
  <svg width="40" height="40" fill="none" stroke="currentColor" strokeWidth="1.2" viewBox="0 0 24 24">
    <polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/>
    <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
  </svg>
)

const FILE_ICON = (
  <svg width="16" height="16" fill="none" stroke="#4f7fff" strokeWidth="1.5" viewBox="0 0 24 24">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <polyline points="14 2 14 8 20 8"/>
  </svg>
)

export default function DocumentUpload({ documents, setDocuments, setTotalChunks, apiUrl }) {
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [alert, setAlert] = useState(null)
  const inputRef = useRef()

  const uploadFile = async file => {
    if (!file) return

    const ext = file.name.split('.').pop().toLowerCase()
    if (!['pdf', 'txt', 'md'].includes(ext)) {
      setAlert({ type: 'error', text: 'Only PDF, TXT, and MD files are supported.' })
      return
    }

    setUploading(true)
    setProgress(20)
    setAlert(null)

    const form = new FormData()
    form.append('file', file)

    try {
      setProgress(50)
      const res = await fetch(`${apiUrl}/api/upload`, { method: 'POST', body: form })
      setProgress(90)
      const data = await res.json()

      if (!res.ok) throw new Error(data.error || 'Upload failed')

      setDocuments(data.documents || [])
      setTotalChunks(prev => prev + (data.chunks || 0))
      setProgress(100)
      setAlert({ type: 'success', text: `${file.name} processed — ${data.chunks} chunks indexed.` })
    } catch (e) {
      setAlert({ type: 'error', text: e.message })
    } finally {
      setUploading(false)
      setTimeout(() => setProgress(0), 800)
    }
  }

  const onDrop = e => {
    e.preventDefault()
    setDragOver(false)
    uploadFile(e.dataTransfer.files[0])
  }

  return (
    <>
      <div className="page-header">
        <div className="page-title">Documents</div>
        <div className="page-sub">Upload PDF, TXT, or MD files to index for chat</div>
      </div>

      <div className="upload-page">
        <div
          className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
        >
          <div className="upload-icon">{UPLOAD_ICON}</div>
          <div className="upload-title">Drag & drop a file here</div>
          <div className="upload-sub">PDF, TXT, or MD · up to 20 MB</div>
          <button className="upload-btn" disabled={uploading}>
            {uploading ? 'Processing…' : 'Browse file'}
          </button>
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.txt,.md"
            style={{ display: 'none' }}
            onChange={e => uploadFile(e.target.files[0])}
          />
        </div>

        {progress > 0 && (
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
        )}

        {alert && (
          <div className={`alert alert-${alert.type}`}>{alert.text}</div>
        )}

        <div className="doc-list" style={{ marginTop: 20 }}>
          <div className="doc-list-header">Indexed documents ({documents.length})</div>
          {documents.length === 0 ? (
            <div className="empty-state">No documents uploaded yet.</div>
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
