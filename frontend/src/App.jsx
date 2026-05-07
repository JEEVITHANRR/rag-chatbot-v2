import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import Dashboard from './components/Dashboard'
import ChatWindow from './components/ChatWindow'
import DocumentUpload from './components/DocumentUpload'

const API = import.meta.env.VITE_API_URL || 'http://localhost:5000'

export default function App() {
  const [activeTab, setActiveTab] = useState('chat')
  const [documents, setDocuments] = useState([])
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'bot',
      text: 'Hello! I\'m your RAG-powered document assistant. Upload a PDF or text file, then ask me anything about it.',
      sources: [],
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }
  ])
  const [totalChunks, setTotalChunks] = useState(0)

  useEffect(() => {
    fetch(`${API}/api/documents`)
      .then(r => r.json())
      .then(d => setDocuments(d.documents || []))
      .catch(() => {})
  }, [])

  const stats = {
    docs: documents.length,
    messages: messages.filter(m => m.role === 'user').length,
    chunks: totalChunks,
  }

  return (
    <div className="app">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} docsCount={documents.length} />
      <main className="main">
        {activeTab === 'dashboard' && (
          <Dashboard stats={stats} documents={documents} messages={messages} />
        )}
        {activeTab === 'chat' && (
          <ChatWindow
            messages={messages}
            setMessages={setMessages}
            documents={documents}
            apiUrl={API}
          />
        )}
        {activeTab === 'upload' && (
          <DocumentUpload
            documents={documents}
            setDocuments={setDocuments}
            setTotalChunks={setTotalChunks}
            apiUrl={API}
          />
        )}
      </main>
    </div>
  )
}
