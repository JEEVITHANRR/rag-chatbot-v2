import { useState, useRef, useEffect } from 'react'

const SEND_ICON = (
  <svg width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
    <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
  </svg>
)

export default function ChatWindow({ messages, setMessages, documents, apiUrl }) {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const send = async () => {
    const q = input.trim()
    if (!q || loading) return

    const userMsg = {
      id: Date.now(),
      role: 'user',
      text: q,
      sources: [],
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }

    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(`${apiUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      })
      const data = await res.json()

      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'bot',
        text: data.answer || data.error || 'Something went wrong.',
        sources: data.sources || [],
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }])
    } catch {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'bot',
        text: 'Could not reach the backend. Make sure the API is running.',
        sources: [],
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }])
    } finally {
      setLoading(false)
    }
  }

  const onKey = e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <>
      <div className="page-header">
        <div className="page-title">Document Chat</div>
        <div className="page-sub">
          {documents.length === 0
            ? 'Upload a document first, then ask questions about it'
            : `${documents.length} document${documents.length > 1 ? 's' : ''} loaded — ask anything`}
        </div>
      </div>

      <div className="chat-layout">
        <div className="chat-main">
          <div className="messages">
            {messages.map(msg => (
              <div key={msg.id} className={`msg ${msg.role}`}>
                <div className="msg-avatar">
                  {msg.role === 'user' ? 'JR' : 'AI'}
                </div>
                <div className="msg-content">
                  <div className="msg-bubble">{msg.text}</div>
                  {msg.sources.length > 0 && (
                    <div className="msg-sources">
                      {msg.sources.map((s, i) => (
                        <span key={i} className="source-tag">{s}</span>
                      ))}
                    </div>
                  )}
                  <div className="msg-time">{msg.time}</div>
                </div>
              </div>
            ))}

            {loading && (
              <div className="msg bot">
                <div className="msg-avatar">AI</div>
                <div className="msg-content">
                  <div className="msg-bubble">
                    <div className="typing-indicator">
                      <div className="typing-dot" />
                      <div className="typing-dot" />
                      <div className="typing-dot" />
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="chat-input-area">
            <div className="chat-input-row">
              <textarea
                className="chat-input"
                placeholder="Ask anything about your documents… (Enter to send)"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={onKey}
                rows={1}
              />
              <button className="send-btn" onClick={send} disabled={!input.trim() || loading}>
                {SEND_ICON} Send
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
