import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Send, Scale, Paperclip, PlusCircle, MessageSquare } from 'lucide-react';

const API_URL = 'http://localhost:8000/api/query';
const UPLOAD_URL = 'http://localhost:8000/api/upload';
const SESSIONS_URL = 'http://localhost:8000/api/sessions';

function App() {
  const [sessions, setSessions] = useState([]);
  const [currentThreadId, setCurrentThreadId] = useState(crypto.randomUUID());
  
  const [messages, setMessages] = useState([
    { role: 'bot', content: 'Hello! I am your Legal RAG Assistant. Ask me anything about the Acme Corp NDAs or Employment Agreements. You can also upload your own PDF documents to ask questions about them!' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    try {
      const res = await axios.get(SESSIONS_URL);
      setSessions(res.data);
    } catch (e) {
      console.error("Failed to fetch sessions", e);
    }
  };

  const loadSession = async (threadId) => {
    try {
      setIsLoading(true);
      const res = await axios.get(`${SESSIONS_URL}/${threadId}`);
      if (res.data.history && res.data.history.length > 0) {
        setMessages(res.data.history);
      } else {
        setMessages([{ role: 'bot', content: 'History empty for this session.' }]);
      }
      setCurrentThreadId(threadId);
    } catch (e) {
      console.error("Failed to load session", e);
    } finally {
      setIsLoading(false);
    }
  };

  const startNewChat = () => {
    setCurrentThreadId(crypto.randomUUID());
    setMessages([
      { role: 'bot', content: 'Hello! I am your Legal RAG Assistant. Ask me anything about the Acme Corp NDAs or Employment Agreements. You can also upload your own PDF documents to ask questions about them!' }
    ]);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      await axios.post(UPLOAD_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setMessages(prev => [...prev, { role: 'bot', content: `Successfully uploaded and indexed: ${file.name}. You can now ask questions about it!` }]);
    } catch (error) {
      console.error("Upload error:", error);
      setMessages(prev => [...prev, { role: 'bot', content: `Failed to upload PDF: ${error.message}` }]);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post(API_URL, { 
        query: userMessage.content,
        thread_id: currentThreadId
      });
      const data = response.data;
      
      if (data.chat_history && data.chat_history.length > 0) {
        setMessages(data.chat_history);
      } else {
        // Fallback if history wasn't returned
        const botMessage = { 
          role: 'bot', 
          content: data.answer,
          intent: data.intent_used,
          sources: data.sources
        };
        setMessages(prev => [...prev, botMessage]);
      }
      
      // Refresh sidebar to show the new interaction
      fetchSessions();
    } catch (error) {
      console.error("Error fetching response:", error);
      setMessages(prev => [...prev, { 
        role: 'bot', 
        content: 'Sorry, I encountered an error. Is the backend server running?' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar for Sessions */}
      <aside className="sidebar">
        <button className="new-chat-btn" onClick={startNewChat}>
          <PlusCircle size={18} /> New Chat
        </button>
        <div style={{fontWeight: 600, color: 'var(--text-muted)', marginBottom: '10px', fontSize: '0.9rem'}}>Past Sessions</div>
        <div className="session-list">
          {sessions.map((session, idx) => (
            <div 
              key={idx} 
              className={`session-item ${session.thread_id === currentThreadId ? 'active' : ''}`}
              onClick={() => loadSession(session.thread_id)}
            >
              <MessageSquare size={14} className="inline-block mr-2" />
              {session.title}
            </div>
          ))}
          {sessions.length === 0 && <div style={{fontSize:'0.8rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '20px'}}>No past chats found</div>}
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="main-area">
        <header className="header">
          <h1><Scale className="inline-block mr-2" size={32} /> Legal RAG Assistant</h1>
          <p style={{ color: 'var(--text-muted)' }}>LangGraph Stateful Memory + LangSmith Tracing</p>
        </header>

        <div className="chat-container">
          <div className="messages">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.role}`}>
                {msg.intent && (
                  <div className="intent-badge">Routing Intent: {msg.intent}</div>
                )}
                <div className="content" style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources">
                    <div style={{fontWeight: 600, marginBottom: '5px'}}>Citations:</div>
                    {msg.sources.map((src, i) => (
                      <div key={i} className="source-item">
                        <div className="source-title">Document [{i+1}]: {src.title}</div>
                        <div style={{fontSize: '0.8rem', marginTop: '4px'}}>
                          "...{src.content.substring(0, 100)}..."
                        </div>
                        <div style={{fontSize: '0.7rem', color: '#64748b', marginTop: '4px'}}>
                          Rerank Score: {src.score?.toFixed(4)}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {isLoading && <div className="loading">Assistant is analyzing legal documents...</div>}
            {isUploading && <div className="loading">Indexing PDF into FAISS...</div>}
          </div>

          <form className="input-area" onSubmit={sendMessage}>
            <input
              type="file"
              accept=".pdf"
              ref={fileInputRef}
              onChange={handleFileUpload}
              style={{ display: 'none' }}
              id="pdf-upload"
            />
            <label htmlFor="pdf-upload" className="upload-btn" style={{cursor: 'pointer', padding: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'transparent', color: 'var(--primary-color)'}}>
              <Paperclip size={20} />
            </label>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question or upload a PDF..."
              disabled={isLoading || isUploading}
            />
            <button type="submit" disabled={isLoading || isUploading || !input.trim()}>
              <Send size={20} />
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;
