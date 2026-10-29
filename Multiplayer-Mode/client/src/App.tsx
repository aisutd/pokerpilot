import { lazy, Suspense, useEffect, useRef, useState, type FormEvent } from 'react';
import type { EmojiClickData } from 'emoji-picker-react';
import './App.css';

const EmojiPicker = lazy(() => import('emoji-picker-react'));

interface ChatMessage {
  sender: string;
  text: string;
  timestamp: string;
}

const CHAT_WS_BASE_URL = import.meta.env.VITE_CHAT_WS_URL
  || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/chat-ws`;

function isChatMessage(message: unknown): message is ChatMessage {
  return typeof message === 'object'
    && message !== null
    && 'sender' in message
    && typeof message.sender === 'string'
    && 'text' in message
    && typeof message.text === 'string'
    && (!('timestamp' in message) || typeof message.timestamp === 'string');
}

function App() {
  const chatRoom = new URLSearchParams(window.location.search).get('room') || '';
  const [username, setUsername] = useState(() => localStorage.getItem('pokerpilot-chat-name') || '');
  const [isJoined, setIsJoined] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('Connecting…');
  const chatEndRef = useRef<HTMLDivElement>(null);
  const emojiPickerRef = useRef<HTMLDivElement>(null);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (event.target instanceof Node && !emojiPickerRef.current?.contains(event.target)) {
        setShowEmojiPicker(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    if (!isJoined) return undefined;

    let reconnectTimer: number | undefined;
    let shouldReconnect = true;

    const connect = () => {
      if (!/^[A-Za-z0-9_-]{20,64}$/.test(chatRoom)) {
        setConnectionStatus('Invalid game chat room');
        return;
      }
      setConnectionStatus('Connecting…');
      const socket = new WebSocket(`${CHAT_WS_BASE_URL}?room=${encodeURIComponent(chatRoom)}`);
      socketRef.current = socket;

      socket.onopen = () => setConnectionStatus('Connected');
      socket.onmessage = (event: MessageEvent<string>) => {
        try {
          const message: unknown = JSON.parse(event.data);
          if (isChatMessage(message)) {
            setConnectionStatus('Connected');
            setMessages((previous) => [...previous, { ...message, timestamp: message.timestamp || '' }].slice(-200));
          }
        } catch {
          setConnectionStatus('Received an invalid chat message');
        }
      };
      socket.onerror = () => setConnectionStatus('Connection issue');
      socket.onclose = () => {
        if (shouldReconnect) {
          setConnectionStatus('Reconnecting…');
          reconnectTimer = window.setTimeout(connect, 1500);
        }
      };
    };

    connect();
    return () => {
      shouldReconnect = false;
      window.clearTimeout(reconnectTimer);
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [chatRoom, isJoined]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleJoin = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const name = username.trim().slice(0, 24);
    if (!name) return;
    localStorage.setItem('pokerpilot-chat-name', name);
    setUsername(name);
    setIsJoined(true);
  };

  const handleEmojiClick = (emojiData: EmojiClickData) => {
    setInput((previous) => previous + emojiData.emoji);
  };

  const sendMessage = () => {
    const text = input.trim().slice(0, 500);
    const socket = socketRef.current;
    if (!text || !socket || socket.readyState !== WebSocket.OPEN) return;

    const message: ChatMessage = {
      sender: username,
      text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
    socket.send(JSON.stringify(message));
    setInput('');
    setShowEmojiPicker(false);
  };

  if (!isJoined) {
    return (
      <div className="join-screen">
        <form onSubmit={handleJoin} className="join-card">
          <h2>Welcome to Chat</h2>
          <p>Enter your name to message other players</p>
          <input
            type="text"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            placeholder="Enter your name…"
            maxLength={24}
            autoComplete="nickname"
            autoFocus
            required
          />
          <button type="submit">Join Chat</button>
        </form>
      </div>
    );
  }

  return (
    <main className="App">
      <header className="chat-header">
        <div className="header-title">Chat with fellow Pokermates</div>
        <div className="user-badge">
          <span className={`connection-dot ${connectionStatus === 'Connected' ? 'connected' : ''}`} aria-hidden="true" />
          <span className="connection-status" role="status">{connectionStatus}</span>
          <strong>{username}</strong>
        </div>
      </header>

      <div className="chat-container" role="log" aria-live="polite" aria-relevant="additions">
        {messages.map((message, index) => {
          const isSelf = message.sender === username;
          return (
            <div key={`${message.timestamp}-${index}`} className={`chat-message-wrapper ${isSelf ? 'self' : 'other'}`}>
              <span className="message-sender">{message.sender}</span>
              <div className="chat-message">{message.text}</div>
              {message.timestamp && <span className="message-time">{message.timestamp}</span>}
            </div>
          );
        })}
        <div ref={chatEndRef} />
      </div>

      {showEmojiPicker && (
        <div className="emoji-picker-container" ref={emojiPickerRef}>
          <Suspense fallback={<div className="emoji-picker-loading">Loading emojis…</div>}>
            <EmojiPicker onEmojiClick={handleEmojiClick} width={Math.min(320, window.innerWidth - 56)} height={360} />
          </Suspense>
        </div>
      )}

      <form
        className="input-container"
        onSubmit={(event) => {
          event.preventDefault();
          sendMessage();
        }}
      >
        <button
          type="button"
          className="emoji-toggle-btn"
          onClick={() => setShowEmojiPicker((previous) => !previous)}
          aria-label="Choose an emoji"
          aria-expanded={showEmojiPicker}
        >
          😊
        </button>
        <input
          type="text"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Type your message…"
          maxLength={500}
          autoComplete="off"
          aria-label="Message"
        />
        <button type="submit" className="send-btn" disabled={!input.trim() || connectionStatus !== 'Connected'}>
          Send
        </button>
      </form>
    </main>
  );
}

export default App;
