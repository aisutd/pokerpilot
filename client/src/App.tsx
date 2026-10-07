import { useState, useEffect, useRef } from 'react';
import EmojiPicker from 'emoji-picker-react';
import './App.css';

function App() {
  const [username, setUsername] = useState('');
  const [isJoined, setIsJoined] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [socket, setSocket] = useState(null);
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);
  const chatEndRef = useRef(null);
  const emojiPickerRef = useRef(null);

  // Close emoji picker when clicking outside of it
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (emojiPickerRef.current && !emojiPickerRef.current.contains(event.target)) {
        setShowEmojiPicker(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Initialize WebSocket connection after the user joins
  useEffect(() => {
    if (!isJoined) return;

    const newSocket = new WebSocket('ws://localhost:3000');
    setSocket(newSocket);

    newSocket.onopen = () => {
      console.log('WebSocket connection established');
    };

    newSocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setMessages((prevMessages) => [...prevMessages, data]);
      } catch (e) {
        // Fallback if the server sends raw text instead of JSON
        setMessages((prevMessages) => [
          ...prevMessages,
          { sender: 'Server', text: event.data.toString() }
        ]);
      }
    };

    newSocket.onclose = () => console.log('WebSocket connection closed');
    newSocket.onerror = (error) => console.error('WebSocket error:', error);

    return () => newSocket.close();
  }, [isJoined]);

  // Auto-scroll to the bottom when new messages arrive
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleJoin = (e) => {
    e.preventDefault();
    if (username.trim() !== '') {
      setIsJoined(true);
    }
  };

  const handleEmojiClick = (emojiData) => {
    setInput((prev) => prev + emojiData.emoji);
  };

  const sendMessage = () => {
    if (socket && input.trim() !== '' && socket.readyState === WebSocket.OPEN) {
      const messagePayload = {
        sender: username,
        text: input,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      // Send structured JSON message over WebSocket
      socket.send(JSON.stringify(messagePayload));
      setInput('');
      setShowEmojiPicker(false);
    }
  };

  // Step 1: Name Entry Screen
  if (!isJoined) {
    return (
      <div className="join-screen">
        <form onSubmit={handleJoin} className="join-card">
          <h2>Welcome to Chat</h2>
          <p>Enter your name to start messaging</p>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Enter your name..."
            autoFocus
            required
          />
          <button type="submit">Join Chat</button>
        </form>
      </div>
    );
  }

  // Step 2: Main Chat Room Screen
  return (
    <div className="App">
      <header className="chat-header">
        <div className="header-title">Chat with fellow Pokermates</div>
        <div className="user-badge">
          Logged in as <strong>{username}</strong>
        </div>
      </header>

      <div className="chat-container">
        {messages.map((msg, index) => {
          const isSelf = msg.sender === username;
          return (
            <div key={index} className={`chat-message-wrapper ${isSelf ? 'self' : 'other'}`}>
              <span className="message-sender">{msg.sender}</span>
              <div className="chat-message">{msg.text}</div>
              {msg.timestamp && <span className="message-time">{msg.timestamp}</span>}
            </div>
          );
        })}
        <div ref={chatEndRef} />
      </div>

      {showEmojiPicker && (
        <div className="emoji-picker-container" ref={emojiPickerRef}>
          <EmojiPicker onEmojiClick={handleEmojiClick} width={320} height={360} />
        </div>
      )}

      <div className="input-container">
        <button
          type="button"
          className="emoji-toggle-btn"
          onClick={() => setShowEmojiPicker((prev) => !prev)}
        >
          😊
        </button>

        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Type your message..."
        />

        <button type="button" className="send-btn" onClick={sendMessage}>
          Send
        </button>
      </div>
    </div>
  );
}

export default App;