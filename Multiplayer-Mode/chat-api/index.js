const express = require("express");
const http = require("http");
const WebSocket = require("ws");
const cors = require("cors");

function createChatServer() {
  const app = express();
  app.use(cors());

  const server = http.createServer(app);
  const wss = new WebSocket.Server({ server, maxPayload: 4096 });
  const chatRooms = new Map();
  const MAX_NAME_LENGTH = 24;
  const MAX_MESSAGE_LENGTH = 500;

  wss.on("connection", (socket, request) => {
    const requestUrl = new URL(request.url, "http://localhost");
    const roomId = requestUrl.searchParams.get("room");
    const sender = (requestUrl.searchParams.get("name") || "Player")
      .trim()
      .slice(0, MAX_NAME_LENGTH) || "Player";
    if (!roomId || !/^[A-Za-z0-9_-]{20,64}$/.test(roomId)) {
      socket.close(1008, "A valid game chat room is required.");
      return;
    }

    let room = chatRooms.get(roomId);
    if (!room) {
      room = new Set();
      chatRooms.set(roomId, room);
    }
    room.add(socket);
    const joinNotice = JSON.stringify({
      type: "join",
      sender,
      text: `${sender} joined the chat`,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    });
    for (const client of room) {
      if (client !== socket && client.readyState === WebSocket.OPEN) client.send(joinNotice);
    }

    socket.on("close", () => {
      room.delete(socket);
      if (room.size === 0) chatRooms.delete(roomId);
    });

    socket.on("message", (rawMessage) => {
      let payload;
      try {
        payload = JSON.parse(rawMessage.toString());
      } catch {
        return;
      }

      if (
        typeof payload !== "object"
        || payload === null
        || typeof payload.text !== "string"
        || !payload.text.trim()
      ) {
        return;
      }

      const message = JSON.stringify({
        type: "message",
        sender,
        text: payload.text.trim().slice(0, MAX_MESSAGE_LENGTH),
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      });

      for (const client of room) {
        if (client.readyState === WebSocket.OPEN) client.send(message);
      }
    });
  });

  return { server, wss };
}

if (require.main === module) {
  const { server } = createChatServer();
  const port = Number(process.env.PORT) || 3000;
  server.listen(port, process.env.HOST || "0.0.0.0", () => {
    console.log(`PokerPilot chat server listening on port ${port}`);
  });
}

module.exports = { createChatServer };
