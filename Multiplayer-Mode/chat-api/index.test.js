const assert = require("node:assert/strict");
const { once } = require("node:events");
const test = require("node:test");
const WebSocket = require("ws");
const { createChatServer } = require("./index");

test("broadcasts chat messages to connected clients with bounded fields", async (context) => {
  const { server, wss } = createChatServer();
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const { port } = server.address();
  const room = "this-is-a-private-game-room-id";
  const otherRoom = "this-is-another-private-room-id";
  const first = new WebSocket(`ws://127.0.0.1:${port}?room=${room}&name=${"A".repeat(30)}`);
  await once(first, "open");
  const joinNotice = once(first, "message");
  const second = new WebSocket(`ws://127.0.0.1:${port}?room=${room}&name=Bob`);
  let secondReceivedJoin = false;
  second.on("message", (message) => {
    if (JSON.parse(message.toString()).type === "join") secondReceivedJoin = true;
  });
  const isolated = new WebSocket(`ws://127.0.0.1:${port}?room=${otherRoom}`);
  let isolatedReceivedJoin = false;
  isolated.on("message", (message) => {
    if (JSON.parse(message.toString()).type === "join") isolatedReceivedJoin = true;
  });

  context.after(async () => {
    for (const socket of [first, second, isolated]) socket.terminate();
    await new Promise((resolve) => wss.close(resolve));
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  });

  await Promise.all([once(second, "open"), once(isolated, "open")]);
  const [joinPayload] = await joinNotice;
  const joinMessage = JSON.parse(joinPayload.toString());
  assert.equal(joinMessage.type, "join");
  assert.equal(joinMessage.sender, "Bob");
  assert.equal(joinMessage.text, "Bob joined the chat");
  assert.equal(typeof joinMessage.timestamp, "string");

  const firstMessage = once(first, "message");
  const secondMessage = once(second, "message");
  let isolatedMessageReceived = false;
  isolated.on("message", () => { isolatedMessageReceived = true; });
  first.send(JSON.stringify({ sender: "spoofed", text: ` ${"x".repeat(600)} ` }));

  const [fromFirst, fromSecond] = await Promise.all([firstMessage, secondMessage]);
  await new Promise((resolve) => setTimeout(resolve, 50));
  const payload = JSON.parse(fromFirst[0].toString());

  assert.deepEqual(JSON.parse(fromSecond[0].toString()), payload);
  assert.equal(secondReceivedJoin, false);
  assert.equal(isolatedReceivedJoin, false);
  assert.equal(isolatedMessageReceived, false);
  assert.equal(payload.type, "message");
  assert.equal(payload.sender, "A".repeat(24));
  assert.equal(payload.text, "x".repeat(500));
  assert.equal(typeof payload.timestamp, "string");
});
