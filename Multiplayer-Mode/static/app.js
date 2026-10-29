'use strict';

const SUITS = { s: '♠', h: '♥', d: '♦', c: '♣' };
const RED_SUITS = new Set(['h', 'd']);
// Where each seat sits, clockwise from you at the bottom, for 2 to 10 seats.
// Tables over six seats use the compact "ring" layout (online.css).
const LAYOUTS = {
  2: ['bottom', 'top'],
  3: ['bottom', 'top-left', 'top-right'],
  4: ['bottom', 'left', 'top', 'right'],
  5: ['bottom', 'left', 'top-left', 'top-right', 'right'],
  6: ['bottom', 'left', 'top-left', 'top', 'top-right', 'right'],
  7: ['bottom', 'left', 'top-left', 'top-l', 'top-r', 'top-right', 'right'],
  8: ['bottom', 'bottom-left', 'left', 'top-left', 'top', 'top-right', 'right', 'bottom-right'],
  9: ['bottom', 'bottom-left', 'left', 'top-left', 'top-l', 'top-r', 'top-right', 'right', 'bottom-right'],
  10: ['bottom', 'bottom-left', 'left', 'top-left', 'top-l', 'top', 'top-r', 'top-right', 'right', 'bottom-right'],
};
const SESSION_KEY = 'pokerpilot-online';
const NAME_KEY = 'pokerpilot-name';
const RECONNECT_MS = [500, 1000, 2000, 4000];
const REQUEST_TIMEOUT_MS = 15000;
const LOADING_MIN_MS = 2000; // the loading screen stays up at least this long

const SETTINGS_KEY = 'pokerpilot-settings';
// Matches the phone layout in style.css.
const narrow = window.matchMedia('(max-width: 860px) and (orientation: portrait)');

const $ = (id) => document.getElementById(id);

let latest = null; // last state message from the server
let view = null; // latest.game
let busy = false; // an action is on its way to the server
let seatEls = null;
let socket = null;
let session = null; // { code, token }
let retries = 0;
let lanUrl = null;
let publicUrl = null; // ngrok link, works from anywhere
let clock = null; // { end, total } in performance.now() milliseconds

// ------------------------------------------------------------- helpers

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, Math.max(0, ms)));

function money(cents) {
  return `$${(cents / 100).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function moneyShort(cents) {
  const options = cents % 100 === 0
    ? { maximumFractionDigits: 0 }
    : { minimumFractionDigits: 2, maximumFractionDigits: 2 };
  return `$${(cents / 100).toLocaleString('en-US', options)}`;
}

// Chip sprite for an amount, by casino colours: white $1, red $5, blue $10,
// green $25, black $100, purple $500, yellow $1,000.
const CHIP_VALUES = [[100000, 'yellow'], [50000, 'purple'], [10000, 'black'], [2500, 'green'], [1000, 'blue'], [500, 'red']];

function chipColor(cents) {
  const found = CHIP_VALUES.find(([value]) => cents >= value);
  return found ? found[1] : 'white';
}

function prettyCard(code) {
  return `${code[0] === 'T' ? '10' : code[0]}${SUITS[code[1]]}`;
}

const capitalize = (text) => text[0].toUpperCase() + text.slice(1);

async function api(path, body) {
  const headers = {};
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    const csrf = document.cookie.split('; ').find((item) => item.startsWith('pokerpilot_csrf='));
    if (csrf) headers['X-CSRF-Token'] = decodeURIComponent(csrf.slice('pokerpilot_csrf='.length));
  }
  const options = body === undefined
    ? { method: 'GET' }
    : { method: 'POST', headers, body: JSON.stringify(body) };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let response;

  try {
    response = await fetch(path, { ...options, signal: controller.signal });
  } catch (error) {
    throw new Error(error.name === 'AbortError'
      ? 'The server took too long to respond.'
      : 'Could not reach the server. Is it still running?');
  } finally {
    clearTimeout(timer);
  }

  let data = null;

  try {
    data = await response.json();
  } catch {
    throw new Error('The server sent an unexpected response.');
  }

  if (!response.ok) {
    if (response.status === 401) {
      location.assign(`/auth?next=${encodeURIComponent(`${location.pathname}${location.search}`)}`);
    }
    throw new Error((data && (data.detail || data.error)) || 'Something went wrong.');
  }
  return data;
}

function showScreen(name) {
  for (const id of ['lobby', 'loading', 'game']) $(id).hidden = id !== name;
  if (name !== 'game') $('game-over').hidden = true;
}

function showPanel(name) {
  $('join-panel').hidden = name !== 'join';
  $('room-panel').hidden = name !== 'room';
}

function showError(message) {
  for (const id of ['game-error', 'lobby-error']) {
    $(id).hidden = !message;
    $(id).textContent = message || '';
  }
}

function saveSession(value) {
  session = value;
  try {
    if (value) sessionStorage.setItem(SESSION_KEY, JSON.stringify(value));
    else sessionStorage.removeItem(SESSION_KEY);
  } catch { /* private mode: the session just won't survive a reload */ }
}

function loadSession() {
  try {
    const value = JSON.parse(sessionStorage.getItem(SESSION_KEY));
    return value && value.code && value.token ? value : null;
  } catch {
    return null;
  }
}

// ------------------------------------------------------------- settings

const settings = { sound: true, hint: true, presets: true, stats: false };

try { Object.assign(settings, JSON.parse(localStorage.getItem(SETTINGS_KEY))); } catch { /* defaults */ }

function applySettings() {
  document.body.classList.toggle('show-stats', settings.stats);
  document.body.classList.toggle('hide-presets', !settings.presets);
  document.querySelectorAll('[data-setting]').forEach((box) => { box.checked = Boolean(settings[box.dataset.setting]); });
  $('sound-toggle').setAttribute('aria-pressed', String(settings.sound));
  $('sound-toggle').title = settings.sound ? 'Sound on' : 'Sound off';
}

function changeSetting(name, value) {
  settings[name] = value;
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch { /* ignore */ }
  applySettings();
  if (view) render(view);
}

// ---------------------------------------------------------------- sound

// Short synthesized effects: cards, chips, your turn, a win. Browsers only
// allow audio after the page is touched, so the context starts on the
// first click or key.
let audio = null;

function unlockAudio() {
  if (audio) return;
  try { audio = new AudioContext(); } catch { audio = null; }
}

document.addEventListener('pointerdown', unlockAudio, { once: true });
document.addEventListener('keydown', unlockAudio, { once: true });

function tone(freq, start, length, type = 'square', volume = .05) {
  const osc = audio.createOscillator();
  const gain = audio.createGain();
  const t = audio.currentTime + start;
  osc.type = type;
  osc.frequency.setValueAtTime(freq, t);
  gain.gain.setValueAtTime(volume, t);
  gain.gain.exponentialRampToValueAtTime(.0001, t + length);
  osc.connect(gain).connect(audio.destination);
  osc.start(t);
  osc.stop(t + length);
}

function swish(start) {
  const length = .07;
  const buffer = audio.createBuffer(1, Math.ceil(audio.sampleRate * length), audio.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / data.length);
  const source = audio.createBufferSource();
  const filter = audio.createBiquadFilter();
  const gain = audio.createGain();
  source.buffer = buffer;
  filter.type = 'bandpass';
  filter.frequency.value = 2800;
  gain.gain.value = .25;
  source.connect(filter).connect(gain).connect(audio.destination);
  source.start(audio.currentTime + start);
}

function playSound(kind) {
  if (!settings.sound || !audio) return;
  try {
    if (audio.state === 'suspended') audio.resume();
    if (kind === 'card') swish(0);
    else if (kind === 'chip') { tone(2300, 0, .05); tone(1750, .05, .06); }
    else if (kind === 'turn') { tone(660, 0, .14, 'sine', .08); tone(880, .12, .2, 'sine', .08); }
    else if (kind === 'win') [523, 659, 784, 1047].forEach((f, i) => tone(f, i * .1, .18, 'square', .04));
  } catch { /* no sound is fine */ }
}

let heard = null; // the last view we made sounds for

function soundsFor(v) {
  const before = heard;
  heard = v;
  if (!before || before.hand_number !== v.hand_number) {
    if (before && v.seats.some((seat) => seat.has_cards)) playSound('card');
    return;
  }
  if (v.board.length > before.board.length) playSound('card');
  else if (v.seats.some((seat, i) => seat.bet > (before.seats[i] ? before.seats[i].bet : 0))) playSound('chip');
  if (v.legal && !before.legal) playSound('turn');
  const won = v.result && v.result.winners.some((w) => w.seat === v.me);
  if (v.hand_over && !before.hand_over && won) playSound('win');
}

// ----------------------------------------------------------- connection

function connect() {
  if (!session) return;
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const params = new URLSearchParams({ code: session.code, token: session.token });
  const ws = new WebSocket(`${scheme}://${location.host}/ws?${params}`);
  socket = ws;

  ws.addEventListener('open', () => {
    retries = 0;
    showError(null);
  });

  ws.addEventListener('message', (event) => {
    if (socket !== ws) return;
    let message;
    try {
      message = JSON.parse(event.data);
    } catch {
      return;
    }
    handle(message);
  });

  ws.addEventListener('close', () => {
    if (socket !== ws) return; // replaced or closed on purpose
    socket = null;
    busy = false;
    if (!session) return;
    showError('Connection lost. Reconnecting…');
    const delay = RECONNECT_MS[Math.min(retries, RECONNECT_MS.length - 1)];
    retries += 1;
    setTimeout(() => { if (session && !socket) connect(); }, delay);
  });
}

function disconnect() {
  const ws = socket;
  socket = null;
  if (ws) ws.close();
}

function sendMessage(message) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    showError('Not connected yet. Try again in a moment.');
    return false;
  }
  socket.send(JSON.stringify(message));
  return true;
}

function handle(message) {
  if (message.type === 'state') {
    busy = false;
    latest = message;
    connectChat(message);
    if (message.room.status === 'lobby') {
      seatEls = null;
      showScreen('lobby');
      showPanel('room');
      renderRoom(message);
    } else {
      showScreen('game');
      render(message.game);
    }
  } else if (message.type === 'error') {
    busy = false;
    showError(message.message);
    if (view) render(view);
  } else if (message.type === 'gone' || message.type === 'left') {
    backToLobby(message.type === 'gone' ? message.message : null);
  }
}

// ----------------------------------------------------------- table chat

// The team's chat box (client/) runs in an iframe and talks to the chat-api
// relay itself; the room only hands each member its private chat id.
let chatRoom = null; // the chat id the frame is showing
let chatToggled = false; // the player minimized or expanded it themselves
let chatOpen = false; // at the table it shows only after the chat button

function showChat(status) {
  const atTable = status !== 'lobby';
  $('chat-widget').hidden = atTable && !chatOpen;
  $('chat-widget').classList.toggle('at-table', atTable);
  $('chat-toggle').setAttribute('aria-expanded', String(chatOpen));
  if (atTable) {
    if (chatOpen) setChatMinimized(false);
  } else if (!chatToggled) {
    setChatMinimized(true); // tucked away in the waiting room
  }
}

function setChatMinimized(minimized) {
  $('chat-widget-content').hidden = minimized;
  $('chat-minimize').setAttribute('aria-expanded', String(!minimized));
  $('chat-minimize').setAttribute('aria-label', minimized ? 'Expand table chat' : 'Minimize table chat');
  $('chat-minimize').textContent = minimized ? '+' : '−';
}

function connectChat(message) {
  const room = message.you.chat_room;
  if (!room) {
    disconnectChat();
    return;
  }

  $('chat-player-names').textContent = `Poker game: ${message.room.code}`;
  showChat(message.room.status);
  if (room === chatRoom) return;

  chatRoom = room;
  // The chat box asks for a name and remembers it here; offer the room name.
  try { localStorage.setItem('pokerpilot-chat-name', message.you.name.slice(0, 24)); } catch { /* ignore */ }
  const frame = $('chat-frame');
  frame.src = `${frame.dataset.src}?room=${encodeURIComponent(room)}`;
}

function disconnectChat() {
  chatRoom = null;
  chatToggled = false;
  chatOpen = false;
  $('chat-toggle').setAttribute('aria-expanded', 'false');
  $('chat-frame').src = 'about:blank';
  $('chat-widget').hidden = true;
}

function setupChatControls() {
  const widget = $('chat-widget');
  const handle = $('chat-drag-handle');
  let drag = null;

  handle.addEventListener('pointerdown', (event) => {
    if (event.button !== 0 || event.target.closest('button')) return;
    const rect = widget.getBoundingClientRect();
    widget.style.left = `${rect.left}px`;
    widget.style.top = `${rect.top}px`;
    widget.style.right = 'auto';
    widget.style.bottom = 'auto';
    drag = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, left: rect.left, top: rect.top };
    handle.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  handle.addEventListener('pointermove', (event) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    const left = Math.max(0, Math.min(window.innerWidth - widget.offsetWidth, drag.left + event.clientX - drag.x));
    const top = Math.max(0, Math.min(window.innerHeight - widget.offsetHeight, drag.top + event.clientY - drag.y));
    widget.style.left = `${left}px`;
    widget.style.top = `${top}px`;
  });

  const stopDragging = (event) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
    drag = null;
  };
  handle.addEventListener('pointerup', stopDragging);
  handle.addEventListener('pointercancel', stopDragging);

  $('chat-minimize').addEventListener('click', () => {
    if (latest && latest.room.status !== 'lobby') {
      chatOpen = false; // at the table, minimizing closes it; the chat button opens it
      showChat(latest.room.status);
      return;
    }
    chatToggled = true;
    setChatMinimized(!$('chat-widget-content').hidden);
  });

  $('chat-toggle').addEventListener('click', () => {
    if (!latest || !chatRoom) return;
    chatOpen = !chatOpen;
    showChat(latest.room.status);
  });

  window.addEventListener('resize', () => {
    if (!widget.style.left) return;
    const left = Math.max(0, Math.min(window.innerWidth - widget.offsetWidth, widget.offsetLeft));
    const top = Math.max(0, Math.min(window.innerHeight - widget.offsetHeight, widget.offsetTop));
    widget.style.left = `${left}px`;
    widget.style.top = `${top}px`;
  });
}

setupChatControls();

function backToLobby(error) {
  saveSession(null);
  disconnect();
  disconnectChat();
  latest = null;
  view = null;
  seatEls = null;
  busy = false;
  heard = null;
  toggleHistory(false);
  closePanels();
  setTray(false);
  showScreen('lobby');
  showPanel('join');
  showError(error);
}

function leaveRoom() {
  if (!sendMessage({ type: 'leave' })) backToLobby(null);
}

// ------------------------------------------------------------ the lobby

function playerName() {
  const name = $('player-name').value.trim();
  if (!name) {
    showError('Enter your name first.');
    $('player-name').focus();
    return null;
  }
  try { localStorage.setItem(NAME_KEY, name); } catch { /* ignore */ }
  return name;
}

async function enterRoom(path, body) {
  showError(null);
  for (const id of ['create-room', 'join-room']) $(id).disabled = true;
  // Show the loading screen right away and keep it up long enough to see.
  $('loading-text').textContent = path === '/api/rooms' ? 'CREATING ROOM' : 'JOINING ROOM';
  showScreen('loading');
  const shown = Date.now();
  try {
    const { code, token } = await api(path, body);
    saveSession({ code, token });
    history.replaceState(null, '', `/?room=${code}`);
    await sleep(LOADING_MIN_MS - (Date.now() - shown));
    connect();
  } catch (error) {
    showScreen('lobby');
    showPanel('join');
    showError(error.message);
  } finally {
    for (const id of ['create-room', 'join-room']) $(id).disabled = false;
  }
}

function inviteLink(code) {
  const local = location.hostname.match(/^(localhost|127\.)/);
  return `${publicUrl || (lanUrl && local ? lanUrl : location.origin)}/?room=${code}`;
}

function renderRoom(message) {
  const { room, you } = message;
  $('room-code').textContent = room.code;

  const link = inviteLink(room.code);
  $('share-text').replaceChildren(
    'Friends on your network can open ',
    Object.assign(document.createElement('b'), { textContent: link }),
    ' or enter the code. Testing alone? Open that link in another browser tab.',
  );

  const items = room.seats.map((member) => {
    const li = document.createElement('li');
    const face = document.createElement('span');
    face.className = 'face';
    const who = document.createElement('span');
    who.className = 'seat-who';

    if (!member) {
      li.className = 'empty';
      who.textContent = room.mode === 'pve' ? `AI player (${room.difficulty})` : 'Open seat';
      li.append(who);
      return li;
    }

    face.dataset.portrait = member.portrait;
    who.textContent = member.name;
    li.classList.toggle('you', member.you);
    li.classList.toggle('away', member.away);
    li.append(face, who);
    const tag = member.host ? 'Host' : member.away ? 'Away' : member.you ? 'You' : '';
    if (tag) li.append(Object.assign(document.createElement('small'), { textContent: tag }));
    return li;
  });
  $('room-seats').replaceChildren(...items);

  $('room-waiting').hidden = !room.waiting.length;
  $('room-waiting').textContent = room.waiting.length
    ? `Waiting for a seat: ${room.waiting.map((p) => p.name).join(', ')}`
    : '';

  document.querySelectorAll('#room-panel [data-mode]').forEach((b) => {
    b.setAttribute('aria-checked', String(b.dataset.mode === room.mode));
    b.disabled = !you.host;
  });
  document.querySelectorAll('#room-panel [data-difficulty]').forEach((b) => {
    b.setAttribute('aria-checked', String(b.dataset.difficulty === room.difficulty));
    b.disabled = !you.host;
  });
  $('difficulty-seg').hidden = room.mode !== 'pve';
  renderStepper($('room-seat-stepper'), room, you, true);

  const seated = room.seats.filter(Boolean).length;
  const canStart = room.mode === 'pve' || seated >= 2;
  $('start-game').hidden = !you.host;
  $('start-game').disabled = !canStart;
  if (!you.host) $('start-hint').textContent = `Waiting for ${room.host} to start the game…`;
  else if (!canStart) $('start-hint').textContent = 'PvP needs at least two players. Share the code, or switch to PvE.';
  else $('start-hint').textContent = '';
}

function renderStepper(el, room, you, canRemove) {
  // PvP only: the host opens or closes seats; removing needs an empty one.
  const count = room.seats.length;
  const open = room.seats.filter((member) => !member).length;
  el.hidden = room.mode !== 'pvp' || (!you.host && el.id === 'game-seat-stepper');
  el.querySelector('.seat-count').textContent = count;
  el.querySelector('[data-seats="-1"]').disabled = !you.host || !canRemove || !open || count <= room.min_seats;
  el.querySelector('[data-seats="1"]').disabled = !you.host || count >= room.max_seats;
  el.querySelector('[data-seats="-1"]').title = !open && count > room.min_seats
    ? 'Only empty seats can be removed'
    : canRemove ? '' : 'Seats can be removed between hands';
}

// --------------------------------------------------------------- cards

function suitGlyph(suit) {
  return Object.assign(document.createElement('i'), { textContent: SUITS[suit] });
}

// Card faces come from the team's deck (Assets/Cards/deck.svg, served at
// /cards/deck.svg); backs are the gold ones from the table mock-up (style.css).
function cardElement(code, mini) {
  const el = document.createElement('div');
  el.className = mini ? 'card mini' : 'card';

  if (!code) {
    el.classList.add('back');
    el.setAttribute('aria-label', 'Face-down card');
    return el;
  }

  if (code === 'slot' || !/^[2-9TJQKA][shdc]$/.test(code)) {
    el.classList.add('slot');
    return el;
  }

  if (RED_SUITS.has(code[1])) el.classList.add('red');
  el.setAttribute('aria-label', prettyCard(code));
  appendCardArt(el, `card-${code}`);

  // Cards beside players on phones are too small for the deck's corner, so
  // they also carry a plain index; style.css shows one or the other.
  if (mini) {
    const corner = document.createElement('span');
    corner.className = 'corner';
    corner.setAttribute('aria-hidden', 'true');
    corner.append(code[0] === 'T' ? '10' : code[0], suitGlyph(code[1]));
    el.appendChild(corner);
  }
  return el;
}

function appendCardArt(container, symbol) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'card-face-svg');
  svg.setAttribute('viewBox', '0 0 250 350');
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  const href = `/cards/deck.svg#${symbol}`;
  use.setAttribute('href', href);
  use.setAttribute('width', '250');
  use.setAttribute('height', '350');
  use.setAttributeNS('http://www.w3.org/1999/xlink', 'xlink:href', href);
  svg.append(use);
  container.append(svg);
}

function renderCards(container, codes, mini = false) {
  // Rebuild only when the cards change, so the deal animation plays once.
  const key = JSON.stringify(codes);
  if (container.dataset.key === key) return;
  container.dataset.key = key;
  container.replaceChildren(...codes.map((code) => cardElement(code, mini)));
}

// --------------------------------------------------------------- seats

function buildSeats(seats) {
  const root = $('seats');
  root.replaceChildren();
  const layout = LAYOUTS[seats.length];
  $('table-area').classList.toggle('ring', seats.length > 6);
  seatEls = seats.map((seat, i) => {
    const pos = layout[i];
    const el = document.createElement('div');
    el.className = 'seat';
    el.dataset.pos = pos;

    const plate = document.createElement('div');
    plate.className = 'plate';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';

    const who = document.createElement('div');
    who.className = 'who';
    const name = document.createElement('div');
    name.className = 'seat-name';
    const stack = document.createElement('div');
    stack.className = 'seat-stack';
    const stats = document.createElement('div');
    stats.className = 'seat-stats';
    stats.title = 'Hands played voluntarily % / Preflop raise % / Hands won';
    who.append(name, stack, stats);

    const dealer = document.createElement('span');
    dealer.className = 'dealer';
    dealer.textContent = 'D';
    dealer.title = 'Dealer button';

    const action = document.createElement('span');
    action.className = 'seat-action';

    const bar = document.createElement('span');
    bar.className = 'clock';
    bar.hidden = true;

    plate.append(avatar, who, dealer, action, bar);

    const cards = document.createElement('div');
    cards.className = i === 0 ? 'my-cards' : 'seat-cards';

    el.append(cards, plate);
    root.appendChild(el);

    const bet = document.createElement('div');
    bet.className = 'bet';
    bet.dataset.pos = pos;
    root.appendChild(bet);

    return { el, avatar, name, stack, stats, dealer, action, bar, cards, bet };
  });
}

function seatLabel(seat) {
  if (seat.last_action) return seat.last_action;
  if (seat.status === 'all-in') return 'All-in';
  if (seat.away && seat.status !== 'empty') return 'Away';
  if (seat.status === 'waiting') return 'Next hand';
  return null;
}

function renderSeats(v, thinkingSeat) {
  if (!seatEls || seatEls.length !== v.seats.length) buildSeats(v.seats);
  const winners = new Set((v.result ? v.result.winners : []).map((w) => w.seat));

  v.seats.forEach((seat, i) => {
    const parts = seatEls[i];
    const mine = i === v.me;
    const empty = seat.status === 'empty';

    // Your plate is set apart by where it sits; in the narrow layout it says so.
    parts.name.textContent = mine && narrow.matches ? `${seat.name} (you)` : seat.name;
    // The portrait sets the face and the player's colour (style.css).
    if (seat.portrait) parts.el.dataset.portrait = seat.portrait;
    else delete parts.el.dataset.portrait;
    parts.avatar.className = seat.portrait || empty ? 'avatar' : 'avatar initial';
    parts.avatar.textContent = seat.portrait || empty ? '' : seat.name[0].toUpperCase();

    parts.stack.textContent = empty ? 'Empty' : seat.status === 'out' ? 'Out' : moneyShort(seat.stack);
    parts.stats.textContent = seat.stats.join(' / ');
    parts.dealer.hidden = !seat.is_button;

    const el = parts.el;
    el.classList.toggle('active', v.to_act === i && !v.hand_over);
    el.classList.toggle('thinking', thinkingSeat === i);
    el.classList.toggle('folded', seat.status === 'folded');
    el.classList.toggle('out', seat.status === 'out' || seat.status === 'waiting');
    el.classList.toggle('empty', empty);
    el.classList.toggle('winner', v.hand_over && winners.has(i));

    const label = seatLabel(seat);
    if (parts.action.textContent !== (label || '')) {
      parts.action.textContent = label || '';
      parts.action.className = 'seat-action';
      if (label) parts.action.classList.add(`is-${label.split(' ')[0].toLowerCase()}`);
    }
    parts.action.hidden = !label;

    let cards = [];
    if (seat.cards) cards = seat.cards;
    else if (seat.has_cards) cards = [null, null];
    renderCards(parts.cards, cards, i !== 0);
    parts.cards.classList.toggle('up', Boolean(seat.cards) && i !== 0);

    parts.bet.hidden = !seat.bet;
    parts.bet.textContent = seat.bet ? moneyShort(seat.bet) : '';
    parts.bet.dataset.chip = chipColor(seat.bet);
  });
}

// ------------------------------------------------------------ rendering

function render(v) {
  view = v;
  const you = latest.you;
  const thinking = v.to_act !== null && v.to_act !== v.me ? v.to_act : null;

  $('room-tag').textContent = latest.room.code;
  $('hand-number').textContent = `Hand #${v.hand_number}`;
  const mode = v.mode === 'pve' ? `AI: ${capitalize(v.difficulty)}` : 'PvP';
  $('street').textContent = `${v.paused ? 'Paused' : v.hand_over ? 'Hand over' : v.street} · ${mode}`;

  renderSeats(v, thinking);
  renderStepper($('game-seat-stepper'), latest.room, you, v.hand_over);

  const board = [...v.board];
  while (board.length < 5) board.push('slot');
  renderCards($('board'), board);
  $('pot').textContent = v.pot ? `Pot: ${moneyShort(v.pot)}` : '';
  renderPotChips(v.pot);
  $('hand-strength').textContent = handHint(v);
  soundsFor(v);

  clock = v.clock !== null ? { end: performance.now() + v.clock * 1000, total: v.clock_total * 1000 } : null;

  renderResult(v);
  renderLog(v);
  renderControls(v, you, thinking);
  tick();

  $('game-over').hidden = !v.paused;
  if (v.paused) {
    const mine = v.champion && v.champion === you.name;
    $('game-over-title').textContent = v.champion
      ? (mine ? 'You have all the chips!' : `${v.champion} has all the chips!`)
      : 'Waiting for players';
    $('game-over-text').textContent = you.host
      ? 'Reset everyone to $200 to play on, or wait for someone to rebuy or join.'
      : 'Waiting for the host to reset the chips, or for someone to rebuy or join.';
    $('modal-rebuy').hidden = !you.can_rebuy;
    $('restart').hidden = !you.host;
  }
}

// The pile over the pot: more chips, and pricier colours, as it grows.
function renderPotChips(pot) {
  const pile = $('pot-chips');
  const colors = !pot ? []
    : [chipColor(pot), 'red', 'white', 'black', 'red'].slice(0, Math.min(5, 2 + Math.floor(pot / 1000)));
  const key = colors.join();
  if (pile.dataset.key === key) return;
  pile.dataset.key = key;
  pile.replaceChildren(...colors.map((color) => Object.assign(document.createElement('img'), {
    src: `/static/art/chips/${color}.png`, alt: '',
  })));
}

function handHint(v) {
  return settings.hint && v.my_hand ? `You have ${v.my_hand}` : '';
}

function renderResult(v) {
  const toast = $('result-toast');
  const result = v.result;

  if (!v.hand_over || !result || v.paused) {
    toast.hidden = true;
    return;
  }

  toast.hidden = false;
  toast.replaceChildren(result.headline);
  const top = result.winners[0];
  let detail = '';

  if (top && top.best) detail = top.best.map(prettyCard).join(' ');
  else if (!result.showdown) detail = 'Everyone else folded.';

  if (detail) {
    const small = document.createElement('small');
    small.textContent = detail;
    toast.appendChild(small);
  }
}

function renderLog(v) {
  const list = $('log');
  const me = latest.you.name;
  const items = v.log.map((line) => {
    const li = document.createElement('li');
    li.textContent = line;
    if (line.startsWith('---')) li.className = 'street';
    else if (me && line.startsWith(`${me} `)) li.className = 'you';
    else if (/collects |is out of chips/.test(line)) li.className = 'end';
    return li;
  });
  list.replaceChildren(...items);
  list.scrollTop = list.scrollHeight;
}

function renderControls(v, you, thinking) {
  const legal = !busy ? v.legal : null;
  $('actions').hidden = !legal;
  if (!legal) setTray(false);

  const between = v.hand_over && !v.paused;
  $('hand-done').hidden = !(between || (you.can_rebuy && !v.paused));
  $('rebuy').hidden = !you.can_rebuy;
  $('next-in').hidden = !between;

  const waiting = $('waiting');
  waiting.hidden = Boolean(legal) || v.hand_over;
  if (!waiting.hidden) {
    const me = v.me !== null ? v.seats[v.me] : null;
    if (busy) waiting.textContent = 'Sending…';
    else if (!you.seated) waiting.textContent = 'You are watching. You will get a seat when one opens.';
    else if (thinking !== null) waiting.textContent = `${v.seats[thinking].name} is thinking…`;
    else if (me && me.status === 'waiting') waiting.textContent = 'You join next hand.';
    else waiting.textContent = 'Dealing…';
  }

  if (!legal) return;

  // Check and call are separate buttons, as in the mock-up.
  const me = v.seats[v.me];
  $('btn-fold').disabled = !legal.fold;
  $('btn-check').disabled = !legal.check;
  $('btn-call').disabled = legal.check;
  $('call-label').textContent = !legal.check && legal.call >= me.stack ? 'All-in' : 'Call';
  $('call-amount').textContent = legal.check ? '' : moneyShort(legal.call);

  const raise = legal.raise;
  for (const id of ['btn-raise', 'raise-slider', 'raise-input']) $(id).disabled = !raise;

  for (const b of document.querySelectorAll('[data-preset]')) {
    b.disabled = !raise;
    b.textContent = presetName(b.dataset.preset);
    b.title = raise ? moneyShort(presetAmount(b.dataset.preset)) : '';
  }

  if (!raise) {
    setTray(false);
    $('raise-input').value = '';
    $('raise-slider').style.setProperty('--fill', '0%');
    return;
  }

  const slider = $('raise-slider');
  const changed = slider.min !== String(raise.min) || slider.max !== String(raise.max);
  slider.min = raise.min;
  slider.max = raise.max;
  slider.step = 1;

  setRaise(changed ? raise.min : Number(slider.value));
}

// BET / RAISE opens the sizing tray; pressing it again makes the bet.
function trayOpen() {
  return !$('raise-tray').hidden;
}

function setTray(open) {
  $('raise-tray').hidden = !open;
  $('btn-raise').setAttribute('aria-expanded', String(open));
  const raise = view && view.legal && view.legal.raise;
  if (raise) updateRaiseUi(Number($('raise-slider').value));
  else $('raise-label').textContent = 'Bet / Raise';
}

// ---------------------------------------------------------------- clock

function tick() {
  if (!view || !seatEls) return;
  const left = clock ? Math.max(0, clock.end - performance.now()) : 0;
  const seconds = Math.ceil(left / 1000);

  seatEls.forEach((parts, i) => {
    const on = Boolean(clock) && view.to_act === i && !view.hand_over;
    parts.bar.hidden = !on;
    if (on) {
      parts.bar.style.transform = `scaleX(${clock.total ? left / clock.total : 0})`;
      parts.bar.classList.toggle('low', left < 8000);
    }
  });

  if (view.hand_over && !view.paused) {
    $('next-in').textContent = clock ? `Next hand in ${seconds}s` : 'Next hand soon…';
  }

  const strength = $('hand-strength');
  const myTurn = view.legal && !busy && clock;
  strength.classList.toggle('your-turn', Boolean(myTurn));
  const base = handHint(view);
  strength.textContent = myTurn ? `Your turn · ${seconds}s${base ? ` · ${base}` : ''}` : base;
}

setInterval(tick, 200);

// -------------------------------------------------------------- sizing

function presetName(preset) {
  return { min: 'Min', 0.5: '½ Pot', 0.75: '¾ Pot', 1: 'Pot', max: 'All-in' }[preset];
}

function clampRaise(cents) {
  const raise = view && view.legal && view.legal.raise;
  if (!raise) return null;
  if (!Number.isFinite(cents)) return raise.min;
  return Math.min(raise.max, Math.max(raise.min, Math.round(cents)));
}

function presetAmount(preset) {
  const raise = view.legal.raise;
  if (preset === 'min') return raise.min;
  if (preset === 'max') return raise.max;

  const toCall = view.legal.call;
  const potAfterCall = view.pot + toCall;
  return clampRaise(view.seats[view.me].bet + toCall + Number(preset) * potAfterCall);
}

function setRaise(cents) {
  const amount = clampRaise(cents);
  if (amount === null) return;
  const slider = $('raise-slider');
  slider.value = amount;
  $('raise-input').value = (amount / 100).toFixed(2);
  updateRaiseUi(amount);
}

function updateRaiseUi(amount) {
  const raise = view.legal.raise;
  const span = raise.max - raise.min;
  const fill = span > 0 ? ((amount - raise.min) / span) * 100 : 100;
  $('raise-slider').style.setProperty('--fill', `${fill}%`);
  const verb = amount === raise.max ? 'All-in' : raise.is_bet ? 'Bet' : 'Raise to';
  $('raise-label').textContent = trayOpen() ? `${verb} ${moneyShort(amount)}` : 'Bet / Raise';
}

function inputCents() {
  const text = $('raise-input').value.replace(/[$,\s]/g, '');
  const dollars = Number(text);
  return text === '' || !Number.isFinite(dollars) ? NaN : Math.round(dollars * 100);
}

// -------------------------------------------------------------- events

function act(action) {
  if (busy || !view || !view.legal || !$('actions').offsetParent) return;
  const legal = view.legal;
  let message = { type: 'action', action };

  if (action === 'fold' && !legal.fold) return;
  if (action === 'check' && !legal.check) return;
  if (action === 'call' && legal.check) return;
  if (action === 'check' || action === 'call') message.action = 'check_call';

  if (action === 'raise') {
    if (!legal.raise) return;
    if (!trayOpen()) {
      setTray(true); // first pick the size
      return;
    }
    message = { type: 'action', action: 'raise', amount: clampRaise(inputCents()) };
  }

  showError(null);
  if (sendMessage(message)) {
    busy = true;
    setTray(false);
    render(view);
  }
}

$('create-room').addEventListener('click', () => {
  const name = playerName();
  if (name) enterRoom('/api/rooms', { name });
});

function joinFromForm() {
  const name = playerName();
  const code = $('join-code').value.trim().toUpperCase();
  if (!name) return;
  if (code.length !== 5) {
    showError('Room codes are five letters.');
    $('join-code').focus();
    return;
  }
  enterRoom('/api/rooms/join', { name, code });
}

$('join-room').addEventListener('click', joinFromForm);
$('join-code').addEventListener('keydown', (e) => { if (e.key === 'Enter') joinFromForm(); });
$('player-name').addEventListener('keydown', (e) => {
  if (e.key !== 'Enter') return;
  if ($('join-code').value.trim()) joinFromForm();
  else $('create-room').click();
});

async function copyInvite(button) {
  const link = inviteLink(latest.room.code);
  try {
    await navigator.clipboard.writeText(link);
    button.textContent = 'Copied!';
    setTimeout(() => { button.textContent = 'Copy invite link'; }, 1500);
  } catch {
    prompt('Copy this link:', link); // clipboard needs https or localhost
  }
}

$('copy-link').addEventListener('click', () => copyInvite($('copy-link')));

document.querySelectorAll('#room-panel [data-mode]').forEach((button) => {
  button.addEventListener('click', () => {
    sendMessage({ type: 'settings', mode: button.dataset.mode, difficulty: latest.room.difficulty });
  });
});

document.querySelectorAll('#room-panel [data-difficulty]').forEach((button) => {
  button.addEventListener('click', () => {
    sendMessage({ type: 'settings', mode: latest.room.mode, difficulty: button.dataset.difficulty });
  });
});

document.querySelectorAll('[data-seats]').forEach((button) => {
  button.addEventListener('click', () => {
    sendMessage({ type: 'seats', count: latest.room.seats.length + Number(button.dataset.seats) });
  });
});

$('start-game').addEventListener('click', () => sendMessage({ type: 'start' }));
$('leave-room').addEventListener('click', leaveRoom);

$('btn-check').addEventListener('click', () => act('check'));
$('btn-fold').addEventListener('click', () => act('fold'));
$('btn-call').addEventListener('click', () => act('call'));
$('btn-raise').addEventListener('click', () => act('raise'));
$('rebuy').addEventListener('click', () => sendMessage({ type: 'rebuy' }));
$('modal-rebuy').addEventListener('click', () => sendMessage({ type: 'rebuy' }));
$('restart').addEventListener('click', () => sendMessage({ type: 'restart' }));
$('to-lobby').addEventListener('click', leaveRoom);
$('logout').addEventListener('click', async () => {
  try {
    await api('/api/auth/logout', {});
    saveSession(null);
    disconnect();
    disconnectChat();
    location.assign('/auth');
  } catch (error) {
    showError(error.message);
  }
});

$('raise-slider').addEventListener('input', (e) => {
  const amount = Number(e.target.value);
  $('raise-input').value = (amount / 100).toFixed(2);
  updateRaiseUi(amount);
});

$('raise-input').addEventListener('input', () => {
  const amount = clampRaise(inputCents());
  if (amount !== null && Number.isFinite(inputCents())) {
    $('raise-slider').value = amount;
    updateRaiseUi(amount);
  }
});

$('raise-input').addEventListener('change', () => setRaise(inputCents()));

$('raise-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    setRaise(inputCents());
    act('raise');
  }
});

document.querySelectorAll('[data-preset]').forEach((button) => {
  button.addEventListener('click', () => setRaise(presetAmount(button.dataset.preset)));
});

function toggleHistory(open) {
  $('history').hidden = !open;
  $('history-toggle').setAttribute('aria-expanded', String(open));
}

$('history-toggle').addEventListener('click', () => {
  closePanels();
  toggleHistory($('history').hidden);
});
$('history-close').addEventListener('click', () => toggleHistory(false));

// MENU and SETTINGS open small panels over the table.
const PANELS = { 'menu-toggle': 'menu-panel', 'settings-toggle': 'settings-panel' };

function closePanels() {
  for (const [button, panel] of Object.entries(PANELS)) {
    $(panel).hidden = true;
    $(button).setAttribute('aria-expanded', 'false');
  }
}

for (const [button, panel] of Object.entries(PANELS)) {
  $(button).addEventListener('click', () => {
    const open = $(panel).hidden;
    closePanels();
    $(panel).hidden = !open;
    $(button).setAttribute('aria-expanded', String(open));
  });
}

document.addEventListener('pointerdown', (e) => {
  if (!e.target.closest('.popover, .hud-btn')) closePanels();
});

document.querySelectorAll('[data-setting]').forEach((box) => {
  box.addEventListener('change', () => changeSetting(box.dataset.setting, box.checked));
});

$('sound-toggle').addEventListener('click', () => {
  changeSetting('sound', !settings.sound);
  playSound('chip');
});

$('copy-invite').addEventListener('click', () => copyInvite($('copy-invite')));

$('quit').addEventListener('click', () => {
  const me = view && view.me !== null ? view.seats[view.me] : null;
  const inHand = me && view && !view.hand_over && me.has_cards;
  if (inHand && !confirm('Leave the room? Your hand will be folded.')) return;
  leaveRoom();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    toggleHistory(false);
    closePanels();
    setTray(false);
  }
  if (e.target.tagName === 'INPUT' || e.ctrlKey || e.metaKey || e.altKey) return;
  if ($('game').hidden) return;

  const key = e.key.toLowerCase();
  if (key === 'f') act('fold');
  else if (key === 'c') act(view && view.legal && view.legal.check ? 'check' : 'call');
  else if (key === 'r') act('raise');
});

// ---------------------------------------------------------------- start

applySettings();
narrow.addEventListener('change', () => { if (view) render(view); });

const invited = new URLSearchParams(location.search).get('room');
if (invited) $('join-code').value = invited.toUpperCase().slice(0, 5);

async function initializeOnlineApp() {
  try {
    const { user } = await api('/api/auth/me');
    $('account-name').textContent = user.username;
    try { $('player-name').value = localStorage.getItem(NAME_KEY) || user.username; }
    catch { $('player-name').value = user.username; }
    const info = await api('/api/info');
    lanUrl = info.lan_url;
    publicUrl = info.public_url;
    // The tab-local room token cannot authenticate a user by itself; a
    // valid account session is also checked on each WebSocket message.
    session = loadSession();
    if (session) {
      showScreen('loading');
      connect();
    }
  } catch (error) {
    showError(error.message);
  }
}

initializeOnlineApp();
