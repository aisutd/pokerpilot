'use strict';

const SUITS = { s: '♠', h: '♥', d: '♦', c: '♣' };
const RED_SUITS = new Set(['h', 'd']);
const POSITIONS = ['bottom', 'left', 'top-left', 'top', 'top-right', 'right'];
const HUMAN = 0;
const LOADING_MIN_MS = 2600;
const FRAME_MS = 650;
const FAST_FRAME_MS = 320;
const REQUEST_TIMEOUT_MS = 30000;

const $ = (id) => document.getElementById(id);

let view = null;
let busy = false;
let epoch = 0; // bumped when leaving the table, to cancel animations
let seatEls = null;

// ------------------------------------------------------------- helpers

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function money(cents) {
  return `$${(cents / 100).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function moneyShort(cents) {
  const options = cents % 100 === 0
    ? { maximumFractionDigits: 0 }
    : { minimumFractionDigits: 2, maximumFractionDigits: 2 };
  return `$${(cents / 100).toLocaleString('en-US', options)}`;
}

function prettyCard(code) {
  return `${code[0] === 'T' ? '10' : code[0]}${SUITS[code[1]]}`;
}

async function api(path, body) {
  const options = body === undefined
    ? { method: 'GET' }
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
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

  if (!response.ok) throw new Error((data && data.error) || 'Something went wrong.');
  return data;
}

function showScreen(name) {
  for (const id of ['lobby', 'loading', 'game']) $(id).hidden = id !== name;
  if (name !== 'game') $('game-over').hidden = true;
}

function showError(message) {
  for (const id of ['game-error', 'lobby-error']) {
    $(id).hidden = !message;
    $(id).textContent = message || '';
  }
}

// --------------------------------------------------------------- cards

function cardElement(code) {
  const el = document.createElement('div');
  el.className = 'card';

  if (!code) {
    el.classList.add('back');
    el.setAttribute('aria-label', 'Face-down card');
    return el;
  }

  if (code === 'slot') {
    el.classList.add('slot');
    return el;
  }

  if (RED_SUITS.has(code[1])) el.classList.add('red');
  el.setAttribute('aria-label', prettyCard(code));

  const rank = document.createElement('span');
  rank.className = 'rank';
  rank.textContent = code[0] === 'T' ? '10' : code[0];
  const suit = document.createElement('span');
  suit.className = 'suit';
  suit.textContent = SUITS[code[1]];
  el.append(rank, suit);
  return el;
}

function renderCards(container, codes) {
  // Rebuild only when the cards change, so the deal animation plays once.
  const key = JSON.stringify(codes);
  if (container.dataset.key === key) return;
  container.dataset.key = key;
  container.replaceChildren(...codes.map(cardElement));
}

// --------------------------------------------------------------- seats

function buildSeats(seats) {
  const root = $('seats');
  root.replaceChildren();
  seatEls = seats.map((seat, i) => {
    const pos = POSITIONS[i];
    const el = document.createElement('div');
    el.className = 'seat';
    el.dataset.pos = pos;
    el.dataset.color = seat.color;

    const plate = document.createElement('div');
    plate.className = 'plate';

    const avatar = document.createElement('div');
    avatar.className = i === HUMAN ? 'avatar you' : 'avatar';
    if (seat.avatar) avatar.textContent = seat.avatar;

    const who = document.createElement('div');
    who.className = 'who';
    const name = document.createElement('div');
    name.className = 'seat-name';
    name.textContent = seat.name;
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

    plate.append(avatar, who, dealer, action);

    const cards = document.createElement('div');
    cards.className = i === HUMAN ? 'my-cards' : 'seat-cards';

    el.append(cards, plate);
    root.appendChild(el);

    const bet = document.createElement('div');
    bet.className = 'bet';
    bet.dataset.pos = pos;
    root.appendChild(bet);

    return { el, stack, stats, dealer, action, cards, bet };
  });
}

function renderSeats(v, thinkingSeat) {
  if (!seatEls) buildSeats(v.seats);
  const winners = new Set((v.result ? v.result.winners : []).map((w) => w.seat));

  v.seats.forEach((seat, i) => {
    const parts = seatEls[i];
    parts.stack.textContent = seat.status === 'out' ? 'Out' : money(seat.stack);
    parts.stats.textContent = seat.stats.join(' / ');
    parts.dealer.hidden = !seat.is_button;

    const el = parts.el;
    el.classList.toggle('active', v.to_act === i && !v.hand_over);
    el.classList.toggle('thinking', thinkingSeat === i);
    el.classList.toggle('folded', seat.status === 'folded');
    el.classList.toggle('out', seat.status === 'out');
    el.classList.toggle('winner', v.hand_over && winners.has(i));

    const label = seat.status === 'all-in' && !seat.last_action ? 'All-in' : seat.last_action;
    if (parts.action.textContent !== (label || '')) {
      parts.action.textContent = label || '';
      parts.action.className = 'seat-action';
      if (label) parts.action.classList.add(`is-${label.split(' ')[0].toLowerCase()}`);
    }
    parts.action.hidden = !label;

    let cards = [];
    if (seat.cards) cards = seat.cards;
    else if (seat.has_cards && i !== HUMAN) cards = [null, null];
    renderCards(parts.cards, cards);
    parts.cards.classList.toggle('up', Boolean(seat.cards) && i !== HUMAN);

    parts.bet.hidden = !seat.bet;
    parts.bet.textContent = seat.bet ? moneyShort(seat.bet) : '';
  });
}

// ------------------------------------------------------------ rendering

function nextBotToAct(v) {
  // Seats are numbered clockwise, so the next actor is the next live seat.
  for (let step = 1; step < v.seats.length; step += 1) {
    const i = (HUMAN + step) % v.seats.length;
    if (v.seats[i].status === 'active') return i;
  }
  return null;
}

function render(v, { animating = false, thinkingSeat = null } = {}) {
  view = v;
  const thinking = thinkingSeat !== null
    ? thinkingSeat
    : (animating && v.to_act !== null && v.to_act !== HUMAN ? v.to_act : null);

  $('hand-number').textContent = `Hand #${v.hand_number}`;
  const difficulty = v.difficulty[0].toUpperCase() + v.difficulty.slice(1);
  $('street').textContent = `${v.hand_over ? 'Hand over' : v.street} · ${difficulty}`;

  renderSeats(v, thinking);

  const board = [...v.board];
  while (board.length < 5) board.push('slot');
  renderCards($('board'), board);
  $('pot').textContent = v.pot ? `Pot: ${money(v.pot)}` : '';
  $('hand-strength').textContent = v.human_hand ? `You have ${v.human_hand}` : '';

  renderResult(v);
  renderLog(v);
  renderControls(v, animating, thinking);

  const over = v.game_over && !animating;
  $('game-over').hidden = !over;
  if (over) {
    $('game-over-title').textContent = v.human_won_game ? 'You cleaned out the table!' : 'You’re out of chips';
    $('game-over-text').textContent = v.human_won_game
      ? `You beat all five AI players on ${v.difficulty} in ${v.hand_number} hands.`
      : `You lasted ${v.hand_number} hands on ${v.difficulty}. Ready for another go?`;
  }
}

function renderResult(v) {
  const toast = $('result-toast');
  const result = v.result;

  if (!v.hand_over || !result) {
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
  const items = v.log.map((line) => {
    const li = document.createElement('li');
    li.textContent = line;
    if (line.startsWith('---')) li.className = 'street';
    else if (line.startsWith('You ')) li.className = 'you';
    else if (/collects? |is out|out of chips/.test(line)) li.className = 'end';
    return li;
  });
  list.replaceChildren(...items);
  list.scrollTop = list.scrollHeight;
}

function renderControls(v, animating, thinking) {
  const legal = !animating && !busy ? v.legal : null;
  $('actions').hidden = !legal;
  $('hand-done').hidden = animating || busy || !v.hand_over || v.game_over;

  const waiting = $('waiting');
  waiting.hidden = Boolean(legal) || (v.hand_over && !animating && !busy);
  if (!waiting.hidden) {
    waiting.textContent = thinking !== null && v.seats[thinking]
      ? `${v.seats[thinking].name} is thinking…`
      : 'Dealing…';
  }

  if (!legal) return;

  $('btn-fold').disabled = !legal.fold;

  const me = v.seats[HUMAN];
  if (legal.check) {
    $('call-label').textContent = 'Check';
    $('call-amount').textContent = '';
  } else {
    $('call-label').textContent = legal.call >= me.stack ? 'All-in' : 'Call';
    $('call-amount').textContent = money(legal.call);
  }

  const raise = legal.raise;
  for (const id of ['btn-raise', 'raise-slider', 'raise-input', 'raise-caret']) $(id).disabled = !raise;
  document.querySelectorAll('[data-preset]').forEach((b) => { b.disabled = !raise; });

  if (!raise) {
    $('raise-label').textContent = 'Raise';
    $('raise-input').value = '';
    $('raise-slider').style.setProperty('--fill', '0%');
    for (const b of document.querySelectorAll('[data-preset]')) b.textContent = presetName(b.dataset.preset);
    return;
  }

  const slider = $('raise-slider');
  const changed = slider.min !== String(raise.min) || slider.max !== String(raise.max);
  slider.min = raise.min;
  slider.max = raise.max;
  slider.step = 1;

  for (const b of document.querySelectorAll('[data-preset]')) {
    b.textContent = `${presetName(b.dataset.preset)} (${moneyShort(presetAmount(b.dataset.preset))})`;
  }

  setRaise(changed ? raise.min : Number(slider.value));
}

// -------------------------------------------------------------- sizing

function presetName(preset) {
  return { min: 'Min', 0.5: '1/2 Pot', 0.75: '3/4 Pot', 1: 'Pot', max: 'All In' }[preset];
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
  return clampRaise(view.seats[HUMAN].bet + toCall + Number(preset) * potAfterCall);
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
  $('raise-label').textContent = amount === raise.max ? 'All-in' : raise.is_bet ? 'Bet' : 'Raise';
}

function inputCents() {
  const text = $('raise-input').value.replace(/[$,\s]/g, '');
  const dollars = Number(text);
  return text === '' || !Number.isFinite(dollars) ? NaN : Math.round(dollars * 100);
}

// ------------------------------------------------------------ playback

async function playFrames(frames, myEpoch) {
  for (const frame of frames || []) {
    if (epoch !== myEpoch) return;
    render(frame, { animating: true });
    const humanOut = frame.seats[HUMAN].status !== 'active';
    await sleep(humanOut ? FAST_FRAME_MS : FRAME_MS);
  }
}

async function resync() {
  // After an error, show the server's real state so the buttons come back.
  try {
    const data = await api('/api/state');
    if (data && data.view) {
      render(data.view);
      return;
    }
  } catch {
    /* fall through to the last known view */
  }
  if (view) render(view);
}

async function send(path, body) {
  if (busy) return;
  busy = true;
  showError(null);

  if (view) render(view, { animating: true, thinkingSeat: path === '/api/action' ? nextBotToAct(view) : null });

  const myEpoch = epoch;
  let response = null;
  let failure = null;

  try {
    response = await api(path, body);
    await playFrames(response.frames, myEpoch);
  } catch (error) {
    failure = error;
  }

  if (epoch !== myEpoch) return; // the player left the table meanwhile
  busy = false;

  if (response) {
    render(response.view);
  } else {
    showError(failure.message);
    await resync();
  }
}

async function startGame(difficulty) {
  if (busy) return;
  busy = true;
  showError(null);
  showScreen('loading');
  seatEls = null;

  const myEpoch = epoch;
  let response = null;

  try {
    [response] = await Promise.all([api('/api/new', { difficulty }), sleep(LOADING_MIN_MS)]);
    showScreen('game');
    await playFrames(response.frames, myEpoch);
  } catch (error) {
    if (!response) {
      busy = false;
      showScreen('lobby');
      showError(error.message);
      return;
    }
  }

  busy = false;
  render(response.view);
}

// -------------------------------------------------------------- events

function act(action) {
  if (busy || !view || !view.legal || !$('actions').offsetParent) return;
  const legal = view.legal;

  if (action === 'fold' && !legal.fold) return;

  if (action === 'raise') {
    if (!legal.raise) return;
    const amount = clampRaise(inputCents());
    send('/api/action', { action: 'raise', amount });
    return;
  }

  send('/api/action', { action });
}

document.querySelectorAll('#lobby .mode').forEach((button) => {
  button.addEventListener('click', () => startGame(button.dataset.difficulty));
});

$('btn-fold').addEventListener('click', () => act('fold'));
$('btn-call').addEventListener('click', () => act('check_call'));
$('btn-raise').addEventListener('click', () => act('raise'));
$('next-hand').addEventListener('click', () => send('/api/next', {}));
$('play-again').addEventListener('click', () => startGame(view.difficulty));

$('raise-caret').addEventListener('click', () => {
  const caret = $('raise-caret');
  const open = caret.getAttribute('aria-expanded') !== 'true';
  caret.setAttribute('aria-expanded', String(open));
  $('presets').hidden = !open;
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

$('history-toggle').addEventListener('click', () => toggleHistory($('history').hidden));
$('history-close').addEventListener('click', () => toggleHistory(false));

function toLobby() {
  epoch += 1;
  busy = false;
  toggleHistory(false);
  showScreen('lobby');
  showError(null);
}

$('to-lobby').addEventListener('click', toLobby);
$('quit').addEventListener('click', () => {
  if (view && !view.hand_over && !view.game_over
      && !confirm('Leave this table? Starting a new game will forfeit it.')) return;
  toLobby();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') toggleHistory(false);
  if (e.target.tagName === 'INPUT' || e.ctrlKey || e.metaKey || e.altKey) return;
  if ($('game').hidden) return;

  const key = e.key.toLowerCase();
  if (key === 'f') act('fold');
  else if (key === 'c') act('check_call');
  else if (key === 'r') act('raise');
  else if (key === 'n' && view && view.hand_over && !view.game_over && !busy) send('/api/next', {});
});

// Resume a game in progress after a page reload.
api('/api/state')
  .then((data) => {
    if (data && data.view) {
      showScreen('game');
      render(data.view);
    }
  })
  .catch(() => { /* no game yet: stay in the lobby */ });
