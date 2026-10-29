'use strict';

const $ = (id) => document.getElementById(id);
let mode = 'global';
let requestVersion = 0;

async function getJSON(path, signal) {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error('Rankings are unavailable. Please try refreshing.');
  return response.json();
}

async function refreshRankings() {
  const version = ++requestVersion;
  const selectedMode = mode;
  if (selectedMode === 'global') return;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const data = await getJSON(selectedMode === 'pilot' ? '/api/leaderboard' : '/api/state', controller.signal);
    if (version !== requestVersion) return;
    const rows = selectedMode === 'pilot' ? data.standings : data?.view?.leaderboard?.standings || [];
    $('players').replaceChildren();
    for (const player of rows) {
      const isGlobal = selectedMode === 'pilot';
      const cents = isGlobal ? player.net_winnings_cents : Math.round(player.net_profit * 100);
      const isYou = isGlobal ? player.is_you : player.player_id === '0';
      const tr = document.createElement('tr');
      if (isYou) tr.className = 'is-you';
      const amount = `${cents > 0 ? '+' : cents < 0 ? '−' : ''}${(Math.abs(cents) / 100).toLocaleString('en-US', { style: 'currency', currency: 'USD' })}`;
      const values = [player.rank, player.name, amount, isGlobal ? player.rounds_participated : player.hands_played];
      values.forEach((value, index) => {
        const td = document.createElement('td');
        td.textContent = value;
        if (index === 1 && isYou) {
          const badge = document.createElement('span');
          badge.className = 'you'; badge.textContent = 'YOU'; td.append(badge);
        }
        if (index === 2) td.className = cents > 0 ? 'positive' : cents < 0 ? 'negative' : '';
        tr.append(td);
      });
      $('players').append(tr);
    }
    $('rankings').hidden = rows.length === 0;
    $('message').hidden = rows.length > 0;
    $('message').textContent = selectedMode === 'pilot'
      ? 'No players yet. Join a game to start building your record.'
      : 'No current game. Go back to the dashboard and join a game to see table rankings.';
    $('updated').textContent = `Updated ${new Date().toLocaleTimeString()} · refreshes every 10 seconds`;
  } catch (error) {
    if (version !== requestVersion) return;
    $('message').hidden = false;
    $('message').textContent = error.name === 'AbortError' ? 'The request timed out. Please refresh.' : error.message;
  } finally { clearTimeout(timeout); }
}

function selectMode(nextMode) {
  mode = nextMode;
  requestVersion += 1;
  $('rankings').hidden = true;
  $('message').hidden = false;
  $('message').textContent = 'Loading rankings…';
  const gpi = mode === 'global';
  const pilot = mode === 'pilot';
  $('global-tab').setAttribute('aria-pressed', String(gpi));
  $('pilot-tab').setAttribute('aria-pressed', String(pilot));
  $('game-tab').setAttribute('aria-pressed', String(mode === 'game'));
  $('gpi-source').hidden = !gpi;
  $('refresh').hidden = gpi;
  $('message').hidden = gpi;
  $('board-title').textContent = gpi ? 'Global Poker Index' : pilot ? 'Poker Pilot leaderboard' : 'This game';
  $('winnings-label').textContent = pilot ? 'Total net winnings' : 'Game net profit';
  $('description').textContent = gpi
    ? 'Professional poker rankings from The Hendon Mob’s Global Poker Index.'
    : pilot ? 'Poker Pilot players ranked by total net winnings across completed rounds.'
    : 'Players at your current table, including AI opponents, ranked by game net profit.';
  if (gpi) $('updated').textContent = 'Source: The Hendon Mob / Global Poker Index';
  refreshRankings();
}

$('global-tab').addEventListener('click', () => selectMode('global'));
$('pilot-tab').addEventListener('click', () => selectMode('pilot'));
$('game-tab').addEventListener('click', () => selectMode('game'));
$('refresh').addEventListener('click', refreshRankings);
setInterval(() => { if (!document.hidden) refreshRankings(); }, 10000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) refreshRankings(); });
selectMode('global');
