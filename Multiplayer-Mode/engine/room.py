"""Online rooms: up to ten people at one No-Limit Hold'em table, run on
pokerkit. In PvE rooms AI players fill the empty seats. All amounts are
integer cents.

A room changes in response to its players' messages and to one timer (AI
turns, turn clocks, the pause between hands). After every change each
connection gets its own view: only that viewer's hole cards, with the seats
rotated so the viewer sits at the bottom.
"""

from __future__ import annotations

import json
import math
import random
import secrets
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

from pokerkit import (
    Automation,
    BlindOrStraddlePosting,
    BoardDealing,
    Card,
    ChipsPulling,
    HoleCardsShowingOrMucking,
    Mode,
    NoLimitTexasHoldem,
    StandardHighHand,
    State,
)

from .bots import CHARACTERS, DIFFICULTIES, Bot, BotView, OpponentInfo, create_bot

SMALL_BLIND = 100  # $1.00
BIG_BLIND = 200  # $2.00
STARTING_STACK = 20_000  # $200.00
SEAT_COUNT = 6  # seats at a new table, and at every PvE table
MIN_SEATS = 2
MAX_SEATS = 10  # PvP tables go up to full ring
MAX_PLAYERS = 16  # seated players plus people waiting for a seat
MAX_NAME = 16
STREET_NAMES = ('Preflop', 'Flop', 'Turn', 'River')
MODES = ('pvp', 'pve')
HUMAN_COLORS = ('teal', 'yellow', 'red', 'indigo', 'blue')
# Player portraits, static/art/portraits/p1.png to p10.png. Each AI player
# has its own; people get a free one, the AI players' faces last.
BOT_PORTRAITS = {'MathNinja': 5, 'CardShark88': 9, 'RiverRat': 10, 'LuckyLlama': 7, 'TiltMaster': 8}
HUMAN_PORTRAITS = (1, 2, 3, 4, 6, 5, 9, 10, 7, 8)

TURN_SECONDS = 30
AWAY_SECONDS = 4  # turn clock for players who disconnected or left
BOT_SECONDS = 0.9  # pause before an AI acts, so people can follow along
NEXT_HAND_SECONDS = 6


class GameError(Exception):
    """An illegal request from a player."""


def card_code(card: Card) -> str:
    return f'{card.rank.value}{card.suit.value}'


def money(cents: int) -> str:
    return f'${cents / 100:,.2f}'


def hand_label(hole: list[Card], board: list[Card]) -> str | None:
    if len(board) < 3 or len(hole) < 2:
        return None

    return StandardHighHand.from_game(hole, board).entry.label.value


def best_five(hole: list[Card], board: list[Card]) -> list[str]:
    """The best five cards, grouped (pairs first) and high to low."""

    cards = StandardHighHand.from_game(hole, board).cards
    ranks = [card.rank.value for card in cards]

    def order(card: Card) -> tuple[int, int]:
        return -ranks.count(card.rank.value), -'23456789TJQKA'.index(card.rank.value)

    return [card_code(card) for card in sorted(cards, key=order)]


@dataclass
class PlayerStats:
    hands: int = 0
    vpip: int = 0
    pfr: int = 0
    aggressive: int = 0
    calls: int = 0
    won: int = 0
    faced_bets: int = 0
    folds_to_bet: int = 0

    @property
    def vpip_rate(self) -> float:
        return (self.vpip + 3) / (self.hands + 10)  # prior of 30 percent

    @property
    def aggression_factor(self) -> float:
        return (self.aggressive + 1) / (self.calls + 1)

    @property
    def fold_to_bet(self) -> float:
        return (self.folds_to_bet + 2) / (self.faced_bets + 5)  # prior of 40 percent

    def summary(self) -> list[int]:
        """VPIP %, PFR %, hands won."""

        if not self.hands:
            return [0, 0, self.won]

        return [
            round(100 * self.vpip / self.hands),
            round(100 * self.pfr / self.hands),
            self.won,
        ]


@dataclass
class Player:
    """A person in the room, seated or waiting for a seat."""

    token: str
    name: str
    color: str
    portrait: int
    account_id: str | int | None = None
    seat: int | None = None
    left: bool = False  # left mid-hand; removed when the hand ends


@dataclass
class Seat:
    name: str
    avatar: str | None
    color: str
    bot: Bot | None = None
    token: str | None = None  # the player sitting here; None for AI players
    stack: int = STARTING_STACK
    stats: PlayerStats = field(default_factory=PlayerStats)
    portrait: int | None = None


class Room:
    def __init__(self, code: str, seed: int | None = None, timers: bool = True) -> None:
        self.lock = threading.RLock()
        self.code = code
        self.rng = random.Random(seed)
        self.timers = timers  # tests turn real timers off and call run_pending()
        self.mode = 'pvp'
        self.difficulty = 'medium'
        self.status = 'lobby'
        self.players: dict[str, Player] = {}
        self.host: str | None = None
        self.sockets: dict[object, str] = {}
        self.seats: list[Seat | None] = [None] * SEAT_COUNT
        self.idle_since: float | None = time.time()
        # The table chat (chat-api) room. Secret, unlike the room code, so
        # only people in this room can read or post in its chat.
        self.chat_room = secrets.token_urlsafe(24)

        self.hand_number = 0
        self.button: int | None = None
        self.state: State | None = None
        self.order: list[int] = []
        self.dealt: dict[int, Seat] = {}  # who was dealt in, in case a seat changes hands
        self.hole: dict[int, list[Card]] = {}
        self.revealed: set[int] = set()
        self.folded: set[int] = set()
        self.last_action: dict[int, str] = {}
        self.result: dict | None = None
        self.paused = False  # hand over and fewer than two players with chips
        self.log: list[str] = []

        self.gen = 0
        self.timer: threading.Timer | None = None
        self.pending: Callable[[], None] | None = None
        self.deadline: float | None = None
        self.clock_total: float | None = None

    # ------------------------------------------------------------- status

    @property
    def hand_over(self) -> bool:
        return self.state is None or not self.state.status

    @property
    def board(self) -> list[Card]:
        return list(self.state.get_board_cards(0)) if self.state else []

    def index_of(self, seat: int) -> int | None:
        return self.order.index(seat) if seat in self.order else None

    @property
    def actor_seat(self) -> int | None:
        if self.hand_over or self.state.actor_index is None:
            return None

        return self.order[self.state.actor_index]

    def in_live_hand(self, seat: int) -> bool:
        return not self.hand_over and seat in self.order and seat not in self.folded

    def connected(self, token: str) -> bool:
        return token in self.sockets.values()

    def present(self) -> list[Player]:
        return [p for p in self.players.values() if not p.left]

    def funded(self) -> list[int]:
        return [i for i, seat in enumerate(self.seats) if seat is not None and seat.stack > 0]

    def player_for(self, token: str) -> Player:
        player = self.players.get(token)

        if player is None or player.left:
            raise GameError('You are not in this room.')

        return player

    # ------------------------------------------------------------ members

    def clean_name(self, name: object) -> str:
        if not isinstance(name, str) or not ' '.join(name.split()):
            raise GameError('Enter a name first.')

        base = ' '.join(name.split())[:MAX_NAME]
        taken = {p.name.lower() for p in self.present()}
        taken.update(c.name.lower() for c in CHARACTERS)
        candidate = base
        number = 2

        while candidate.lower() in taken:
            suffix = str(number)
            candidate = base[:MAX_NAME - len(suffix)] + suffix
            number += 1

        return candidate

    def join(self, name: object, account_id: str | int | None = None) -> str:
        """Add a player and return their secret token."""

        with self.lock:
            if len(self.present()) >= MAX_PLAYERS:
                raise GameError('This room is full.')

            name = self.clean_name(name)
            used = {p.color for p in self.present()}
            color = next((c for c in HUMAN_COLORS if c not in used), HUMAN_COLORS[len(self.players) % len(HUMAN_COLORS)])
            faces = {p.portrait for p in self.present()} | {s.portrait for s in self.seats if s is not None}
            portrait = next((n for n in HUMAN_PORTRAITS if n not in faces),
                            HUMAN_PORTRAITS[len(self.players) % len(HUMAN_PORTRAITS)])
            token = secrets.token_urlsafe(16)
            player = Player(token, name, color, portrait, account_id=account_id)
            self.players[token] = player

            if self.host is None:
                self.host = token

            if None in self.seats:
                self.sit(player, self.seats.index(None))

            if self.status == 'playing':
                where = 'joins next hand' if player.seat is not None else 'is waiting for a seat'
                self.log.append(f'{name} {where}.')

            if self.status == 'playing':
                self.paused = False

            self.changed()

            return token

    def sit(self, player: Player, seat: int) -> None:
        self.seats[seat] = Seat(player.name, None, player.color, token=player.token, portrait=player.portrait)
        player.seat = seat

    def remove(self, player: Player) -> None:
        if player.seat is not None and self.seats[player.seat] is not None \
                and self.seats[player.seat].token == player.token:
            self.seats[player.seat] = None

        self.players.pop(player.token, None)

        if self.host == player.token:
            self.host = next((p.token for p in self.present()), None)

    def leave(self, token: str) -> None:
        with self.lock:
            player = self.player_for(token)

            if self.status == 'playing':
                self.log.append(f'{player.name} left the table.')

            if player.seat is not None and not self.hand_over and player.seat in self.order:
                # Keep the seat until the hand ends; the clock folds it.
                player.left = True

                if self.host == token:
                    self.host = next((p.token for p in self.present()), None)
            else:
                self.remove(player)

            for ws, owner in list(self.sockets.items()):
                if owner == token:
                    del self.sockets[ws]

            self.changed()

    @property
    def empty(self) -> bool:
        return not self.present()

    # -------------------------------------------------------- connections

    def connect(self, token: str, ws: object) -> None:
        with self.lock:
            self.player_for(token)
            first = not self.sockets
            returning = not self.connected(token)
            self.sockets[ws] = token
            self.idle_since = None
            actor = self.actor_seat

            # Restart the schedule if nothing was running or the player on
            # the clock is back; a returning player is no longer shown away;
            # otherwise only the new tab needs a view.
            if first or (actor is not None and self.seats[actor].token == token):
                self.changed()
            elif returning:
                self.broadcast()
            else:
                self.send(ws, self.view_for(token))

    def disconnect(self, ws: object) -> None:
        with self.lock:
            token = self.sockets.pop(ws, None)

            if not self.sockets:
                self.idle_since = time.time()

            if token is None:
                return

            actor = self.actor_seat

            if actor is not None and self.seats[actor].token == token and not self.connected(token):
                self.changed()  # switch to the short away clock
            else:
                self.broadcast()

    def send(self, ws: object, message: dict) -> None:
        try:
            ws.send(json.dumps(message))
        except Exception:
            self.sockets.pop(ws, None)

    def broadcast(self) -> None:
        for ws, token in list(self.sockets.items()):
            self.send(ws, self.view_for(token))

    # ---------------------------------------------------------- scheduling

    def changed(self) -> None:
        """Schedule whatever happens next, then show everyone the state."""

        self.gen += 1
        self.pending = None
        self.deadline = None
        self.clock_total = None

        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

        # Nobody watching: freeze until someone reconnects.
        if self.status == 'playing' and self.sockets:
            if self.hand_over:
                if not self.paused:
                    self.schedule(NEXT_HAND_SECONDS, self.deal, clock=True)
            else:
                seat = self.seats[self.actor_seat]

                if seat.bot is not None:
                    self.schedule(BOT_SECONDS, self.bot_turn)
                else:
                    player = self.players.get(seat.token)
                    away = player is None or player.left or not self.connected(seat.token)
                    self.schedule(AWAY_SECONDS if away else TURN_SECONDS, self.timeout, clock=True)

        self.broadcast()

    def schedule(self, delay: float, fn: Callable[[], None], clock: bool = False) -> None:
        self.pending = fn

        if clock:
            self.deadline = time.time() + delay
            self.clock_total = delay

        if self.timers:
            self.timer = threading.Timer(delay, self.fire, (self.gen,))
            self.timer.daemon = True
            self.timer.start()

    def fire(self, gen: int) -> None:
        with self.lock:
            if gen != self.gen or self.pending is None:
                return

            fn = self.pending
            self.pending = None

            try:
                fn()
            except Exception:
                traceback.print_exc()
                self.changed()

    def run_pending(self) -> bool:
        """Run the scheduled step now (for tests). False if none."""

        with self.lock:
            if self.pending is None:
                return False

            self.fire(self.gen)
            return True

    def close(self) -> None:
        with self.lock:
            self.gen += 1
            self.pending = None

            if self.timer is not None:
                self.timer.cancel()

    # --------------------------------------------------------- game flow

    def update_settings(self, token: str, mode: object, difficulty: object) -> None:
        with self.lock:
            self.require_host(token)

            if self.status != 'lobby':
                raise GameError('Settings can only change before the game starts.')

            if mode not in MODES:
                raise GameError('Unknown game mode.')

            if difficulty not in DIFFICULTIES:
                raise GameError(f'Difficulty must be one of: {", ".join(DIFFICULTIES)}.')

            self.mode = mode
            self.difficulty = difficulty

            if mode == 'pve':
                # A 6-max table with AI players in the empty seats (bigger
                # only if more people are already sitting down).
                target = max(SEAT_COUNT, len(self.seats) - self.seats.count(None))
                self.seats += [None] * (target - len(self.seats))

                while len(self.seats) > target:
                    self.drop_seat(len(self.seats) - 1 - self.seats[::-1].index(None))

            self.changed()

    def resize(self, token: str, count: object) -> None:
        """PvP: the host opens or closes seats. Seats can be added at any
        time; an empty seat can be removed before the game or between hands
        (a live hand refers to seats by number)."""

        with self.lock:
            self.require_host(token)

            if self.mode != 'pvp':
                raise GameError('Seats can only be changed in Players only (PvP) rooms.')

            if isinstance(count, bool) or not isinstance(count, int) or not MIN_SEATS <= count <= MAX_SEATS:
                raise GameError(f'A table has {MIN_SEATS} to {MAX_SEATS} seats.')

            if count < len(self.seats):
                if not self.hand_over:
                    raise GameError('Seats can only be removed between hands.')

                if self.seats.count(None) < len(self.seats) - count:
                    raise GameError('Only empty seats can be removed.')

                while len(self.seats) > count:
                    self.drop_seat(len(self.seats) - 1 - self.seats[::-1].index(None))
            else:
                while len(self.seats) < count:
                    self.seats.append(None)

                    # Someone waiting for a seat takes it (and plays from the next hand).
                    waiting = next((p for p in self.present() if p.seat is None), None)

                    if waiting is not None:
                        self.sit(waiting, len(self.seats) - 1)

                        if self.status == 'playing':
                            self.log.append(f'{waiting.name} joins next hand.')

            if self.status == 'playing':
                self.log.append(f'The host set the table to {count} seats.')
                self.paused = False

            self.changed()

    def drop_seat(self, index: int) -> None:
        """Remove an empty seat; seats after it move down one number."""

        def move(seat: int) -> int:
            return seat - 1 if seat > index else seat

        del self.seats[index]

        for player in self.players.values():
            if player.seat is not None:
                player.seat = move(player.seat)

        if self.button == index:
            # Step back so the button still moves on to the next seat.
            self.button = (index - 1) % len(self.seats)
        elif self.button is not None:
            self.button = move(self.button)

        # What the finished hand still shows.
        self.order = [move(s) for s in self.order if s != index]
        self.dealt = {move(s): seat for s, seat in self.dealt.items() if s != index}
        self.hole = {move(s): cards for s, cards in self.hole.items() if s != index}
        self.revealed = {move(s) for s in self.revealed if s != index}
        self.folded = {move(s) for s in self.folded if s != index}
        self.last_action = {move(s): text for s, text in self.last_action.items() if s != index}

        if self.result is not None:
            self.result['winners'] = [w | {'seat': move(w['seat'])} for w in self.result['winners'] if w['seat'] != index]

    def require_host(self, token: str) -> None:
        self.player_for(token)

        if token != self.host:
            raise GameError('Only the host can do that.')

    def start_game(self, token: str) -> None:
        with self.lock:
            self.require_host(token)

            if self.status != 'lobby':
                raise GameError('The game has already started.')

            seated = sum(1 for seat in self.seats if seat is not None)

            if self.mode == 'pvp' and seated < 2:
                raise GameError('You need at least two players. Share the room code, or add AI players.')

            self.status = 'playing'
            self.log = []
            self.tidy()
            self.start_hand()
            self.changed()

    def tidy(self) -> None:
        """Between hands: free departed players' seats, seat whoever is
        waiting, and (PvE) give AI players the empty seats."""

        for player in list(self.players.values()):
            if player.left:
                self.remove(player)

        for player in self.present():
            if player.seat is not None:
                continue

            if None in self.seats:
                self.sit(player, self.seats.index(None))
            elif self.mode == 'pve':
                bots = [i for i, seat in enumerate(self.seats) if seat.bot is not None]

                if bots:
                    # Prefer replacing an AI player who is out of chips.
                    seat = min(bots, key=lambda i: (self.seats[i].stack > 0, i))
                    self.log.append(f'{player.name} takes {self.seats[seat].name}\'s seat.')
                    self.sit(player, seat)

        if self.mode == 'pve':
            used = {seat.name for seat in self.seats if seat is not None}
            faces = {seat.portrait for seat in self.seats if seat is not None}
            # AI players whose face someone already has come last.
            spare = sorted((c for c in CHARACTERS if c.name not in used),
                           key=lambda c: BOT_PORTRAITS[c.name] in faces)

            for i in range(len(self.seats)):
                if self.seats[i] is None and spare:
                    character = spare.pop(0)
                    bot = create_bot(self.difficulty, character, random.Random(self.rng.random()))
                    self.seats[i] = Seat(character.name, character.avatar, character.color, bot,
                                         portrait=BOT_PORTRAITS[character.name])

    def deal(self) -> None:
        self.tidy()

        if len(self.funded()) >= 2:
            self.start_hand()
        else:
            self.paused = True
            self.state = None
            self.order = []
            self.dealt = {}
            self.hole = {}
            self.revealed = set()
            self.folded = set()
            self.last_action = {}

        self.changed()

    def restart(self, token: str) -> None:
        with self.lock:
            self.require_host(token)

            if not self.paused:
                raise GameError('The game is still running.')

            for seat in self.seats:
                if seat is not None:
                    seat.stack = STARTING_STACK

            self.log.append('The host reset everyone to ' + money(STARTING_STACK) + '.')
            self.paused = False
            self.deal()

    def rebuy(self, token: str) -> None:
        with self.lock:
            player = self.player_for(token)

            if player.seat is None:
                raise GameError('You do not have a seat yet.')

            seat = self.seats[player.seat]

            if seat.stack > 0 or self.in_live_hand(player.seat):
                raise GameError('You can only rebuy when you are out of chips.')

            seat.stack = STARTING_STACK
            self.log.append(f'{player.name} rebuys for {money(STARTING_STACK)}.')
            self.paused = False
            self.changed()

    # -------------------------------------------------------------- hands

    def next_active(self, seat: int) -> int:
        count = len(self.seats)

        for step in range(1, count + 1):
            candidate = (seat + step) % count

            if self.seats[candidate] is not None and self.seats[candidate].stack > 0:
                return candidate

        raise GameError('No players left.')

    def start_hand(self) -> None:
        active = self.funded()
        self.paused = False

        if self.button is None:
            self.button = self.rng.choice(active)
        else:
            self.button = self.next_active(self.button)

        # pokerkit order: small blind first, button last (also heads-up).
        order = []
        seat = self.button

        for _ in range(len(active)):
            seat = self.next_active(seat)
            order.append(seat)

        self.order = order
        self.dealt = {s: self.seats[s] for s in order}
        self.hand_number += 1
        self.state = NoLimitTexasHoldem.create_state(
            tuple(Automation),
            True,
            0,
            (SMALL_BLIND, BIG_BLIND),
            BIG_BLIND,
            tuple(self.seats[s].stack for s in order),
            len(order),
            mode=Mode.CASH_GAME,
        )
        # Hole cards are snapshotted because pokerkit clears losing hands.
        self.hole = {s: list(self.state.hole_cards[i]) for i, s in enumerate(order)}
        self.revealed = set()
        self.folded = set()
        self.result = None
        self.op_cursor = 0
        self.board_dealt = 0
        self.preflop_action = {s: 'none' for s in order}
        self.postflop_raises = {s: 0 for s in order}
        self.preflop_raiser: int | None = None
        self.raises_this_street = 0
        self.last_action = {}
        self.vpip_seen: set[int] = set()
        self.pfr_seen: set[int] = set()

        for s in order:
            self.seats[s].stats.hands += 1

        self.log.append(f'Hand #{self.hand_number} - {self.seats[self.button].name} has the button.')
        self.sync()

    def describe_action(self, seat: int, action: str, amount: int | None) -> str:
        """Phrase an action before it is applied, and remember it as the
        seat's latest action for display."""

        state = self.state
        name = self.seats[seat].name

        if action == 'fold':
            short = 'Fold'
            text = f'{name} folds'
        elif action == 'check_call':
            to_call = state.checking_or_calling_amount

            if to_call:
                all_in = to_call == state.stacks[state.actor_index]
                short = 'All-in' if all_in else f'Call {money(to_call)}'
                text = f'{name} calls {money(to_call)}' + (' (all-in)' if all_in else '')
            else:
                short = 'Check'
                text = f'{name} checks'
        else:
            is_bet = max(state.bets) == 0
            all_in = amount == state.max_completion_betting_or_raising_to_amount
            verb = 'Bet' if is_bet else 'Raise'
            short = 'All-in' if all_in else f'{verb} {money(amount)}'
            phrase = 'bets' if is_bet else 'raises to'
            text = f'{name} {phrase} {money(amount)}' + (' (all-in)' if all_in else '')

        self.last_action[seat] = short

        return text

    def sync(self) -> None:
        """Turn newly automated pokerkit operations into log lines."""

        operations = self.state.operations
        blinds = []

        def flush_blinds() -> None:
            # pokerkit posts the big blind first; list the small blind first.
            self.log.extend(text for _, text in sorted(blinds, key=lambda b: b[0]))
            blinds.clear()

        while self.op_cursor < len(operations):
            op = operations[self.op_cursor]
            self.op_cursor += 1

            if not isinstance(op, BlindOrStraddlePosting):
                flush_blinds()

            if isinstance(op, BlindOrStraddlePosting):
                seat = self.order[op.player_index]
                is_small = op.player_index == (len(self.order) - 1 if len(self.order) == 2 else 0)
                kind = 'small' if is_small else 'big'
                blinds.append((kind != 'small', f'{self.seats[seat].name} posts the {kind} blind ({money(op.amount)}).'))
            elif isinstance(op, BoardDealing):
                self.board_dealt += len(op.cards)
                name = STREET_NAMES[min(self.board_dealt - 2, 3)]
                cards = ' '.join(card_code(card) for card in op.cards)
                self.log.append(f'--- {name}: {cards} ---')
                self.raises_this_street = 0
                self.last_action = {}
            elif isinstance(op, HoleCardsShowingOrMucking):
                seat = self.order[op.player_index]
                name = self.seats[seat].name

                if op.hole_cards:
                    self.revealed.add(seat)
                    cards = ' '.join(card_code(card) for card in op.hole_cards)
                    self.log.append(f'{name} shows {cards}.')
                else:
                    self.log.append(f'{name} mucks.')
            elif isinstance(op, ChipsPulling):
                seat = self.order[op.player_index]
                self.log.append(f'{self.seats[seat].name} collects {money(op.amount)}.')

        flush_blinds()

        if self.hand_over and self.result is None:
            self.finish_hand()

    def finish_hand(self) -> None:
        state = self.state

        for i, seat in enumerate(self.order):
            self.seats[seat].stack = state.stacks[i]

        board = self.board
        showdown = len(self.revealed) >= 2
        winners = []

        for i, seat in enumerate(self.order):
            net = state.payoffs[i]

            if net > 0:
                self.seats[seat].stats.won += 1
                hand = None
                best = None

                if showdown and seat in self.revealed:
                    hand = hand_label(self.hole[seat], board)
                    best = best_five(self.hole[seat], board)

                winners.append({
                    'seat': seat,
                    'token': self.seats[seat].token,
                    'name': self.seats[seat].name,
                    'net': net,
                    'hand': hand,
                    'best': best,
                })

        winners.sort(key=lambda w: -w['net'])

        if not winners:
            headline = 'Split pot.'
        else:
            top = winners[0]
            headline = f'{top["name"]} wins {money(top["net"])}'
            headline += f' with {top["hand"]}.' if top['hand'] else '.'

        self.result = {
            'headline': headline,
            'winners': winners,
            'showdown': showdown,
            'nets': {self.seats[seat].token: state.payoffs[i] for i, seat in enumerate(self.order)},
        }

        for seat in self.order:
            if self.seats[seat].stack == 0:
                self.log.append(f'{self.seats[seat].name} is out of chips.')

    # ------------------------------------------------------------ actions

    def track(self, seat: int, action: str) -> None:
        """Record stats and public action history before applying."""

        state = self.state
        stats = self.seats[seat].stats
        facing_bet = state.checking_or_calling_amount > 0
        preflop = state.street_index == 0

        # Fold-to-bet only counts after the flop; preflop folds are just
        # unplayable hands.
        if facing_bet and not preflop:
            stats.faced_bets += 1

        if action == 'fold':
            self.folded.add(seat)

            if facing_bet and not preflop:
                stats.folds_to_bet += 1
        elif action == 'check_call' and facing_bet:
            stats.calls += 1

            if preflop:
                if self.preflop_action[seat] == 'none':
                    self.preflop_action[seat] = 'call'

                if seat not in self.vpip_seen:
                    self.vpip_seen.add(seat)
                    stats.vpip += 1
        elif action == 'raise':
            stats.aggressive += 1
            self.raises_this_street += 1

            if preflop:
                self.preflop_action[seat] = 'raise'
                self.preflop_raiser = seat

                for seen, counter in ((self.vpip_seen, 'vpip'), (self.pfr_seen, 'pfr')):
                    if seat not in seen:
                        seen.add(seat)
                        setattr(stats, counter, getattr(stats, counter) + 1)
            else:
                self.postflop_raises[seat] += 1

    def apply(self, seat: int, action: str, amount: int | None = None) -> None:
        state = self.state

        if action == 'fold':
            if state.checking_or_calling_amount == 0:
                raise GameError('There is nothing to fold to - you can check.')

            if not state.can_fold():
                raise GameError('You cannot fold right now.')
        elif action == 'check_call':
            if not state.can_check_or_call():
                raise GameError('You cannot check or call right now.')
        elif action == 'raise':
            if amount is None or not state.can_complete_bet_or_raise_to(amount):
                raise GameError('That bet size is not allowed.')
        else:
            raise GameError(f'Unknown action: {action!r}')

        self.track(seat, action)
        self.log.append(self.describe_action(seat, action, amount))

        if action == 'fold':
            state.fold()
        elif action == 'check_call':
            state.check_or_call()
        else:
            state.complete_bet_or_raise_to(amount)

        self.sync()

    def act(self, token: str, action: object, amount: object = None) -> None:
        with self.lock:
            player = self.player_for(token)

            if player.seat is None or self.actor_seat != player.seat:
                raise GameError('It is not your turn.')

            if action == 'raise':
                if isinstance(amount, bool) or not isinstance(amount, (int, float))                         or not math.isfinite(amount) or amount != int(amount):
                    raise GameError('Bet sizes must be whole cents.')

                amount = int(amount)
            else:
                amount = None

            self.apply(player.seat, action, amount)
            self.changed()

    def timeout(self) -> None:
        """The clock ran out: check if that is free, otherwise fold."""

        seat = self.actor_seat

        if seat is not None and self.seats[seat].bot is None:
            player = self.players.get(self.seats[seat].token)

            if player is not None and not player.left and self.connected(player.token):
                self.log.append(f'{player.name} ran out of time.')

            free = self.state.checking_or_calling_amount == 0
            self.apply(seat, 'check_call' if free else 'fold')

        self.changed()

    # ---------------------------------------------------------------- bots

    def players_behind(self, index: int) -> int:
        """Opponents still to act after ``index`` in the preflop order."""

        state = self.state
        count = len(self.order)
        sequence = [1, 0] if count == 2 else list(range(2, count)) + [0, 1]
        later = sequence[sequence.index(index) + 1:]

        return sum(1 for i in later if state.statuses[i] and state.stacks[i] > 0)

    def bot_view(self, seat: int) -> BotView:
        state = self.state
        index = self.index_of(seat)
        can_raise = state.can_complete_bet_or_raise_to()
        opponents = []

        for i, other in enumerate(self.order):
            if other == seat or not state.statuses[i]:
                continue

            stats = self.seats[other].stats
            opponents.append(OpponentInfo(
                preflop=self.preflop_action[other],
                postflop_raises=self.postflop_raises[other],
                vpip_rate=stats.vpip_rate,
                aggression_factor=stats.aggression_factor,
                fold_to_bet=stats.fold_to_bet,
            ))

        limpers = sum(1 for s, a in self.preflop_action.items() if a == 'call' and s != seat)

        return BotView(
            hole=list(self.hole[seat]),
            board=self.board,
            street=state.street_index,
            to_call=state.checking_or_calling_amount,
            pot=state.total_pot_amount,
            my_bet=state.bets[index],
            my_stack=state.stacks[index],
            big_blind=BIG_BLIND,
            min_raise_to=state.min_completion_betting_or_raising_to_amount if can_raise else None,
            max_raise_to=state.max_completion_betting_or_raising_to_amount if can_raise else None,
            opponents=opponents,
            players_behind=self.players_behind(index),
            raises_this_street=self.raises_this_street,
            limpers=limpers,
            was_preflop_raiser=self.preflop_raiser == seat,
        )

    def bot_turn(self) -> None:
        seat = self.actor_seat

        if seat is not None and self.seats[seat].bot is not None:
            action, amount = self.seats[seat].bot.decide(self.bot_view(seat))
            state = self.state

            # Never fold for free, and fall back to calling on bad sizes.
            if action == 'fold' and state.checking_or_calling_amount == 0:
                action = 'check_call'

            if action == 'raise' and (
                    amount is None or not state.can_complete_bet_or_raise_to(amount)
            ):
                action, amount = 'check_call', None

            self.apply(seat, action, amount if action == 'raise' else None)

        self.changed()

    # --------------------------------------------------------------- views

    def legal_actions(self, seat: int) -> dict | None:
        if self.actor_seat != seat:
            return None

        state = self.state
        to_call = state.checking_or_calling_amount
        can_raise = state.can_complete_bet_or_raise_to()

        return {
            'fold': to_call > 0,
            'check': to_call == 0,
            'call': to_call,
            'raise': {
                'min': state.min_completion_betting_or_raising_to_amount,
                'max': state.max_completion_betting_or_raising_to_amount,
                'is_bet': max(state.bets) == 0,
            } if can_raise else None,
        }

    def seat_view(self, seat: int, viewer: str | None) -> dict:
        info = self.seats[seat]

        if info is None:
            return {
                'name': 'Open seat', 'avatar': None, 'portrait': None, 'color': 'teal', 'kind': 'empty',
                'stack': 0, 'bet': 0, 'cards': None, 'has_cards': False, 'status': 'empty',
                'is_button': False, 'stats': [0, 0, 0], 'last_action': None, 'away': False,
            }

        index = self.index_of(seat)
        in_hand = index is not None and self.dealt.get(seat) is info
        live = in_hand and not self.hand_over
        mine = info.token is not None and info.token == viewer
        show_cards = in_hand and (mine or seat in self.revealed)

        if not in_hand:
            status = 'out' if info.stack == 0 else 'waiting'
        elif seat in self.folded:
            status = 'folded'
        elif live and self.state.stacks[index] == 0:
            status = 'all-in'
        else:
            status = 'active'

        player = self.players.get(info.token) if info.token else None

        return {
            'name': info.name,
            'avatar': info.avatar,
            'portrait': info.portrait,
            'color': info.color,
            'kind': 'bot' if info.bot is not None else 'human',
            'stack': self.state.stacks[index] if live else info.stack,
            'bet': self.state.bets[index] if live else 0,
            'cards': [card_code(c) for c in self.hole[seat]] if show_cards else None,
            'has_cards': in_hand and seat not in self.folded,
            'status': status,
            'is_button': seat == self.button and self.state is not None,
            'stats': info.stats.summary(),
            'last_action': self.last_action.get(seat),
            'away': info.bot is None and (player is None or player.left or not self.connected(info.token)),
        }

    def room_view(self, viewer: str | None) -> dict:
        def member(token: str | None, seat: Seat | None) -> dict | None:
            if seat is None:
                return None

            return {
                'name': seat.name,
                'avatar': seat.avatar,
                'portrait': seat.portrait,
                'color': seat.color,
                'kind': 'bot' if seat.bot is not None else 'human',
                'host': token is not None and token == self.host,
                'you': token is not None and token == viewer,
                'away': seat.bot is None and not self.connected(token),
            }

        waiting = [
            {'name': p.name, 'you': p.token == viewer, 'host': p.token == self.host}
            for p in self.present() if p.seat is None
        ]

        return {
            'code': self.code,
            'status': self.status,
            'mode': self.mode,
            'difficulty': self.difficulty,
            'host': self.players[self.host].name if self.host in self.players else None,
            'seats': [member(seat.token if seat else None, seat) for seat in self.seats],
            'min_seats': MIN_SEATS,
            'max_seats': MAX_SEATS,
            'waiting': waiting,
        }

    def view_for(self, token: str | None) -> dict:
        player = self.players.get(token) if token else None
        me = player.seat if player is not None and not player.left else None
        seat = self.seats[me] if me is not None else None

        return {
            'type': 'state',
            'room': self.room_view(token),
            'you': {
                'name': player.name if player else None,
                'host': token is not None and token == self.host,
                'seated': me is not None,
                'can_rebuy': seat is not None and seat.stack == 0 and not self.in_live_hand(me),
                'chat_room': self.chat_room if player is not None and not player.left else None,
            },
            'game': self.game_view(token, me) if self.status == 'playing' else None,
        }

    def game_view(self, token: str | None, me: int | None) -> dict:
        rot = me if me is not None else 0

        def turn(seat: int | None) -> int | None:
            return None if seat is None else (seat - rot) % len(self.seats)

        state = self.state
        live = not self.hand_over
        result = None

        if self.result is not None:
            winners = [
                {key: value for key, value in w.items() if key != 'token'} | {'seat': turn(w['seat'])}
                for w in self.result['winners']
            ]
            top = self.result['winners'][0] if winners else None
            headline = self.result['headline']

            if top is not None and token is not None and top['token'] == token:
                headline = f'You win {money(top["net"])}!'

            result = {
                'headline': headline,
                'winners': winners,
                'showdown': self.result['showdown'],
                'my_net': self.result['nets'].get(token) if token else None,
            }

        funded = self.funded()
        my_hand = None

        if me is not None and self.dealt.get(me) is self.seats[me] and me not in self.folded:
            my_hand = hand_label(self.hole[me], self.board)

        return {
            'mode': self.mode,
            'difficulty': self.difficulty,
            'hand_number': self.hand_number,
            'blinds': [SMALL_BLIND, BIG_BLIND],
            'me': 0 if me is not None else None,
            'seats': [self.seat_view((i + rot) % len(self.seats), token) for i in range(len(self.seats))],
            'board': [card_code(c) for c in self.board],
            'my_hand': my_hand,
            'street': STREET_NAMES[state.street_index] if live and state.street_index is not None else 'Showdown',
            'pot': state.total_pot_amount if live else 0,
            'to_act': turn(self.actor_seat),
            'legal': self.legal_actions(me) if me is not None else None,
            'log': list(self.log),
            'hand_over': self.hand_over,
            'result': result,
            'paused': self.paused,
            'champion': self.seats[funded[0]].name if self.paused and len(funded) == 1 else None,
            'clock': max(0.0, self.deadline - time.time()) if self.deadline else None,
            'clock_total': self.clock_total,
        }
