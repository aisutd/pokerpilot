"""A 6-max No-Limit Hold'em table: one human against five AI players, run on
pokerkit. All amounts are integer cents."""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field

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
SEAT_COUNT = 6
HUMAN_SEAT = 0
STREET_NAMES = ('Preflop', 'Flop', 'Turn', 'River')


class GameError(Exception):
    """An illegal request from the client."""


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
class Seat:
    name: str
    avatar: str | None
    color: str
    bot: Bot | None
    stack: int = STARTING_STACK
    stats: PlayerStats = field(default_factory=PlayerStats)


class Table:
    def __init__(self, difficulty: str, seed: int | None = None) -> None:
        if difficulty not in DIFFICULTIES:
            raise GameError(f'Unknown difficulty: {difficulty!r}')

        self.lock = threading.Lock()
        self.rng = random.Random(seed)
        self.difficulty = difficulty
        self.seats = [Seat('You', None, 'teal', None)]

        for character in CHARACTERS:
            bot = create_bot(difficulty, character, random.Random(self.rng.random()))
            self.seats.append(Seat(character.name, character.avatar, character.color, bot))

        self.hand_number = 0
        self.button: int | None = None
        self.state: State | None = None
        self.result: dict | None = None
        self.frames: list[dict] = []
        self.start_hand()

    # ------------------------------------------------------------- status

    @property
    def hand_over(self) -> bool:
        return self.state is not None and not self.state.status

    @property
    def game_over(self) -> bool:
        if not self.hand_over:
            return False

        human_busted = self.seats[HUMAN_SEAT].stack == 0
        bots_left = sum(1 for seat in self.seats[1:] if seat.stack > 0)

        return human_busted or bots_left == 0

    @property
    def board(self) -> list[Card]:
        return list(self.state.get_board_cards(0))

    def index_of(self, seat: int) -> int | None:
        return self.order.index(seat) if seat in self.order else None

    @property
    def actor_seat(self) -> int | None:
        if not self.state.status or self.state.actor_index is None:
            return None

        return self.order[self.state.actor_index]

    # -------------------------------------------------------------- setup

    def next_active(self, seat: int) -> int:
        for step in range(1, SEAT_COUNT + 1):
            candidate = (seat + step) % SEAT_COUNT

            if self.seats[candidate].stack > 0:
                return candidate

        raise GameError('No players left.')

    def start_hand(self) -> None:
        if self.state is not None:
            if not self.hand_over:
                raise GameError('The current hand is still in progress.')

            if self.game_over:
                raise GameError('The game is over. Start a new game.')

        active = [i for i, seat in enumerate(self.seats) if seat.stack > 0]

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
        self.revealed: set[int] = set()
        self.folded: set[int] = set()
        self.result = None
        self.op_cursor = 0
        self.board_dealt = 0
        self.preflop_action = {s: 'none' for s in order}
        self.postflop_raises = {s: 0 for s in order}
        self.preflop_raiser: int | None = None
        self.raises_this_street = 0
        self.last_action: dict[int, str] = {}
        self.vpip_seen: set[int] = set()
        self.pfr_seen: set[int] = set()

        for s in order:
            self.seats[s].stats.hands += 1

        role = {self.button: 'the button'}
        small_blind_seat = order[-1] if len(order) == 2 else order[0]
        role.setdefault(small_blind_seat, 'the small blind')
        role.setdefault(order[1] if len(order) > 2 else order[0], 'the big blind')
        where = role.get(HUMAN_SEAT, 'in middle position')
        self.log = [f'Hand #{self.hand_number} - you are {where}.']
        self.frames = []
        self.sync()
        self.snapshot()
        self.run_bots()

    # ------------------------------------------------------------ logging

    def describe_action(self, seat: int, action: str, amount: int | None) -> str:
        """Phrase an action before it is applied, and remember it as the
        seat's latest action for display."""

        state = self.state
        name = self.seats[seat].name
        you = seat == HUMAN_SEAT
        s = '' if you else 's'

        if action == 'fold':
            short = 'Fold'
            text = f'{name} fold{s}'
        elif action == 'check_call':
            to_call = state.checking_or_calling_amount

            if to_call:
                all_in = to_call == state.stacks[state.actor_index]
                short = 'All-in' if all_in else f'Call {money(to_call)}'
                text = f'{name} call{s} {money(to_call)}' + (' (all-in)' if all_in else '')
            else:
                short = 'Check'
                text = f'{name} check{s}'
        else:
            is_bet = max(state.bets) == 0
            all_in = amount == state.max_completion_betting_or_raising_to_amount
            verb = 'Bet' if is_bet else 'Raise'
            short = 'All-in' if all_in else f'{verb} {money(amount)}'
            phrase = (f'bet{s}' if is_bet else f'raise{s} to')
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
                s = '' if seat == HUMAN_SEAT else 's'
                blinds.append((kind != 'small', f'{self.seats[seat].name} post{s} the {kind} blind ({money(op.amount)}).'))
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
                s = '' if seat == HUMAN_SEAT else 's'

                if op.hole_cards:
                    self.revealed.add(seat)
                    cards = ' '.join(card_code(card) for card in op.hole_cards)
                    self.log.append(f'{name} show{s} {cards}.')
                else:
                    self.log.append(f'{name} muck{s}.')
            elif isinstance(op, ChipsPulling):
                seat = self.order[op.player_index]
                s = '' if seat == HUMAN_SEAT else 's'
                self.log.append(f'{self.seats[seat].name} collect{s} {money(op.amount)}.')

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
                    'name': self.seats[seat].name,
                    'net': net,
                    'hand': hand,
                    'best': best,
                })

        winners.sort(key=lambda w: -w['net'])
        human_net = state.payoffs[self.index_of(HUMAN_SEAT)] if HUMAN_SEAT in self.order else 0

        if not winners:
            headline = 'Split pot.'
        elif winners[0]['seat'] == HUMAN_SEAT:
            headline = f'You win {money(winners[0]["net"])}!'
        else:
            top = winners[0]
            headline = f'{top["name"]} wins {money(top["net"])}'
            headline += f' with {top["hand"]}.' if top['hand'] else '.'

        self.result = {
            'headline': headline,
            'human_net': human_net,
            'winners': winners,
            'showdown': showdown,
        }

        for seat in self.order:
            if self.seats[seat].stack == 0:
                you = seat == HUMAN_SEAT
                self.log.append('You are out of chips.' if you else f'{self.seats[seat].name} is out.')

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
        self.snapshot()

    def snapshot(self) -> None:
        self.frames.append(self.view())

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

    def run_bots(self) -> None:
        while (seat := self.actor_seat) is not None and seat != HUMAN_SEAT:
            bot = self.seats[seat].bot
            action, amount = bot.decide(self.bot_view(seat))
            state = self.state

            # Never fold for free, and fall back to calling on bad sizes.
            if action == 'fold' and state.checking_or_calling_amount == 0:
                action = 'check_call'

            if action == 'raise' and (
                    amount is None or not state.can_complete_bet_or_raise_to(amount)
            ):
                action, amount = 'check_call', None

            self.apply(seat, action, amount if action == 'raise' else None)

    # ---------------------------------------------------------- public API

    def human_action(self, action: str, amount: int | None = None) -> None:
        if self.actor_seat != HUMAN_SEAT:
            raise GameError('It is not your turn.')

        self.frames = []
        self.apply(HUMAN_SEAT, action, amount if action == 'raise' else None)
        self.run_bots()

    def next_hand(self) -> None:
        self.start_hand()

    def legal_actions(self) -> dict | None:
        if self.actor_seat != HUMAN_SEAT:
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

    def seat_view(self, seat: int) -> dict:
        info = self.seats[seat]
        index = self.index_of(seat)
        in_hand = index is not None
        live = in_hand and self.state.status
        show_cards = in_hand and (seat == HUMAN_SEAT or seat in self.revealed)

        if not in_hand:
            status = 'out'
        elif seat in self.folded:
            status = 'folded'
        elif live and self.state.stacks[index] == 0:
            status = 'all-in'
        else:
            status = 'active'

        return {
            'name': info.name,
            'avatar': info.avatar,
            'color': info.color,
            'stack': self.state.stacks[index] if live else info.stack,
            'bet': self.state.bets[index] if live else 0,
            'cards': [card_code(c) for c in self.hole[seat]] if show_cards else None,
            'has_cards': in_hand and seat not in self.folded,
            'status': status,
            'is_button': seat == self.button,
            'stats': info.stats.summary(),
            'last_action': self.last_action.get(seat),
        }

    def view(self) -> dict:
        state = self.state
        street = state.street_index
        actor = self.actor_seat

        return {
            'difficulty': self.difficulty,
            'hand_number': self.hand_number,
            'blinds': [SMALL_BLIND, BIG_BLIND],
            'seats': [self.seat_view(seat) for seat in range(SEAT_COUNT)],
            'board': [card_code(c) for c in self.board],
            'human_hand': hand_label(self.hole.get(HUMAN_SEAT, []), self.board)
            if HUMAN_SEAT not in self.folded else None,
            'street': STREET_NAMES[street] if street is not None else 'Showdown',
            'pot': state.total_pot_amount if state.status else 0,
            'to_act': actor,
            'legal': self.legal_actions(),
            'log': list(self.log),
            'hand_over': self.hand_over,
            'result': self.result,
            'game_over': self.game_over,
            'human_won_game': self.game_over and self.seats[HUMAN_SEAT].stack > 0,
        }

    def response(self) -> dict:
        """The final view plus every intermediate frame for animation."""

        view = self.view()
        frames = self.frames[:-1] if self.frames else []

        return {'frames': frames, 'view': view}
