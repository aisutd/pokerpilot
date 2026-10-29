import json
import random
import unittest

from engine.room import MAX_SEATS, SEAT_COUNT, STARTING_STACK, GameError, Room


class FakeSocket:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def send(self, text: str) -> None:
        self.messages.append(json.loads(text))

    @property
    def last(self) -> dict:
        return self.messages[-1]


def make_room(names: list[str], mode: str = 'pvp', difficulty: str = 'easy', seed: int = 1):
    room = Room('TEST', seed=seed, timers=False)
    tokens = [room.join(name) for name in names]
    sockets = {}

    for token in tokens:
        sockets[token] = FakeSocket()
        room.connect(token, sockets[token])

    room.update_settings(tokens[0], mode, difficulty)
    return room, tokens, sockets


def total_chips(room: Room) -> int:
    if room.hand_over:
        return sum(seat.stack for seat in room.seats if seat is not None)

    in_hand = sum(room.state.stacks) + room.state.total_pot_amount
    others = sum(seat.stack for i, seat in enumerate(room.seats) if seat is not None and i not in room.order)
    return in_hand + others


def actor_token(room: Room) -> str | None:
    seat = room.actor_seat
    return None if seat is None else room.seats[seat].token


def random_move(room: Room, token: str, rng: random.Random) -> None:
    legal = room.legal_actions(room.players[token].seat)
    options = ['check_call', 'check_call'] + (['fold'] if legal['fold'] else []) + (['raise'] if legal['raise'] else [])
    action = rng.choice(options)
    amount = None

    if action == 'raise':
        low, high = legal['raise']['min'], legal['raise']['max']
        amount = rng.choice([low, high, rng.randint(low, high)])

    room.act(token, action, amount)


class RoomTest(unittest.TestCase):
    def test_room_member_is_bound_to_authenticated_account(self) -> None:
        room = Room('AUTH', timers=False)
        token = room.join('Ann', account_id='user-123')
        self.assertEqual(room.players[token].account_id, 'user-123')
        self.assertNotEqual(token, 'user-123')

    def check_views(self, room: Room, sockets: dict[str, FakeSocket]) -> None:
        for token, ws in sockets.items():
            if token not in room.players:
                continue

            game = ws.last['game']
            player = room.players[token]
            self.assertEqual(ws.last['you']['name'], player.name)

            if player.seat is not None:
                self.assertEqual(game['me'], 0)
                self.assertEqual(game['seats'][0]['name'], room.seats[player.seat].name)

            if not game['hand_over']:
                for i, seat in enumerate(game['seats']):
                    if i != game['me']:
                        self.assertIsNone(seat['cards'], 'someone else\'s cards leaked mid-hand')

            legal = game['legal'] is not None
            self.assertEqual(legal, actor_token(room) == token)

    def play(self, room, tokens, sockets, hands: int, seed: int) -> None:
        rng = random.Random(seed)
        start = total_chips(room)

        while room.hand_number <= hands and not room.paused:
            token = actor_token(room)

            if token is not None:
                self.check_views(room, sockets)
                random_move(room, token, rng)
            else:
                self.assertTrue(room.run_pending(), 'the room stalled')

            self.assertEqual(total_chips(room), start)

    def test_pvp_three_players(self) -> None:
        room, tokens, sockets = make_room(['Ann', 'Bob', 'Cy'])
        room.start_game(tokens[0])
        self.assertEqual(sum(1 for seat in room.seats if seat is not None), 3)
        self.play(room, tokens, sockets, hands=25, seed=2)
        self.assertTrue(all(seat is None or seat.bot is None for seat in room.seats))

    def test_pve_fills_seats_with_ai(self) -> None:
        for difficulty in ('easy', 'medium', 'hard'):
            with self.subTest(difficulty=difficulty):
                room, tokens, sockets = make_room(['Ann', 'Bob'], 'pve', difficulty)
                room.start_game(tokens[0])
                bots = [seat for seat in room.seats if seat is not None and seat.bot is not None]
                self.assertEqual(len(bots), SEAT_COUNT - 2)
                self.play(room, tokens, sockets, hands=6, seed=3)

    def test_pvp_needs_two_players_and_only_host_starts(self) -> None:
        room, tokens, _ = make_room(['Ann'])

        with self.assertRaises(GameError):
            room.start_game(tokens[0])

        bob = room.join('Bob')

        with self.assertRaises(GameError):
            room.start_game(bob)

        with self.assertRaises(GameError):
            room.update_settings(bob, 'pve', 'hard')

        room.start_game(tokens[0])
        self.assertEqual(room.status, 'playing')

    def test_host_can_add_and_remove_pvp_seats(self) -> None:
        room, tokens, sockets = make_room(['Ann', 'Bob', 'Cy'])
        room.resize(tokens[0], 3)
        self.assertEqual(len(room.seats), 3)
        self.assertEqual(len(sockets[tokens[1]].last['room']['seats']), 3)

        with self.assertRaises(GameError):
            room.resize(tokens[0], 2)  # every seat is taken

        with self.assertRaises(GameError):
            room.resize(tokens[1], 4)  # not the host

        with self.assertRaises(GameError):
            room.resize(tokens[0], MAX_SEATS + 1)

        dee = room.join('Dee')
        self.assertIsNone(room.players[dee].seat)
        room.resize(tokens[0], 4)
        self.assertEqual(room.players[dee].seat, 3)  # the new seat goes to whoever was waiting

        room.leave(tokens[1])  # frees seat 1; later seats move down when it is removed
        room.resize(tokens[0], 3)
        self.assertEqual([seat.name for seat in room.seats], ['Ann', 'Cy', 'Dee'])
        self.assertEqual([room.players[t].seat for t in (tokens[0], tokens[2], dee)], [0, 1, 2])

        room.update_settings(tokens[0], 'pve', 'easy')
        self.assertEqual(len(room.seats), SEAT_COUNT)

        with self.assertRaises(GameError):
            room.resize(tokens[0], 4)

    def test_seats_change_mid_game(self) -> None:
        room, tokens, sockets = make_room(['Ann', 'Bob', 'Cy'])
        room.resize(tokens[0], 4)
        room.start_game(tokens[0])
        room.resize(tokens[0], 5)  # adding works mid-hand
        self.assertEqual(len(room.seats), 5)

        with self.assertRaises(GameError):
            room.resize(tokens[0], 3)  # removing waits for the hand to end

        rng = random.Random(6)

        while not room.hand_over:
            random_move(room, actor_token(room), rng)

        room.resize(tokens[0], 3)
        self.assertEqual(len(room.seats), 3)
        self.assertTrue(all(seat is not None for seat in room.seats))
        self.play(room, tokens, sockets, hands=8, seed=7)

    def test_ten_player_table(self) -> None:
        names = [f'P{i}' for i in range(MAX_SEATS)]
        room, tokens, sockets = make_room(names[:SEAT_COUNT])
        room.resize(tokens[0], MAX_SEATS)

        for name in names[SEAT_COUNT:]:
            token = room.join(name)
            tokens.append(token)
            sockets[token] = FakeSocket()
            room.connect(token, sockets[token])

        self.assertNotIn(None, room.seats)
        eleventh = room.join('Late')
        self.assertIsNone(room.players[eleventh].seat)  # the table is full
        room.start_game(tokens[0])
        self.assertEqual(len(room.order), MAX_SEATS)
        self.play(room, tokens, sockets, hands=12, seed=8)

        # Switching a big room to PvE keeps everyone seated.
        room, tokens, _ = make_room(names[:8])
        room.resize(tokens[0], 9)
        room.update_settings(tokens[0], 'pve', 'easy')
        self.assertEqual(len(room.seats), 8)
        room.start_game(tokens[0])
        self.assertTrue(all(seat is not None and seat.bot is None for seat in room.seats))

    def test_each_room_has_a_private_chat(self) -> None:
        room, tokens, sockets = make_room(['Ann', 'Bob'])
        other, _, other_sockets = make_room(['Cy'])
        chat = sockets[tokens[0]].last['you']['chat_room']
        self.assertGreaterEqual(len(chat), 20)  # chat-api accepts 20-64 [A-Za-z0-9_-]
        self.assertRegex(chat, r'^[A-Za-z0-9_-]{20,64}$')
        self.assertEqual(sockets[tokens[1]].last['you']['chat_room'], chat)
        self.assertNotEqual(other_sockets[next(iter(other_sockets))].last['you']['chat_room'], chat)
        self.assertNotIn(chat, json.dumps(sockets[tokens[0]].last['room']))  # not in shared room info
        self.assertIsNone(room.view_for('not-a-member')['you']['chat_room'])
        room.leave(tokens[1])
        self.assertIsNone(room.view_for(tokens[1])['you']['chat_room'])

    def test_duplicate_names_are_renamed(self) -> None:
        room, _, _ = make_room(['Ann'])
        room.join('ann')
        room.join('MathNinja')
        names = [p.name for p in room.players.values()]
        self.assertEqual(names, ['Ann', 'ann2', 'MathNinja2'])

        with self.assertRaises(GameError):
            room.join('   ')

    def test_everyone_gets_their_own_portrait(self) -> None:
        room, tokens, sockets = make_room(['Ann', 'Bob'], 'pve')
        room.start_game(tokens[0])
        seats = sockets[tokens[0]].last['game']['seats']
        faces = [seat['portrait'] for seat in seats]
        self.assertEqual(len(set(faces)), SEAT_COUNT)
        self.assertTrue(all(face in range(1, 11) for face in faces))
        self.assertEqual(faces[:2], [1, 2])  # people first get faces no AI player uses
        self.assertEqual([m['portrait'] for m in sockets[tokens[1]].last['room']['seats']],
                         [room.seats[i].portrait for i in range(SEAT_COUNT)])

        # A full ten-seat PvP table still has ten different faces.
        names = [f'P{i}' for i in range(MAX_SEATS)]
        room, tokens, _ = make_room(names[:SEAT_COUNT])
        room.resize(tokens[0], MAX_SEATS)
        for name in names[SEAT_COUNT:]:
            room.join(name)
        self.assertEqual(sorted(seat.portrait for seat in room.seats), list(range(1, 11)))

    def test_only_the_actor_can_act(self) -> None:
        room, tokens, _ = make_room(['Ann', 'Bob', 'Cy'])
        room.start_game(tokens[0])
        actor = actor_token(room)
        other = next(t for t in tokens if t != actor)

        with self.assertRaises(GameError):
            room.act(other, 'check_call')

        with self.assertRaises(GameError):
            room.act(actor, 'raise', float('nan'))

        with self.assertRaises(GameError):
            room.act(actor, 'dance')

    def test_timeout_folds_and_away_players_are_skipped(self) -> None:
        room, tokens, sockets = make_room(['Ann', 'Bob'])
        room.start_game(tokens[0])
        actor = actor_token(room)
        room.disconnect(sockets[actor])
        hand = room.hand_number
        room.run_pending()  # the away clock acts for them
        self.assertNotEqual(actor_token(room), actor)
        self.assertTrue(room.hand_number == hand or room.hand_over)

    def test_leaving_mid_hand_frees_the_seat_after_the_hand(self) -> None:
        room, tokens, sockets = make_room(['Ann', 'Bob', 'Cy'])
        room.start_game(tokens[0])
        leaver = tokens[2]
        seat = room.players[leaver].seat
        room.leave(leaver)
        self.assertIsNotNone(room.seats[seat])
        rng = random.Random(4)

        while not room.hand_over:
            token = actor_token(room)

            if token is None or room.players[token].left:
                room.run_pending()
            else:
                random_move(room, token, rng)

        room.run_pending()  # deal the next hand
        self.assertIsNone(room.seats[seat])
        self.assertNotIn(leaver, room.players)

    def test_host_leaving_hands_over_host(self) -> None:
        room, tokens, _ = make_room(['Ann', 'Bob'])
        room.leave(tokens[0])
        self.assertEqual(room.host, tokens[1])
        room.leave(tokens[1])
        self.assertTrue(room.empty)

    def test_late_joiner_replaces_an_ai_player(self) -> None:
        room, tokens, sockets = make_room(['Ann'], 'pve')
        room.start_game(tokens[0])
        late = room.join('Dee')
        self.assertIsNone(room.players[late].seat)
        self.assertEqual(sockets[tokens[0]].last['room']['waiting'][0]['name'], 'Dee')
        rng = random.Random(5)

        while room.hand_number < 2:
            token = actor_token(room)
            random_move(room, token, rng) if token else room.run_pending()

        seat = room.players[late].seat
        self.assertIsNotNone(seat)
        self.assertEqual(room.seats[seat].token, late)
        self.assertIn(seat, room.order)  # dealt into the next hand

    def test_bust_pause_rebuy_and_restart(self) -> None:
        room, tokens, sockets = make_room(['Ann', 'Bob'])
        room.start_game(tokens[0])

        while not room.paused:
            token = actor_token(room)

            if token is None:
                room.run_pending()
                continue

            legal = room.legal_actions(room.players[token].seat)
            room.act(token, 'raise', legal['raise']['max']) if legal['raise'] else room.act(token, 'check_call')

        loser = next(t for t in tokens if room.seats[room.players[t].seat].stack == 0)
        winner = next(t for t in tokens if t != loser)
        view = sockets[loser].last
        self.assertTrue(view['you']['can_rebuy'])
        self.assertEqual(view['game']['champion'], room.players[winner].name)

        with self.assertRaises(GameError):
            room.rebuy(winner)

        room.rebuy(loser)
        self.assertFalse(room.paused)
        hand = room.hand_number
        room.run_pending()
        self.assertEqual(room.hand_number, hand + 1)
        self.assertEqual(total_chips(room), 3 * STARTING_STACK)


if __name__ == '__main__':
    unittest.main()
