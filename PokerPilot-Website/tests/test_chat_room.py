import time
import unittest

from app import app, match_chat_rooms, matches


class ChatRoomTest(unittest.TestCase):
    def setUp(self) -> None:
        self.first_match = f'first-{id(self)}'
        self.second_match = f'second-{id(self)}'
        self.first_client = app.test_client()
        self.second_client = app.test_client()
        with self.first_client.session_transaction() as flask_session:
            flask_session['match_id'] = self.first_match
        with self.second_client.session_transaction() as flask_session:
            flask_session['match_id'] = self.second_match
        matches[self.first_match] = (None, time.time())
        matches[self.second_match] = (None, time.time())

    def tearDown(self) -> None:
        with app.app_context():
            matches.pop(self.first_match, None)
            matches.pop(self.second_match, None)
            match_chat_rooms.pop(self.first_match, None)
            match_chat_rooms.pop(self.second_match, None)

    def test_each_game_gets_its_own_stable_chat_room(self) -> None:
        first_room = self.first_client.get('/api/chat-room').get_json()['room']
        repeated_room = self.first_client.get('/api/chat-room').get_json()['room']
        second_room = self.second_client.get('/api/chat-room').get_json()['room']

        self.assertEqual(first_room, repeated_room)
        self.assertNotEqual(first_room, second_room)
        self.assertGreaterEqual(len(first_room), 20)

    def test_chat_room_requires_an_active_game(self) -> None:
        client = app.test_client()
        response = client.get('/api/chat-room')

        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.get_json())


if __name__ == '__main__':
    unittest.main()
