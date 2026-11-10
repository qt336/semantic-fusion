"""Offline regression checks; no network requests or API credentials are used."""
import unittest
from unittest.mock import patch
import requests

from llm.common import make_prompt
from llm.run import PROVIDERS, make_payload, run_one
from llm.scoring import proposed_answer, outcome


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.item = dict(sample_id='example', language='en', group='semantic', a='ice', b='cream', c='lake',
                         prompt=make_prompt('ice', 'cream', 'lake', 'en', 'semantic'))

    def test_first_answer_not_first_output_token(self):
        cases = [('The word is cream', 'B'), ('ice. Actually, cream', 'A'),
                 ('cream. Actually, ice', 'B'), ('ice cream', 'A'),
                 ('ice cream lake. The word immediately before "lake" is cream.', 'B'),
                 (self.item['prompt'], 'Other'), ('', 'Other')]
        for text, label in cases:
            with self.subTest(text=text):
                answer, _ = proposed_answer(dict(self.item, answer_raw=text))
                self.assertEqual(outcome(self.item, answer), label)

    def test_exact_request_and_thinking_controls(self):
        for model, config in PROVIDERS.items():
            body = make_payload(model, self.item)
            self.assertEqual(body['messages'], [{'role': 'user', 'content': self.item['prompt']}])
            self.assertEqual(body['temperature'], 0)
            self.assertEqual(body['max_tokens'], 32)
            for key, value in config['extra'].items():
                self.assertEqual(body[key], value)

    @patch('llm.run.requests.post')
    def test_wrong_answer_is_retained_without_retry(self, post):
        post.return_value.status_code = 200
        post.return_value.json.return_value = dict(model='kimi-k3', choices=[dict(message=dict(content='ice'))])
        row = run_one('kimi', self.item, 'test-placeholder', 'https://example.invalid/v1')
        self.assertEqual(row['answer_raw'], 'ice')
        self.assertEqual(row['status'], 'completed')
        self.assertEqual(post.call_count, 1)

    @patch('llm.run.requests.post')
    def test_reasoning_response_is_flagged(self, post):
        post.return_value.status_code = 200
        post.return_value.json.return_value = dict(model='kimi-k3', choices=[dict(message=dict(content='cream', reasoning_content='reason'))])
        row = run_one('kimi', self.item, 'test-placeholder', 'https://example.invalid/v1')
        self.assertEqual(row['status'], 'invalid_response')
        self.assertEqual(post.call_count, 1)

    @patch('llm.run.time.sleep')
    @patch('llm.run.requests.post')
    def test_malformed_success_is_not_retried(self, post, sleep):
        post.return_value.status_code = 200
        post.return_value.json.side_effect = requests.exceptions.JSONDecodeError('invalid JSON', 'x', 0)
        row = run_one('kimi', self.item, 'test-placeholder', 'https://example.invalid/v1')
        self.assertEqual(row['status'], 'invalid_response')
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    @patch('llm.run.requests.post')
    def test_malformed_message_does_not_abort_batch(self, post):
        post.return_value.status_code = 200
        post.return_value.json.return_value = dict(choices=[dict(message=None)])
        row = run_one('kimi', self.item, 'test-placeholder', 'https://example.invalid/v1')
        self.assertEqual(row['status'], 'invalid_response')
        self.assertEqual(post.call_count, 1)


if __name__ == '__main__':
    unittest.main()
