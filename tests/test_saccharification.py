import unittest

from web_product import app


class SaccharificationFeatureTest(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_page_exposes_complete_saccharification_process(self):
        response = self.client.get("/saccharification")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("糖化过程智能监测", html)
        self.assertIn(">糖化</a>", html)
        for step in ("蒸煮糊化", "降温拌曲", "酶解糖化", "还原糖释放", "糖化完成"):
            self.assertIn(step, html)
        for control_id in ("raw_material", "saccharification_hours", "saccharification_temperature"):
            self.assertIn(f'id="{control_id}"', html)

    def test_api_returns_trajectory_and_process_guidance(self):
        response = self.client.get(
            "/api/saccharification?hours=1&temperature=60&raw_material=糯米"
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertGreater(payload["output"]["reducing_sugar"], 2.5)
        self.assertGreater(len(payload["trajectory"]["time"]), 20)
        self.assertEqual(len(payload["trajectory"]["time"]), len(payload["trajectory"]["reducing_sugar"]))
        self.assertEqual(payload["guidance"]["level"], "good")

    def test_api_warns_when_temperature_is_too_high(self):
        response = self.client.get(
            "/api/saccharification?hours=1&temperature=70&raw_material=糯米"
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["guidance"]["level"], "warning")
        self.assertTrue(any("降温" in item for item in payload["guidance"]["recommendations"]))


if __name__ == "__main__":
    unittest.main()
