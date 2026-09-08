from locust import HttpUser, between, task


class SentimentUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict_sentiment(self):
        payload = {
            "request": {
                "text": "المنتج ممتاز جدا وأنصح به"
            }
        }

        with self.client.post(
            "/predict",
            json=payload,
            name="/predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"Unexpected status code: {response.status_code}"
                )
                return

            body = response.json()

            required_keys = {
                "label",
                "confidence",
                "model_version",
            }

            if not required_keys.issubset(body):
                response.failure(
                    f"Missing keys in response: {body}"
                )