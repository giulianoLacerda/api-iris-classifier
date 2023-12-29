from locust import HttpUser, task, between


class SimpleLocustTest(HttpUser):
    wait_time = between(1, 3)

    @task
    def post_classifier(self):
        url = "https://localhost/v1/classify"
        payload_iris = {
            "sepal_length": 7.7,
            "sepal_width": 2.6,
            "petal_length": 6.9,
            "petal_width": 2.3
        }
        self.client.post(url, json=payload_iris)
