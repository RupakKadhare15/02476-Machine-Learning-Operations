import os
from locust import HttpUser, task, between, LoadTestShape

class ToxicCommentUser(HttpUser):
    wait_time = between(1, 3)  
    @task(3)
    def predict_non_toxic(self):
        """Simulate and Test normal usage patterns."""
        self.client.post(
            "/predict", 
            json={"text": "This project is fantastic and the team is great!"},
            name="/predict (Non-Toxic)"
        )

    @task(1)
    def predict_toxic(self):
        """Test model behavior on 'bad' inputs."""
        self.client.post(
            "/predict", 
            json={"text": "You are terrible and this is a waste of time."},
            name="/predict (Toxic)"
        )

    @task(1)
    def health_check(self):
        """Ensures the service remains responsive even under load."""
        self.client.get("/health", name="/health")


class DynamicShape(LoadTestShape):
    # This reads 'TEST_TYPE' from your terminal command
    test_type = os.getenv("TEST_TYPE", "normal").lower()

    def tick(self):
        """Method allows the script to toggle between stress, spike, and 
        normal traffic patterns"""
        run_time = self.get_run_time()

        if self.test_type == "stress":
            stages = [
                {"duration": 60,  "users": 10,  "spawn_rate": 10}, 
                {"duration": 180, "users": 100, "spawn_rate": 1},   
                {"duration": 240, "users": 100, "spawn_rate": 10},
            ]
        elif self.test_type == "spike":
            stages = [
                {"duration": 10, "users": 1,  "spawn_rate": 1},
                {"duration": 20, "users": 50, "spawn_rate": 50},
                {"duration": 60, "users": 50, "spawn_rate": 50},
                {"duration": 70, "users": 1,  "spawn_rate": 50},
            ]
        else: 
            stages = [{"duration": 300, "users": 5, "spawn_rate": 1}]

        for stage in stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])
        return None
