import ray
import time

@ray.remote
class MessageActor:
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def get_and_clear(self):
        out = self.messages
        self.messages = []
        return out
    
@ray.remote
def worker(message_actor, worker_no):
    for i in range(20):
        time.sleep(1)
        message_actor.add_message.remote(f"Zprava od {worker_no} cislo {i}")

message_actor = MessageActor.remote()

for worker_no in range(3):
    worker.remote(message_actor, worker_no)

for _ in range(100):
    new_messages = ray.get(message_actor.get_and_clear.remote())
    print(f"Nove zpravy: {new_messages}")
    time.sleep(1)