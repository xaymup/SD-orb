import random
import time

class VJPrompter:
    def __init__(self, prompt_list, interval=10):
        self.prompts = prompt_list
        self.interval = interval
        self.last_change = 0
        self.current_prompt = prompt_list[0] if prompt_list else ""

    def update(self):
        if time.time() - self.last_change > self.interval:
            self.current_prompt = random.choice(self.prompts)
            self.last_change = time.time()
            return True # Prompt changed
        return False

    def get_prompt(self):
        return self.current_prompt