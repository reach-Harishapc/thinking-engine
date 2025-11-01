# core/learning_manager.py
"""
Lightweight manager that coordinates policy trainer and selector persistence.
"""

class LearningManager:
    def __init__(self):
        # Placeholder for future implementation
        pass

    def offline_train(self, episodes, epochs=1):
        # Placeholder
        print(f"Offline training with {len(episodes)} episodes, {epochs} epochs")

    def online_feedback(self, candidates, prompt_tokens, chosen, reward):
        # Placeholder
        print(f"Online feedback: chosen={chosen}, reward={reward}")

    def learn(self, data):
        # Simple learning method - placeholder
        print(f"Learning from {len(data)} data points")
        # Could implement actual learning here
        pass

    def export_state(self):
        # Export learning state
        return {}

    def import_state(self, state):
        # Import learning state
        pass
