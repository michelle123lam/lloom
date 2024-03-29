# Concept induction concept functions
# =================================================

# Imports
import uuid

# CONCEPT class ================================
class Concept:
    def __init__(self, name, prompt, example_ids, active, summary=None):
        concept_id = str(uuid.uuid4())
        self.id = concept_id
        self.name = name
        self.prompt = prompt
        self.example_ids = example_ids
        self.active = active
        self.summary = summary

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "example_ids": list(self.example_ids),
            "active": self.active,
            "summary": self.summary
        }
