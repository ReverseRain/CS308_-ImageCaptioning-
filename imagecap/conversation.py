from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .constants import DEFAULT_IMAGE_TOKEN

@dataclass
class Conversation:
    """A class for keeping track of conversation state."""
    system: str
    roles: List[str]
    messages: List[Dict[str, str]]
    offset: int
    
    def get_prompt(self):
        """Get the prompt for generation."""
        prompt = self.system + "\n"
        for message in self.messages:
            role = message["role"]
            content = message["content"]
            prompt += f"{role}: {content}\n"
        return prompt
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append({
            "role": role,
            "content": content
        })

def get_default_conv_template():
    """Get the default conversation template."""
    conv = Conversation(
        system="You are a helpful assistant that can generate image descriptions.",
        roles=["user", "assistant"],
        messages=[],
        offset=0
    )
    return conv 