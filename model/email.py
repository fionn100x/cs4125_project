import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Email:
    content: str
    summary: str
    ticket_id: Optional[int] = None
    email_type: Optional[str] = None
    
    def to_features(self) -> np.ndarray:
        # placeholder implementation
        # will be replaced with the implemetation after i deal with the embeddings.
        return np.array([0.0])