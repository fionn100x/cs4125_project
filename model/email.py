import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Email:
    content: str
    summary: str
    ticket_id: Optional[int] = None
    email_type: Optional[str] = None

    def to_features(self, vectorizer) -> np.ndarray:
        """
        Converts the email content and summary into a feature vector using the given vectorizer.
        """
        combined_text = f"{self.content} {self.summary}"
        features = vectorizer.transform([combined_text])
        return features.toarray()[0]  # Return as a dense NumPy array
