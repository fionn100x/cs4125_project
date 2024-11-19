# email.py

import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class Email:
    content: str
    summary: str
    ticket_id: Optional[int] = None
    email_type: Optional[str] = None

    def to_features(self, vectorizer: TfidfVectorizer) -> np.ndarray:
        # Combine content and summary
        text = self.content + " " + self.summary
        # Use the provided vectorizer to transform text
        features = vectorizer.transform([text]).toarray()
        return features.flatten()