from abc import ABC, abstractmethod

# Strategy interface
class PreprocessingStrategy(ABC):
    @abstractmethod
    def process(self, df):
        pass


# Concrete strategies
class DeduplicationStrategy(PreprocessingStrategy):
    def process(self, df):
        return df.drop_duplicates()


class NoiseRemovalStrategy(PreprocessingStrategy):
    def process(self, df):
        df['Interaction content'] = df['Interaction content'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
        df['Ticket Summary'] = df['Ticket Summary'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
        return df


class TranslationStrategy(PreprocessingStrategy):
    def process(self, df):
        df['Ticket Summary'] = translate_to_en(df['Ticket Summary'].tolist())
        return df


def translate_to_en(texts):
    return texts  # Stub for translation logic
