from preprocess import DeduplicationStrategy, NoiseRemovalStrategy, TranslationStrategy

class PreprocessorFactory:
    @staticmethod
    def create_preprocessor(preprocessor_type: str):
        if preprocessor_type == "deduplication":
            return DeduplicationStrategy()
        elif preprocessor_type == "noise_removal":
            return NoiseRemovalStrategy()
        elif preprocessor_type == "translation":
            return TranslationStrategy()
        else:
            raise ValueError(f"Unknown preprocessor type: {preprocessor_type}")