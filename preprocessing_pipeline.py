class PreprocessingPipeline:
    def __init__(self, strategies):
        """
        Initialize with a list of preprocessing strategies.
        :param strategies: List of PreprocessingStrategy objects.
        """
        self.strategies = strategies

    def process(self, df):
        """Apply all preprocessing steps in sequence."""
        for strategy in self.strategies:
            df = strategy.process(df)
        return df
