class random:
    @staticmethod
    def default_rng(seed=None):
        import random as _random
        rng = _random.Random(seed)
        return rng
