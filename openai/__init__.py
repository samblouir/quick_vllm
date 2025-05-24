class OpenAI:
    def __init__(self, *args, **kwargs):
        pass
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                raise RuntimeError('OpenAI stub not functional')
    def models(self):
        return type('models', (), {'list': lambda self: []})()
