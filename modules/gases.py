
class CarbonDioxide:

    def __init__(self, fraction=0.0004):
        self.mfp = 1e10 if fraction == 0 else 0.0004 / fraction

    def get_free_path(self):
        return self.mfp
