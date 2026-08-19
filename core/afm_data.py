import numpy as np

class AFMData:
    def __init__(self, stack, meta):
        self.stack = stack          # stack original (NanoLocz)
        self.meta = meta            # metadatos JPK

        # Se llenan después
        self.stack_ecc1 = None
        self.stack_drift = None
        self.stack_ecc2 = None

        self.masks = {}
        self.drift_vectors = None
        self.kymos = []
