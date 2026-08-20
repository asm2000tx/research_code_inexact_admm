class fista_const:
    def __init__(self, A, b, beta):
        self.A = A
        self.b = b
        self.beta = beta
    
    def prox_step(self):
        print("Calling proximal step ....")