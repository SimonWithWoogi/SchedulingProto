class PreSetInfo:
    def __init__(self, M, P, maxQ, minQ, maxT, setuptime, maxalltime, capacity, distribution):
        if not is_number(M):
            raise ValueError("Machine number must be digits")
        if not is_number(P):
            raise ValueError("Product kinds must be digits")
        if not is_number(maxQ):
            raise ValueError("Max order Q must be digits")
        if not is_number(minQ):
            raise ValueError("Min order Q must be digits")
        if not is_number(maxT):
            raise ValueError("Max Due Date Time must be digits")
        if not is_number(setuptime):
            raise ValueError("Setup time must be digits")
        if not is_number(maxalltime):
            raise ValueError("Limitation Time must be digits")
        if not is_number(capacity):
            raise ValueError("Machine capacity must be digits")
        if not is_string(distribution):
            raise ValueError("Distribution must be string")

        self._M = M
        self._P = P
        self._MaxQ = maxQ
        self._MinQ = minQ
        self._MaxT = maxT
        self._SetUpT = setuptime
        self._LimT = maxalltime
        self._MCapa = capacity
        self._Distrib = distribution

    def MachinesNumber(self):
        return self._M
    def MachineCapa(self):
        return self._MCapa
    def ProductKinds(self):
        return self._P
    def MaxOrderQ(self):
        return self._MaxQ
    def MinOrderQ(self):
        return self._MinQ
    def MaxDueDate(self):
        return self._MaxT
    def SetUpTime(self):
        return self._SetUpT
    def LimitationTime(self):
        return self._LimT
    def Distribution(self):
        return self._Distrib

def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
def is_string(value):
    try:
        if value.isalpha():
            return True
        else:
            return False
    except ValueError:
        return False