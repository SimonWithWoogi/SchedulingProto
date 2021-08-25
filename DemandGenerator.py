from Parameters import PreSetInfo

def Initialize(M=None, P=None, maxQ=None, minQ=None, maxT=None,
               setuptime=None, maxalltime=None, capacity=None,
               distribution=None):
    # Number of Machines in single line         =   M
    # Number of Kinds about Products            =   P
    # Max order quantity in a requirement       =   MaxQ
    # Min order quantity in a requirement       =   MinQ
    # Max due date about a requirement          =   MaxT
    # Set-up time(model change)                 =   setuptime
    # Max time of a single demand statements    =   maxalltime
    # A single machine capacity                 =   capacity

    if M is None:
        M = 17
    if P is None:
        P = 8
    if maxQ is None:
        maxQ = 8000
    if minQ is None:
        minQ = 100
    if maxT is None:
        maxT = 12
    if setuptime is None:
        setuptime = 1
    if maxalltime is None:
        maxalltime = 148
    if capacity is None:
        capacity = 1000
    if distribution is None:
        distribution = "Uniform"

    PreSet = PreSetInfo(M, P, maxQ, minQ, maxT, setuptime, maxalltime, capacity, distribution)
    return PreSet

def main():
    Initialize(16, 4)
    Params = Initialize()
    print(Params.MachinesNumber())
    print(Params.ProductKinds())
    print(Params.MaxOrderQ())
    print(Params.MinOrderQ())
    print(Params.MaxDueDate())
    print(Params.SetUpTime())
    print(Params.LimitationTime())
    print(Params.MachineCapa())
    print(Params.Distribution())

if __name__ == '__main__':
    main()