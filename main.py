from relax import *

n = 5000
eps = 0.000001
omega = 1.1
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


def createA(n):
    A = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                A[i, j] = A[j, i] = np.random.random()
            else:
                A[i, j] = np.random.uniform(n, 2 * n)
    return A


A = createA(n) if rank == 0 else None
b = np.random.rand(n) if rank == 0 else None
diag = np.ravel(np.diagonal(A)[None, :]) if rank == 0 else None

if rank == 0:
    for proc in range(1, comm.size):
        comm.send(A, dest=proc, tag=1)
        comm.send(b, dest=proc, tag=2)
else:
    A = comm.recv(source=0, tag=1)
    b = comm.recv(source=0, tag=2)

c = np.zeros(n, dtype=float)
local_n = np.array([0])

if rank == 0:
    if (n % size != 0):
        print("the number of processors must evenly divide n.")
        comm.Abort()

local_n = np.array([n // size])


# начальное приближение
comm.Bcast(local_n, root=0)
local_diag = np.zeros(local_n)
local_b = np.zeros(local_n)
comm.Scatterv(diag, local_diag, root=0)
comm.Scatterv(b, local_b, root=0)
local_c = np.divide(local_b, local_diag)
comm.Barrier()

comm.Allgatherv(local_c, c)

res, count, time = relax(A, b, c, comm, eps, omega, n)

if rank == 0:
    print("time in ns: ")
    print(time)
