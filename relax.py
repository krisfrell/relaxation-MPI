import time

import numpy as np
from mpi4py import MPI


def condition(next, prev, n):
    sub = np.subtract(next, prev)
    max = np.abs(sub[0])
    for i in range (1, n):
        if np.abs(sub[i]) > max:
            max = np.abs(sub[i])

    return max


def relax(A, b, c, comm, eps, omega, n):
    start = time.time_ns()
    counter = 0
    k = (n + 1) // comm.size
    i1 = k * comm.rank
    i2 = k * (comm.rank + 1)
    if comm.rank == comm.size - 1:
        i2 = n + 1
    next = np.copy(c)
    while True:
        prev = np.copy(next)
        for i in range(n):
            proc_sum = np.zeros(1)
            sum = np.zeros(1)
            np.put(next, i, (1 - omega) * prev[i])
            for j in range(0, i):
                if (i1 <= j) and (j < i2):
                    proc_sum += (A[i, j] / A[i, i]) * next[j]
            comm.Barrier()
            comm.Reduce([proc_sum, MPI.DOUBLE], [sum, MPI.DOUBLE], op=MPI.SUM, root=0)
            next[i] -= omega * sum
            proc_sum = np.zeros(1)
            sum = np.zeros(1)
            for j in range(i + 1, n):
                if (i1 <= j) and (j < i2):
                    proc_sum[0] += (A[i, j] / A[i, i]) * prev[j]
            comm.Barrier()
            comm.Reduce([proc_sum, MPI.DOUBLE], [sum, MPI.DOUBLE], op=MPI.SUM, root=0)
            if comm.rank == 0:
                next[i] -= omega * sum
                next[i] += omega * (b[i] / A[i, i])

        if comm.rank == 0:
            for proc in range(1, comm.size):
                comm.send(next, dest=proc, tag=3)
                comm.send(prev, dest=proc, tag=4)
        else:
            next = comm.recv(source=0, tag=3)
            prev = comm.recv(source=0, tag=4)

        counter += 1
        if not condition(next, prev, n) > omega * eps:
            end = time.time_ns() - start
            return next, counter, end

