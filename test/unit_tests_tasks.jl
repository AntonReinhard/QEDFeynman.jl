using QEDFeynman
using ComputableDAGs

S1 = QEDFeynman.ComputeTaskABC_S1()
S2 = QEDFeynman.ComputeTaskABC_S2()
U = QEDFeynman.ComputeTaskABC_U()
V = QEDFeynman.ComputeTaskABC_V()
Sum = QEDFeynman.ComputeTaskABC_Sum()

Data10 = DataTask(10)
Data20 = DataTask(20)

@test compute_effort(S1) == 11
@test compute_effort(S2) == 12
@test compute_effort(U) == 1
@test compute_effort(V) == 6
@test compute_effort(Sum) == 1
@test compute_effort(Data10) == 0
@test compute_effort(Data20) == 0

@test data(S1) == 0
@test data(S2) == 0
@test data(U) == 0
@test data(V) == 0
@test data(Sum) == 0
@test data(Data10) == 10
@test data(Data20) == 20

@test S1 != S2
@test Data10 != Data20

Data10_2 = DataTask(10)

# two data tasks with same data are identical, their nodes need not be
@test Data10_2 == Data10

@test Data10 == Data10
@test S1 == S1

Data10_3 = copy(Data10)

@test Data10_3 == Data10

S1_2 = copy(S1)

@test S1_2 == S1
@test S1 == QEDFeynman.ComputeTaskABC_S1()
