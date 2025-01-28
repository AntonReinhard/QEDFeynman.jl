using QEDFeynman
using ComputableDAGs

nC1 = make_node(QEDFeynman.ComputeTaskABC_U())
nC2 = make_node(QEDFeynman.ComputeTaskABC_V())
nC3 = make_node(QEDFeynman.ComputeTaskABC_S1())
nC4 = make_node(QEDFeynman.ComputeTaskABC_Sum())

nD1 = make_node(DataTask(10))
nD2 = make_node(DataTask(20))

@test_throws ErrorException make_edge(nC1, nC2)
@test_throws ErrorException make_edge(nC1, nC1)
@test_throws ErrorException make_edge(nC3, nC4)
@test_throws ErrorException make_edge(nD1, nD2)
@test_throws ErrorException make_edge(nD1, nD1)

ed1 = make_edge(nC1, nD1)
ed2 = make_edge(nD1, nC2)
ed3 = make_edge(nC2, nD2)
ed4 = make_edge(nD2, nC3)

@test nC1 != nC2
@test nD1 != nD2
@test nC1 != nD1
@test nC3 != nC4

nC1_2 = copy(nC1)
@test nC1_2 != nC1

nD1_2 = copy(nD1)
@test nD1_2 != nD1

nD1_c = make_node(DataTask(10))
@test nD1_c != nD1
