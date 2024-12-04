using QEDFeynman
using ComputableDAGs

graph = DAG()

d_exit = insert_node!(graph, DataTask(10))

s0 = insert_node!(graph, ComputeTaskABC_S2())

ED = insert_node!(graph, DataTask(3))
FD = insert_node!(graph, DataTask(3))

EC = insert_node!(graph, ComputeTaskABC_V())
FC = insert_node!(graph, ComputeTaskABC_V())

A1D = insert_node!(graph, DataTask(4))
B1D_1 = insert_node!(graph, DataTask(4))
B1D_2 = insert_node!(graph, DataTask(4))
C1D = insert_node!(graph, DataTask(4))

A1C = insert_node!(graph, ComputeTaskABC_U())
B1C_1 = insert_node!(graph, ComputeTaskABC_U())
B1C_2 = insert_node!(graph, ComputeTaskABC_U())
C1C = insert_node!(graph, ComputeTaskABC_U())

AD = insert_node!(graph, DataTask(5))
BD = insert_node!(graph, DataTask(5))
CD = insert_node!(graph, DataTask(5))

insert_edge!(graph, s0, d_exit)
insert_edge!(graph, ED, s0)
insert_edge!(graph, FD, s0)
insert_edge!(graph, EC, ED)
insert_edge!(graph, FC, FD)

insert_edge!(graph, A1D, EC)
insert_edge!(graph, B1D_1, EC)

insert_edge!(graph, B1D_2, FC)
insert_edge!(graph, C1D, FC)

insert_edge!(graph, A1C, A1D)
insert_edge!(graph, B1C_1, B1D_1)
insert_edge!(graph, B1C_2, B1D_2)
insert_edge!(graph, C1C, C1D)

insert_edge!(graph, AD, A1C)
insert_edge!(graph, BD, B1C_1)
insert_edge!(graph, BD, B1C_2)
insert_edge!(graph, CD, C1C)

@test is_valid(graph)

@test is_exit_node(d_exit)
@test is_entry_node(AD)
@test is_entry_node(BD)
@test is_entry_node(CD)

opt = get_operations(graph)

@test length(opt) == (node_reductions=1, node_splits=1)

nr = first(opt.node_reductions)
@test Set(nr.input) == Set([B1C_1, B1C_2])
push_operation!(graph, nr)
opt = get_operations(graph)

@test length(opt) == (node_reductions=1, node_splits=1)

nr = first(opt.node_reductions)
@test Set(nr.input) == Set([B1D_1, B1D_2])
push_operation!(graph, nr)
opt = get_operations(graph)

@test is_valid(graph)

@test length(opt) == (node_reductions=0, node_splits=1)

pop_operation!(graph)

opt = get_operations(graph)
@test length(opt) == (node_reductions=1, node_splits=1)

reset_graph!(graph)

opt = get_operations(graph)
@test length(opt) == (node_reductions=1, node_splits=1)

@test is_valid(graph)
