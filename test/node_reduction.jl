using QEDFeynman
using ComputableDAGs

dag = DAG()

d_exit = insert_node!(dag, DataTask(10))

s0 = insert_node!(dag, ComputeTaskABC_S2())

ED = insert_node!(dag, DataTask(3))
FD = insert_node!(dag, DataTask(3))

EC = insert_node!(dag, ComputeTaskABC_V())
FC = insert_node!(dag, ComputeTaskABC_V())

A1D = insert_node!(dag, DataTask(4))
B1D_1 = insert_node!(dag, DataTask(4))
B1D_2 = insert_node!(dag, DataTask(4))
C1D = insert_node!(dag, DataTask(4))

A1C = insert_node!(dag, ComputeTaskABC_U())
B1C_1 = insert_node!(dag, ComputeTaskABC_U())
B1C_2 = insert_node!(dag, ComputeTaskABC_U())
C1C = insert_node!(dag, ComputeTaskABC_U())

AD = insert_node!(dag, DataTask(5))
BD = insert_node!(dag, DataTask(5))
CD = insert_node!(dag, DataTask(5))

insert_edge!(dag, s0, d_exit)
insert_edge!(dag, ED, s0)
insert_edge!(dag, FD, s0)
insert_edge!(dag, EC, ED)
insert_edge!(dag, FC, FD)

insert_edge!(dag, A1D, EC)
insert_edge!(dag, B1D_1, EC)

insert_edge!(dag, B1D_2, FC)
insert_edge!(dag, C1D, FC)

insert_edge!(dag, A1C, A1D)
insert_edge!(dag, B1C_1, B1D_1)
insert_edge!(dag, B1C_2, B1D_2)
insert_edge!(dag, C1C, C1D)

insert_edge!(dag, AD, A1C)
insert_edge!(dag, BD, B1C_1)
insert_edge!(dag, BD, B1C_2)
insert_edge!(dag, CD, C1C)

@test is_valid(dag)

@test is_exit_node(d_exit)
@test is_entry_node(AD)
@test is_entry_node(BD)
@test is_entry_node(CD)

opt = get_operations(dag)

@test length(opt) == (node_reductions = 1, node_splits = 1)

nr = first(opt.node_reductions)
@test Set(nr.input) == Set([B1C_1.id, B1C_2.id])
push_operation!(dag, nr)
opt = get_operations(dag)

@test length(opt) == (node_reductions = 1, node_splits = 1)

nr = first(opt.node_reductions)
@test Set(nr.input) == Set([B1D_1.id, B1D_2.id])
push_operation!(dag, nr)
opt = get_operations(dag)

@test is_valid(dag)

@test length(opt) == (node_reductions = 0, node_splits = 1)

pop_operation!(dag)

opt = get_operations(dag)
@test length(opt) == (node_reductions = 1, node_splits = 1)

reset_graph!(dag)

opt = get_operations(dag)
@test length(opt) == (node_reductions = 1, node_splits = 1)

@test is_valid(dag)
