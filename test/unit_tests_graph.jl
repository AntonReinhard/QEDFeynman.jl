using QEDFeynman
using ComputableDAGs

dag = DAG()

@test length(dag.nodes) == 0
@test length(dag.applied_operations) == 0
@test length(dag.operations_to_apply) == 0
@test length(dag.dirty_nodes) == 0
@test length(dag.diff) == (added_nodes = 0, removed_nodes = 0, added_edges = 0, removed_edges = 0)
@test length(get_operations(dag)) == (node_reductions = 0, node_splits = 0)

# s to output (exit node)
d_exit = insert_node!(dag, DataTask(10))

@test length(dag.nodes) == 1
@test length(dag.dirty_nodes) == 0

# final s compute
s0 = insert_node!(dag, ComputeTaskABC_S2())

@test length(dag.nodes) == 2
@test length(dag.dirty_nodes) == 0

# data from v0 and v1 to s0
d_v0_s0 = insert_node!(dag, DataTask(5))
d_v1_s0 = insert_node!(dag, DataTask(5))

# v0 and v1 compute
v0 = insert_node!(dag, ComputeTaskABC_V())
v1 = insert_node!(dag, ComputeTaskABC_V())

# data from uB, uA, uBp and uAp to v0 and v1
d_uB_v0 = insert_node!(dag, DataTask(3))
d_uA_v0 = insert_node!(dag, DataTask(3))
d_uBp_v1 = insert_node!(dag, DataTask(3))
d_uAp_v1 = insert_node!(dag, DataTask(3))

# uB, uA, uBp and uAp computes
uB = insert_node!(dag, ComputeTaskABC_U())
uA = insert_node!(dag, ComputeTaskABC_U())
uBp = insert_node!(dag, ComputeTaskABC_U())
uAp = insert_node!(dag, ComputeTaskABC_U())

# entry nodes getting data for U computes
d_uB = insert_node!(dag, DataTask(6))
d_uA = insert_node!(dag, DataTask(6))
d_uBp = insert_node!(dag, DataTask(6))
d_uAp = insert_node!(dag, DataTask(6))

@test length(dag.nodes) == 18
@test length(dag.dirty_nodes) == 0

# now for all the edges
insert_edge!(dag, d_uB, uB)
insert_edge!(dag, d_uA, uA)
insert_edge!(dag, d_uBp, uBp)
insert_edge!(dag, d_uAp, uAp)

insert_edge!(dag, uB, d_uB_v0)
insert_edge!(dag, uA, d_uA_v0)
insert_edge!(dag, uBp, d_uBp_v1)
insert_edge!(dag, uAp, d_uAp_v1)

insert_edge!(dag, d_uB_v0, v0)
insert_edge!(dag, d_uA_v0, v0)
insert_edge!(dag, d_uBp_v1, v1)
insert_edge!(dag, d_uAp_v1, v1)

insert_edge!(dag, v0, d_v0_s0)
insert_edge!(dag, v1, d_v1_s0)

insert_edge!(dag, d_v0_s0, s0)
insert_edge!(dag, d_v1_s0, s0)

insert_edge!(dag, s0, d_exit)

@test length(dag.nodes) == 18
@test length(dag.applied_operations) == 0
@test length(dag.operations_to_apply) == 0
@test length(dag.dirty_nodes) == 0
@test length(dag.diff) == (added_nodes = 0, removed_nodes = 0, added_edges = 0, removed_edges = 0)

@test is_valid(dag)

@test is_entry_node(d_PB)
@test is_entry_node(d_PA)
@test is_entry_node(d_PBp)
@test is_entry_node(d_PBp)
@test !is_entry_node(PB)
@test !is_entry_node(v0)
@test !is_entry_node(d_exit)

@test is_exit_node(d_exit)
@test !is_exit_node(d_uB_v0)
@test !is_exit_node(v0)

@test length(children(v0)) == 2
@test length(children(v1)) == 2
@test length(parents(v0)) == 1
@test length(parents(v1)) == 1

@test get_exit_node(dag) == d_exit

@test length(partners(s0)) == 1
@test length(siblings(s0)) == 1

operations = get_operations(dag)
@test length(operations) == (node_reductions = 0, node_splits = 0)
@test length(dag.dirty_nodes) == 0

@test operations == get_operations(dag)

properties = get_properties(dag)
@test properties.compute_effort == 28
@test properties.data == 62
@test properties.compute_intensity ≈ 28 / 62
@test properties.number_of_nodes == 26
@test properties.number_of_edges == 25

operations = get_operations(dag)
@test length(dag.dirty_nodes) == 0

@test length(operations) == (node_reductions = 0, node_splits = 0)
@test isempty(operations)
@test length(dag.dirty_nodes) == 0
@test length(dag.nodes) == 26
@test length(dag.applied_operations) == 0
@test length(dag.operations_to_apply) == 0

reset_graph!(dag)

@test length(dag.dirty_nodes) == 0
@test length(dag.nodes) == 26
@test length(dag.applied_operations) == 0
@test length(dag.operations_to_apply) == 0

properties = get_properties(dag)
@test properties.number_of_nodes == 26
@test properties.number_of_edges == 25
@test properties.compute_effort == 28
@test properties.data == 62
@test properties.compute_intensity ≈ 28 / 62

operations = get_operations(dag)
@test length(operations) == (node_reductions = 0, node_splits = 0)

@test is_valid(dag)
