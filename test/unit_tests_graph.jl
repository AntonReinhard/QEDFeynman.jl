using QEDFeynman
using ComputableDAGs

graph = DAG()

@test length(graph.nodes) == 0
@test length(graph.applied_operations) == 0
@test length(graph.operations_to_apply) == 0
@test length(graph.dirty_nodes) == 0
@test length(graph.diff) == (addedNodes=0, removedNodes=0, addedEdges=0, removedEdges=0)
@test length(get_operations(graph)) == (node_reductions=0, node_splits=0)

# s to output (exit node)
d_exit = insert_node!(graph, DataTask(10))

@test length(graph.nodes) == 1
@test length(graph.dirty_nodes) == 0

# final s compute
s0 = insert_node!(graph, ComputeTaskABC_S2())

@test length(graph.nodes) == 2
@test length(graph.dirty_nodes) == 0

# data from v0 and v1 to s0
d_v0_s0 = insert_node!(graph, DataTask(5))
d_v1_s0 = insert_node!(graph, DataTask(5))

# v0 and v1 compute
v0 = insert_node!(graph, ComputeTaskABC_V())
v1 = insert_node!(graph, ComputeTaskABC_V())

# data from uB, uA, uBp and uAp to v0 and v1
d_uB_v0 = insert_node!(graph, DataTask(3))
d_uA_v0 = insert_node!(graph, DataTask(3))
d_uBp_v1 = insert_node!(graph, DataTask(3))
d_uAp_v1 = insert_node!(graph, DataTask(3))

# uB, uA, uBp and uAp computes
uB = insert_node!(graph, ComputeTaskABC_U())
uA = insert_node!(graph, ComputeTaskABC_U())
uBp = insert_node!(graph, ComputeTaskABC_U())
uAp = insert_node!(graph, ComputeTaskABC_U())

# data from PB, PA, PBp and PAp to uB, uA, uBp and uAp
d_PB_uB = insert_node!(graph, DataTask(6))
d_PA_uA = insert_node!(graph, DataTask(6))
d_PBp_uBp = insert_node!(graph, DataTask(6))
d_PAp_uAp = insert_node!(graph, DataTask(6))

# P computes PB, PA, PBp and PAp
PB = insert_node!(graph, ComputeTaskABC_P())
PA = insert_node!(graph, ComputeTaskABC_P())
PBp = insert_node!(graph, ComputeTaskABC_P())
PAp = insert_node!(graph, ComputeTaskABC_P())

# entry nodes getting data for P computes
d_PB = insert_node!(graph, DataTask(4))
d_PA = insert_node!(graph, DataTask(4))
d_PBp = insert_node!(graph, DataTask(4))
d_PAp = insert_node!(graph, DataTask(4))

@test length(graph.nodes) == 26
@test length(graph.dirty_nodes) == 0

# now for all the edges
insert_edge!(graph, d_PB, PB)
insert_edge!(graph, d_PA, PA)
insert_edge!(graph, d_PBp, PBp)
insert_edge!(graph, d_PAp, PAp)

insert_edge!(graph, PB, d_PB_uB)
insert_edge!(graph, PA, d_PA_uA)
insert_edge!(graph, PBp, d_PBp_uBp)
insert_edge!(graph, PAp, d_PAp_uAp)

insert_edge!(graph, d_PB_uB, uB)
insert_edge!(graph, d_PA_uA, uA)
insert_edge!(graph, d_PBp_uBp, uBp)
insert_edge!(graph, d_PAp_uAp, uAp)

insert_edge!(graph, uB, d_uB_v0)
insert_edge!(graph, uA, d_uA_v0)
insert_edge!(graph, uBp, d_uBp_v1)
insert_edge!(graph, uAp, d_uAp_v1)

insert_edge!(graph, d_uB_v0, v0)
insert_edge!(graph, d_uA_v0, v0)
insert_edge!(graph, d_uBp_v1, v1)
insert_edge!(graph, d_uAp_v1, v1)

insert_edge!(graph, v0, d_v0_s0)
insert_edge!(graph, v1, d_v1_s0)

insert_edge!(graph, d_v0_s0, s0)
insert_edge!(graph, d_v1_s0, s0)

insert_edge!(graph, s0, d_exit)

@test length(graph.nodes) == 26
@test length(graph.applied_operations) == 0
@test length(graph.operations_to_apply) == 0
@test length(graph.dirty_nodes) == 0
@test length(graph.diff) == (addedNodes=0, removedNodes=0, addedEdges=0, removedEdges=0)

@test is_valid(graph)

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

@test get_exit_node(graph) == d_exit

@test length(partners(s0)) == 1
@test length(siblings(s0)) == 1

operations = get_operations(graph)
@test length(operations) == (node_reductions=0, node_splits=0)
@test length(graph.dirty_nodes) == 0

@test operations == get_operations(graph)

properties = get_properties(graph)
@test properties.compute_effort == 28
@test properties.data == 62
@test properties.compute_intensity ≈ 28 / 62
@test properties.number_of_nodes == 26
@test properties.number_of_edges == 25

operations = get_operations(graph)
@test length(graph.dirty_nodes) == 0

@test length(operations) == (node_reductions=0, node_splits=0)
@test isempty(operations)
@test length(graph.dirty_nodes) == 0
@test length(graph.nodes) == 26
@test length(graph.applied_operations) == 0
@test length(graph.operations_to_apply) == 0

reset_graph!(graph)

@test length(graph.dirty_nodes) == 0
@test length(graph.nodes) == 26
@test length(graph.applied_operations) == 0
@test length(graph.operations_to_apply) == 0

properties = get_properties(graph)
@test properties.number_of_nodes == 26
@test properties.number_of_edges == 25
@test properties.compute_effort == 28
@test properties.data == 62
@test properties.compute_intensity ≈ 28 / 62

operations = get_operations(graph)
@test length(operations) == (node_reductions=0, node_splits=0)

@test is_valid(graph)
