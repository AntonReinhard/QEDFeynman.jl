using QEDFeynman
using ComputableDAGs

function test_op_specific(estimator, dag::DAG, nr::NodeReduction)
    estimate = operation_effect(estimator, dag, nr)

    input_node = dag.nodes[first(nr.input)]
    t = task(input_node)

    data_reduce = sum(data.(task.(children(dag, input_node)))) * (length(nr.input) - 1)
    compute_effort_reduce = compute_effort(t) * (length(nr.input) - 1)

    @test isapprox(estimate.data, -data_reduce; atol = eps(Float64))
    @test isapprox(estimate.compute_effort, -compute_effort_reduce)
    @test isapprox(compute_intensity(estimate), compute_effort_reduce / data_reduce)

    return nothing
end

function test_op_specific(estimator, dag, ns::NodeSplit)
    estimate = operation_effect(estimator, dag, ns)

    input_node = dag.nodes[ns.input]

    copies = length(input_node.parents) - 1

    data_increase = zero(Float64)
    compute_effort_increase = compute_effort(input_node.task) * copies

    @test isapprox(estimate.data, data_increase; atol = eps(Float64))
    @test isapprox(estimate.compute_effort, compute_effort_increase)

    @test isnan(compute_intensity(estimate)) && isnan(compute_effort_increase / data_increase) || isapprox(compute_intensity(estimate), compute_effort_increase / data_increase)

    return nothing
end

function test_op(estimator, dag, op)
    estimate_before = graph_cost(estimator, dag)

    estimate = operation_effect(estimator, dag, op)

    keys_before = Set(keys(dag.nodes))
    push_operation!(dag, op)
    estimate_after_apply = graph_cost(estimator, dag)
    reset_graph!(dag)
    keys_after = Set(keys(dag.nodes))

    # symmetric set difference should be empty
    @test setdiff(keys_before, keys_after) == setdiff(keys_after, keys_before)

    @test isapprox((estimate_before + estimate).data, estimate_after_apply.data)
    @test isapprox(
        (estimate_before + estimate).compute_effort, estimate_after_apply.compute_effort
    )
    @test isapprox(
        compute_intensity(estimate_before + estimate),
        compute_intensity(estimate_after_apply),
    )

    test_op_specific(estimator, dag, op)
    return nothing
end

@testset "Global Metric Estimator" for (dag_string, exp_data, exp_compute_effort) in
    zip(["AB->AB", "AB->ABBB"], [784, 10656], [53, 1075])
    estimator = GlobalMetricEstimator()

    @test cost_type(estimator) == CDCost

    proc = parse_process(dag_string, ABCModel())
    dag = parse_dag(joinpath(@__DIR__, "..", "input", "$(dag_string).txt"), proc)

    @testset "DAG Cost" begin
        estimate = graph_cost(estimator, dag)

        @test estimate.data == exp_data
        @test estimate.compute_effort == exp_compute_effort
        @test isapprox(compute_intensity(estimate), exp_compute_effort / exp_data)
    end

    @testset "Operation Cost" begin
        ops = get_operations(dag)
        nrs = copy(ops.node_reductions)
        nss = copy(ops.node_splits)

        for nr in nrs
            test_op(estimator, dag, nr)
        end
        for ns in nss
            test_op(estimator, dag, ns)
        end
    end
end
