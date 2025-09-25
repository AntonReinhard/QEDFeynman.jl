using QEDFeynman
using ComputableDAGs
using Random

RNG = Random.MersenneTwister(0)

proc = parse_process("AB->ABBB", ABCModel())
graph = parse_dag(joinpath(@__DIR__, "..", "input", "AB->ABBB.txt"), proc)

# create the optimizers
FIXPOINT_OPTIMIZERS = [
    GreedyOptimizer(GlobalMetricEstimator()), ReductionOptimizer(), SplitOptimizer(),
]
NO_FIXPOINT_OPTIMIZERS = [RandomWalkOptimizer(RNG)]

@testset "Optimizer $optimizer" for optimizer in
    vcat(NO_FIXPOINT_OPTIMIZERS, FIXPOINT_OPTIMIZERS)
    @test operation_stack_length(graph) == 0
    @test optimize_step!(optimizer, graph)

    @test !fixpoint_reached(optimizer, graph)
    @test operation_stack_length(graph) == 1

    @test optimize!(optimizer, graph, 2)

    @test !fixpoint_reached(optimizer, graph)

    reset_graph!(graph)
end

@testset "Fixpoint optimizer $optimizer" for optimizer in FIXPOINT_OPTIMIZERS
    @test operation_stack_length(graph) == 0

    optimize_to_fixpoint!(optimizer, graph)

    @test fixpoint_reached(optimizer, graph)
    @test !optimize_step!(optimizer, graph)
    @test !optimize!(optimizer, graph, 10)

    reset_graph!(graph)
end

@testset "No fixpoint optimizer $optimizer" for optimizer in NO_FIXPOINT_OPTIMIZERS
    @test_throws MethodError optimize_to_fixpoint!(optimizer, graph)
end
