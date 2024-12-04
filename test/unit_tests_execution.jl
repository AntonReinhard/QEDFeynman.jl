using QEDFeynman
using ComputableDAGs
using QEDcore
using QEDprocesses
using AccurateArithmetic
using Random
using StaticArrays

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

import QEDFeynman.ABCParticle
import QEDFeynman.interaction_result

const RTOL = sqrt(eps(Float64))
RNG = Random.MersenneTwister(0)

function check_particle_reverse_moment(p1::SFourMomentum, p2::SFourMomentum)
    @test isapprox(abs(p1.E), abs(p2.E))
    @test isapprox(p1.px, -p2.px)
    @test isapprox(p1.py, -p2.py)
    @test isapprox(p1.pz, -p2.pz)
    return nothing
end

function ground_truth_graph_result(input::PhaseSpacePoint)
    # formula for one diagram:
    # u_Bp * iλ * u_Ap * S_C * u_B * iλ * u_A
    # for the second diagram:
    # u_B * iλ * u_Ap * S_C * u_Bp * iλ * u_Ap
    # the "u"s are all 1, we ignore the i, λ is 1/137.

    constant = (1 / 137.0)^2

    # calculate particle C in diagram 1
    diagram1_C = ParticleStateful(
        Outgoing(),
        ParticleC(),
        momentum(input, Incoming(), ParticleA()) + momentum(input, Incoming(), ParticleB()),
    )
    diagram2_C = ParticleStateful(
        Outgoing(),
        ParticleC(),
        momentum(input, Incoming(), ParticleA()) - momentum(input, Outgoing(), ParticleB()),
    )

    diagram1_Cp = ParticleStateful(
        Incoming(),
        ParticleC(),
        -momentum(input, Outgoing(), ParticleA()) -
        momentum(input, Outgoing(), ParticleB()),
    )
    diagram2_Cp = ParticleStateful(
        Incoming(),
        ParticleC(),
        -momentum(input, Outgoing(), ParticleA()) +
        momentum(input, Incoming(), ParticleB()),
    )

    check_particle_reverse_moment(momentum(diagram1_Cp), momentum(diagram1_C))
    check_particle_reverse_moment(momentum(diagram2_Cp), momentum(diagram2_C))
    @test isapprox(getMass2(momentum(diagram1_C)), getMass2(momentum(diagram1_Cp)))
    @test isapprox(getMass2(momentum(diagram2_C)), getMass2(momentum(diagram2_Cp)))

    inner1 = QEDFeynman.ABC_inner_edge(diagram1_C)
    inner2 = QEDFeynman.ABC_inner_edge(diagram2_C)

    diagram1_result = inner1 * constant
    diagram2_result = inner2 * constant

    return sum_kbn([diagram1_result, diagram2_result])
end

machine = cpu_st()

process_2_2 = GenericABCProcess((ParticleA(), ParticleB()), (ParticleA(), ParticleB()))

particles_2_2 = PhaseSpacePoint(
    process_2_2,
    PerturbativeABC(),
    PhasespaceDefinition(SphericalCoordinateSystem(), ElectronRestFrame()),
    (
        SFourMomentum(0.823648, 0.0, 0.0, 0.823648),
        SFourMomentum(0.823648, 0.0, 0.0, -0.823648),
    ),
    (
        SFourMomentum(0.823648, -0.835061, -0.474802, 0.277915),
        SFourMomentum(0.823648, 0.835061, 0.474802, -0.277915),
    ),
)

expected_result = ground_truth_graph_result(particles_2_2)

@testset "AB->AB no optimization" begin
    for _ in 1:10   # test in a loop because graph layout should not change the result
        graph = parse_dag(joinpath(@__DIR__, "..", "input", "AB->AB.txt"), process_2_2)

        func = get_compute_function(graph, process_2_2, machine, @__MODULE__)
        @test isapprox(func(particles_2_2), expected_result; rtol=RTOL)
    end
end

@testset "AB->AB after random walk" begin
    for i in 1:200
        graph = parse_dag(joinpath(@__DIR__, "..", "input", "AB->AB.txt"), process_2_2)
        optimize!(RandomWalkOptimizer(RNG), graph, 50)

        @test is_valid(graph)

        func = get_compute_function(graph, process_2_2, machine, @__MODULE__)
        @test isapprox(func(particles_2_2), expected_result; rtol=RTOL)

        # graph should be fully scheduled after being executed
        @test is_scheduled(graph)
    end
end

process_2_4 = GenericABCProcess(
    (ParticleA(), ParticleB()), (ParticleA(), ParticleB(), ParticleB(), ParticleB())
)

particles_2_4 = gen_process_input(process_2_4)
graph = parse_dag(joinpath(@__DIR__, "..", "input", "AB->ABBB.txt"), process_2_4)
groundtruth_func = get_compute_function(graph, process_2_4, machine, @__MODULE__)
expected_result = groundtruth_func(particles_2_4)

@testset "AB->ABBB no optimization" begin
    for _ in 1:5   # test in a loop because graph layout should not change the result
        graph = parse_dag(joinpath(@__DIR__, "..", "input", "AB->ABBB.txt"), process_2_4)

        func = get_compute_function(graph, process_2_4, machine, @__MODULE__)
        @test isapprox(func(particles_2_4), expected_result; rtol=RTOL)
    end
end

#=
TODO: fix precision(?) issues
@testset "AB->ABBB after random walk" begin
    for i in 1:50
        graph = parse_dag(joinpath(@__DIR__, "..", "input", "AB->ABBB.txt"), process_2_4)
        optimize!(RandomWalkOptimizer(RNG), graph, 100)
        @test is_valid(graph)

        func = get_compute_function(graph, process_2_4, machine, @__MODULE__)
        @test isapprox(func(particles_2_4), expected_result; rtol=RTOL)
    end
end
=#

@testset "$(process) after random walk" for process in ["ke->ke", "ke->kke", "ke->kkke"]
    process = parse_process("ke->kkke", QEDModel())
    inputs = [gen_process_input(process) for _ in 1:100]
    graph = graph(process)

    f_gt = get_compute_function(graph, process, machine, @__MODULE__)
    gt = f_gt.(inputs)
    for i in 1:50
        graph = graph(process)

        optimize!(RandomWalkOptimizer(RNG), graph, 100)
        @test is_valid(graph)

        func = get_compute_function(graph, process, machine, @__MODULE__)
        @test isapprox(func.(inputs), gt; rtol=RTOL)
    end
end
