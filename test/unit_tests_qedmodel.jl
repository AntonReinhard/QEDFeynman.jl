using QEDFeynman
using ComputableDAGs
using QEDbase
using QEDcore
using QEDprocesses
using Random

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

import QEDFeynman.caninteract
import QEDFeynman.issame
import QEDFeynman.interaction_result
import QEDFeynman.QED_vertex

def_momentum = SFourMomentum(1.0, 0.0, 0.0, 0.0)

RNG = Random.MersenneTwister(0)

testparticleTypes = [
    ParticleStateful{Incoming,Photon,SFourMomentum},
    ParticleStateful{Outgoing,Photon,SFourMomentum},
    ParticleStateful{Incoming,Electron,SFourMomentum},
    ParticleStateful{Outgoing,Electron,SFourMomentum},
    ParticleStateful{Incoming,Positron,SFourMomentum},
    ParticleStateful{Outgoing,Positron,SFourMomentum},
]

testparticleTypesPropagated = [
    ParticleStateful{Outgoing,Photon,SFourMomentum},
    ParticleStateful{Incoming,Photon,SFourMomentum},
    ParticleStateful{Outgoing,Electron,SFourMomentum},
    ParticleStateful{Incoming,Electron,SFourMomentum},
    ParticleStateful{Outgoing,Positron,SFourMomentum},
    ParticleStateful{Incoming,Positron,SFourMomentum},
]

function compton_groundtruth(input::PhaseSpacePoint)
    # p1k1 -> p2k2
    # formula: −(ie)^2 (u(p2) slashed(ε1) S(p2 − k1) slashed(ε2) u(p1) + u(p2) slashed(ε2) S(p1 + k1) slashed(ε1) u(p1))

    p1 = momentum(psp, Incoming(), 2)
    p2 = momentum(psp, Outgoing(), 2)

    k1 = momentum(psp, Incoming(), 1)
    k2 = momentum(psp, Outgoing(), 1)

    u_p1 = base_state(Electron(), Incoming(), p1.momentum, spin_or_pol(p1))
    u_p2 = base_state(Electron(), Outgoing(), p2.momentum, spin_or_pol(p2))

    eps_1 = base_state(Photon(), Incoming(), k1.momentum, spin_or_pol(k1))
    eps_2 = base_state(Photon(), Outgoing(), k2.momentum, spin_or_pol(k2))

    virt1_mom = p2.momentum - k1.momentum
    @test isapprox(p1.momentum - k2.momentum, virt1_mom)

    virt2_mom = p1.momentum + k1.momentum
    @test isapprox(p2.momentum + k2.momentum, virt2_mom)

    s_p2_k1 = QEDbase.propagator(Electron(), virt1_mom)
    s_p1_k1 = QEDbase.propagator(Electron(), virt2_mom)

    diagram1 = u_p2 * (eps_1 * QED_vertex()) * s_p2_k1 * (eps_2 * QED_vertex()) * u_p1
    diagram2 = u_p2 * (eps_2 * QED_vertex()) * s_p1_k1 * (eps_1 * QED_vertex()) * u_p1

    return diagram1 + diagram2
end

#=
TODO: Rewrite for particle -> type version of interaction_result
@testset "Interaction Result" begin
    import QEDFeynman.QED_conserve_momentum

    for p1 in testparticleTypes, p2 in testparticleTypes
        if !caninteract(p1, p2)
            @test_throws AssertionError interaction_result(p1, p2)
            continue
        end

        @test interaction_result(p1, p2) in setdiff(testparticleTypes, [p1, p2])
        @test issame(interaction_result(p1, p2), interaction_result(p2, p1))

        testParticle1 = p1(rand(RNG, SFourMomentum))
        testParticle2 = p2(rand(RNG, SFourMomentum))
        p3 = interaction_result(p1, p2)

        resultParticle = QED_conserve_momentum(testParticle1, testParticle2)

        @test issame(typeof(resultParticle), interaction_result(p1, p2))

        totalMom = zero(SFourMomentum)
        for (p, mom) in [(p1, momentum(testParticle1)), (p2, momentum(testParticle2)), (p3, momentum(resultParticle))]
            if is_incoming(p)
                totalMom += mom
            else
                totalMom -= mom
            end
        end

        @test isapprox(totalMom, zero(SFourMomentum); atol = sqrt(eps()))
    end
end
=#

#= TODO REWRITE for propagated_particle
@testset "Propagation Result" begin
    for (p, propResult) in zip(testparticleTypes, testparticleTypesPropagated)
        @test issame(typeof(propagated_particle(p)), propResult)
        @test particle_direction(propagated_particle(p)) != particle_direction(p(def_momentum))
    end
end
=#

@testset "Parse Process" begin
    @testset "Known processes" begin
        proc = parse_process("ke->ke", QEDModel())
        @test incoming_particles(proc) == (Photon(), Electron())
        @test outgoing_particles(proc) == (Photon(), Electron())

        proc = parse_process("kp->kp", QEDModel())
        @test incoming_particles(proc) == (Photon(), Positron())
        @test outgoing_particles(proc) == (Photon(), Positron())

        proc = parse_process("ke->eep", QEDModel())
        @test incoming_particles(proc) == (Photon(), Electron())
        @test outgoing_particles(proc) == (Electron(), Electron(), Positron())

        proc = parse_process("kk->pe", QEDModel())
        @test incoming_particles(proc) == (Photon(), Photon())
        @test outgoing_particles(proc) == (Positron(), Electron())

        proc = parse_process("pe->kk", QEDModel())
        @test incoming_particles(proc) == (Positron(), Electron())
        @test outgoing_particles(proc) == (Photon(), Photon())
    end
end

@testset "Generate Process Inputs" begin
    @testset "Process $proc_str" for proc_str in ["ke->ke", "kp->kp", "kk->ep", "ep->kk"]
        # currently can only generate for 2->2 processes
        process = parse_process(proc_str, QEDModel())

        for i in 1:100
            input = gen_process_input(process)
            @test isapprox(
                sum(momenta(input, Incoming())),
                sum(momenta(input, Outgoing()));
                atol=sqrt(eps()),
            )
        end
    end
end

#=
@testset "Compton" begin
    model = QEDModel()
    process = parse_process("ke->ke", model)
    machine = cpu_st()

    graph = DAG()

    # manually build a graph for compton
    graph = DAG()

    # s to output (exit node)
    d_exit = insert_node!(graph, DataTask(16))

    sum_node = insert_node!(graph, ComputeTaskQED_Sum(2))

    d_s0_sum = insert_node!(graph, DataTask(16))
    d_s1_sum = insert_node!(graph, DataTask(16))

    # final s compute
    s0 = insert_node!(graph, ComputeTaskQED_S2())
    s1 = insert_node!(graph, ComputeTaskQED_S2())

    # data from v0 and v1 to s0
    d_v0_s0 = insert_node!(graph, DataTask(96))
    d_v1_s0 = insert_node!(graph, DataTask(96))
    d_v2_s1 = insert_node!(graph, DataTask(96))
    d_v3_s1 = insert_node!(graph, DataTask(96))

    # v0 and v1 compute
    v0 = insert_node!(graph, ComputeTaskQED_V())
    v1 = insert_node!(graph, ComputeTaskQED_V())
    v2 = insert_node!(graph, ComputeTaskQED_V())
    v3 = insert_node!(graph, ComputeTaskQED_V())

    # data from uPhIn, uPhOut, uElIn, uElOut to v0 and v1
    d_uPhIn_v0 = insert_node!(graph, DataTask(96))
    d_uElIn_v0 = insert_node!(graph, DataTask(96))
    d_uPhOut_v1 = insert_node!(graph, DataTask(96))
    d_uElOut_v1 = insert_node!(graph, DataTask(96))

    # data from uPhIn, uPhOut, uElIn, uElOut to v2 and v3
    d_uPhOut_v2 = insert_node!(graph, DataTask(96))
    d_uElIn_v2 = insert_node!(graph, DataTask(96))
    d_uPhIn_v3 = insert_node!(graph, DataTask(96))
    d_uElOut_v3 = insert_node!(graph, DataTask(96))

    # uPhIn, uPhOut, uElIn and uElOut computes
    uPhIn = insert_node!(graph, ComputeTaskQED_U())
    uPhOut = insert_node!(graph, ComputeTaskQED_U())
    uElIn = insert_node!(graph, ComputeTaskQED_U())
    uElOut = insert_node!(graph, ComputeTaskQED_U())

    # data into U
    d_uPhIn = insert_node!(graph, DataTask(16), "ki1")
    d_uPhOut = insert_node!(graph, DataTask(16), "ko1")
    d_uElIn = insert_node!(graph, DataTask(16), "ei1")
    d_uElOut = insert_node!(graph, DataTask(16), "eo1")

    # now for all the edges
    insert_edge!(graph, d_uPhIn, uPhIn)
    insert_edge!(graph, d_uPhOut, uPhOut)
    insert_edge!(graph, d_uElIn, uElIn)
    insert_edge!(graph, d_uElOut, uElOut)

    insert_edge!(graph, uPhIn, d_uPhIn_v0)
    insert_edge!(graph, uPhOut, d_uPhOut_v1)
    insert_edge!(graph, uElIn, d_uElIn_v0)
    insert_edge!(graph, uElOut, d_uElOut_v1)

    insert_edge!(graph, uPhIn, d_uPhIn_v3)
    insert_edge!(graph, uPhOut, d_uPhOut_v2)
    insert_edge!(graph, uElIn, d_uElIn_v2)
    insert_edge!(graph, uElOut, d_uElOut_v3)

    insert_edge!(graph, d_uPhIn_v0, v0)
    insert_edge!(graph, d_uPhOut_v1, v1)
    insert_edge!(graph, d_uElIn_v0, v0)
    insert_edge!(graph, d_uElOut_v1, v1)

    insert_edge!(graph, d_uPhIn_v3, v3)
    insert_edge!(graph, d_uPhOut_v2, v2)
    insert_edge!(graph, d_uElIn_v2, v2)
    insert_edge!(graph, d_uElOut_v3, v3)

    insert_edge!(graph, v0, d_v0_s0)
    insert_edge!(graph, v1, d_v1_s0)
    insert_edge!(graph, v2, d_v2_s1)
    insert_edge!(graph, v3, d_v3_s1)

    insert_edge!(graph, d_v0_s0, s0)
    insert_edge!(graph, d_v1_s0, s0)

    insert_edge!(graph, d_v2_s1, s1)
    insert_edge!(graph, d_v3_s1, s1)

    insert_edge!(graph, s0, d_s0_sum)
    insert_edge!(graph, s1, d_s1_sum)

    insert_edge!(graph, d_s0_sum, sum_node)
    insert_edge!(graph, d_s1_sum, sum_node)

    insert_edge!(graph, sum_node, d_exit)

    input = [gen_process_input(process) for _ in 1:1000]

    compton_function = get_compute_function(graph, process, machine, @__MODULE__)
    @test isapprox(compton_function.(input), compton_groundtruth.(input))

    graph_generated = graph(process)

    compton_function = get_compute_function(graph_generated, process, machine, @__MODULE__)
    @test isapprox(compton_function.(input), compton_groundtruth.(input))
end
=#

@testset "Equal results after optimization" for optimizer in [
    ReductionOptimizer(), RandomWalkOptimizer(MersenneTwister(0))
]
    @testset "Process $proc_str" for proc_str in ["ke->ke", "ke->kke", "ke->kkke"]
        model = QEDModel()
        process = parse_process(proc_str, model)
        machine = cpu_st()
        graph = graph(process)

        compute_function = get_compute_function(graph, process, machine, @__MODULE__)

        if (typeof(optimizer) <: RandomWalkOptimizer)
            optimize!(optimizer, graph, 100)
        elseif (typeof(optimizer) <: ReductionOptimizer)
            optimize_to_fixpoint!(optimizer, graph)
        end
        reduced_compute_function = get_compute_function(
            graph, process, machine, @__MODULE__
        )

        input = [gen_process_input(process) for _ in 1:100]

        @test isapprox(compute_function.(input), reduced_compute_function.(input))
    end
end
