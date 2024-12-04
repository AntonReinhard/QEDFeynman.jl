ComputeTaskQED_Sum() = ComputeTaskQED_Sum(0)

function _svector_from_type(processDescription::ScatteringProcess, type, particles)
    if haskey(incoming_particles(processDescription), type)
        return SVector{incoming_particles(processDescription)[type],type}(
            filter(x -> typeof(x) <: type, particles)
        )
    end
    if haskey(outgoing_particles(processDescription), type)
        return SVector{outgoing_particles(processDescription)[type],type}(
            filter(x -> typeof(x) <: type, particles)
        )
    end
end

"""
    gen_process_input(processDescription::ScatteringProcess)

Return a `PhaseSpacePoint` of randomly generated particles from a `QEDprocesses.ScatteringProcess`. The process description can be created manually or parsed from a string using [`parse_process`](@ref).

Note: This uses RAMBO to create a valid process with conservation of momentum and energy.
"""
function gen_process_input(processDescription::ScatteringProcess)
    mass_sum = 0
    input_masses = Vector{Float64}()
    for particle in incoming_particles(processDescription)
        mass_sum += mass(particle)
        push!(input_masses, mass(particle))
    end
    output_masses = Vector{Float64}()
    for particle in outgoing_particles(processDescription)
        mass_sum += mass(particle)
        push!(output_masses, mass(particle))
    end

    # add some extra random mass to allow for some momentum
    mass_sum += rand(rng[threadid()]) * (length(input_masses) + length(output_masses))

    initial_momenta = generate_initial_moms(mass_sum, input_masses)
    final_momenta = generate_physical_massive_moms(rng[threadid()], mass_sum, output_masses)

    return PhaseSpacePoint(
        processDescription,
        PerturbativeQED(),
        PhasespaceDefinition(SphericalCoordinateSystem(), ElectronRestFrame()),
        tuple(initial_momenta...),
        tuple(final_momenta...),
    )
end

"""
    graph(process_description::ScatteringProcess)

For a given `QEDprocesses.ScatteringProcess`, return the `DAG` that computes it.
"""
function ComputableDAGs.graph(process_description::ScatteringProcess)
    initial_diagram = FeynmanDiagram(process_description)
    diagrams = gen_diagrams(initial_diagram)

    graph = DAG()

    COMPLEX_SIZE = sizeof(ComplexF64)
    PARTICLE_VALUE_SIZE = 96.0

    # TODO: Not all diagram outputs should always be summed at the end, if they differ by fermion exchange they need to be diffed
    # Should not matter for n-Photon Compton processes though
    sum_node = insert_node!(graph, ComputeTaskQED_Sum(0))
    global_data_out = insert_node!(graph, DataTask(COMPLEX_SIZE))
    insert_edge!(graph, sum_node, global_data_out)

    # remember the data out nodes for connection
    data_out_nodes = Dict()

    for particle in initial_diagram.particles
        # generate data in and U tasks
        data_in = insert_node!(graph, DataTask(PARTICLE_VALUE_SIZE), String(particle)) # read particle data node
        compute_u = insert_node!(graph, ComputeTaskQED_U()) # compute U node
        data_out = insert_node!(graph, DataTask(PARTICLE_VALUE_SIZE)) # transfer data out from u (one ParticleValue object)

        insert_edge!(graph, data_in, compute_u)
        insert_edge!(graph, compute_u, data_out)

        # remember the data_out node for future edges
        data_out_nodes[String(particle)] = data_out
    end

    # TODO: this should be parallelizable somewhat easily
    for diagram in diagrams
        tie = diagram.tie[]

        # handle the vertices
        for vertices in diagram.vertices
            for vertex in vertices
                data_in1 = data_out_nodes[String(vertex.in1)]
                data_in2 = data_out_nodes[String(vertex.in2)]

                compute_V = insert_node!(graph, ComputeTaskQED_V()) # compute vertex

                insert_edge!(graph, data_in1, compute_V)
                insert_edge!(graph, data_in2, compute_V)

                data_V_out = insert_node!(graph, DataTask(PARTICLE_VALUE_SIZE))

                insert_edge!(graph, compute_V, data_V_out)

                if (vertex.out == tie.in1 || vertex.out == tie.in2)
                    # out particle is part of the tie -> there will be an S2 task with it later, don't make S1 task
                    data_out_nodes[String(vertex.out)] = data_V_out
                    continue
                end

                # otherwise, add S1 task
                compute_S1 = insert_node!(graph, ComputeTaskQED_S1()) # compute propagator

                insert_edge!(graph, data_V_out, compute_S1)

                data_S1_out = insert_node!(graph, DataTask(PARTICLE_VALUE_SIZE))

                insert_edge!(graph, compute_S1, data_S1_out)

                # overrides potentially different nodes from previous diagrams, which is intentional
                data_out_nodes[String(vertex.out)] = data_S1_out
            end
        end

        # handle the tie
        data_in1 = data_out_nodes[String(tie.in1)]
        data_in2 = data_out_nodes[String(tie.in2)]

        compute_S2 = insert_node!(graph, ComputeTaskQED_S2())

        data_S2 = insert_node!(graph, DataTask(PARTICLE_VALUE_SIZE))

        insert_edge!(graph, data_in1, compute_S2)
        insert_edge!(graph, data_in2, compute_S2)

        insert_edge!(graph, compute_S2, data_S2)

        insert_edge!(graph, data_S2, sum_node)
        add_child!(task(sum_node))
    end

    return graph
end
