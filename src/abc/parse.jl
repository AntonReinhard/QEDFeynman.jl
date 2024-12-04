# functions for importing DAGs from a file
regex_a = r"^[A-C]\d+$"                     # Regex for the initial particles
regex_c = r"^[A-C]\(([^']*),([^']*)\)$"     # Regex for the combinations of 2 particles
regex_m = r"^M\(([^']*),([^']*),([^']*)\)$" # Regex for the combinations of 3 particles
regex_plus = r"^\+$"                        # Regex for the sum

const PARTICLE_VALUE_SIZE::Int = 48
const FLOAT_SIZE::Int = 8

function _node_name(node::String, proc::GenericABCProcess)
    # first part of the name is the letter we want
    particle_type = if node[1] == 'A'
        ParticleA()
    elseif node[1] == 'B'
        ParticleB()
    elseif node[1] == 'C'
        ParticleC()
    else
        throw("cannot parse node name from $(node)")
    end

    # second part is the particle n
    particle_n = parse(Int, node[2:end])

    incoming_particles_of_type = count(p -> p == particle_type, incoming_particles(proc))
    dir_str = "i"
    if (particle_n > incoming_particles_of_type)
        particle_n -= incoming_particles_of_type
        dir_str = "o"
    end

    return String(particle_type) * dir_str * string(particle_n)
end

"""
    parse_nodes(input::AbstractString)

Parse the given string into a vector of strings containing each node.
"""
function parse_nodes(input::AbstractString)
    regex = r"'([^']*)'"
    matches = eachmatch(regex, input)
    output = [match.captures[1] for match in matches]
    return output
end

"""
    parse_edges(input::AbstractString)

Parse the given string into a vector of strings containing each edge. Currently unused since the entire graph can be read from just the node names.
"""
function parse_edges(input::AbstractString)
    regex = r"\('([^']*)', '([^']*)'\)"
    matches = eachmatch(regex, input)
    output = [(match.captures[1], match.captures[2]) for match in matches]
    return output
end

"""
    parse_dag(filename::String, proc::GenericABCProcess; verbose::Bool = false)

Read an ABC-model process from the given file. If `verbose` is set to true, print some progress information to stdout.

Returns a valid `DAG`.
"""
function parse_dag(filename::AbstractString, proc::GenericABCProcess, verbose::Bool=false)
    file = open(filename, "r")

    if (verbose)
        println("Opened file")
    end
    nodes_string = readline(file)
    nodes = parse_nodes(nodes_string)

    close(file)
    if (verbose)
        println("Read file")
    end

    graph = DAG()

    # estimate total number of nodes
    # try to slightly overestimate so no resizing is necessary
    # data nodes are not included in length(nodes) and there are a few more than compute nodes
    estimate_no_nodes = round(Int, length(nodes) * 4)
    if (verbose)
        println("Estimating ", estimate_no_nodes, " Nodes")
    end
    sizehint!(graph.nodes, estimate_no_nodes)

    sum_node = insert_node!(graph, ComputeTaskABC_Sum(0))
    global_data_out = insert_node!(graph, DataTask(FLOAT_SIZE))
    insert_edge!(graph, sum_node, global_data_out)

    # remember the data out nodes for connection
    data_out_nodes = Dict()

    if (verbose)
        println("Building graph")
    end
    number_of_nodes = 0
    nodesToRead = length(nodes)
    while !isempty(nodes)
        node = popfirst!(nodes)
        number_of_nodes += 1
        if (number_of_nodes % 100 == 0)
            if (verbose)
                percent = string(
                    round(100.0 * number_of_nodes / nodesToRead; digits=2), "%"
                )
                print("\rReading Nodes... $percent")
            end
        end
        if occursin(regex_a, node)
            name = _node_name(string(node), proc)

            # add nodes and edges for the state reading to u(P(Particle))
            data_in = insert_node!(graph, DataTask(PARTICLE_VALUE_SIZE), name) # read particle data node
            compute_P = insert_node!(graph, ComputeTaskABC_P()) # compute P node
            data_Pu = insert_node!(graph, DataTask(PARTICLE_VALUE_SIZE)) # transfer data from P to u (one ParticleValue object)
            compute_u = insert_node!(graph, ComputeTaskABC_U()) # compute U node
            data_out = insert_node!(graph, DataTask(PARTICLE_VALUE_SIZE)) # transfer data out from u (one ParticleValue object)

            insert_edge!(graph, data_in, compute_P)
            insert_edge!(graph, compute_P, data_Pu)
            insert_edge!(graph, data_Pu, compute_u)
            insert_edge!(graph, compute_u, data_out)

            # remember the data_out node for future edges
            data_out_nodes[node] = data_out
        elseif occursin(regex_c, node)
            capt = match(regex_c, node)

            in1 = capt.captures[1]
            in2 = capt.captures[2]

            compute_v = insert_node!(graph, ComputeTaskABC_V())
            data_out = insert_node!(graph, DataTask(PARTICLE_VALUE_SIZE))

            if (occursin(regex_c, in1))
                # put an S node after this input
                compute_S = insert_node!(graph, ComputeTaskABC_S1())
                data_S_v = insert_node!(graph, DataTask(PARTICLE_VALUE_SIZE))

                insert_edge!(graph, data_out_nodes[in1], compute_S)
                insert_edge!(graph, compute_S, data_S_v)

                insert_edge!(graph, data_S_v, compute_v)
            else
                insert_edge!(graph, data_out_nodes[in1], compute_v)
            end

            if (occursin(regex_c, in2))
                # i think the current generator only puts the combined particles in the first space, so this case might never be entered
                # put an S node after this input
                compute_S = insert_node!(graph, ComputeTaskABC_S1())
                data_S_v = insert_node!(graph, DataTask(PARTICLE_VALUE_SIZE))

                insert_edge!(graph, data_out_nodes[in2], compute_S)
                insert_edge!(graph, compute_S, data_S_v)

                insert_edge!(graph, data_S_v, compute_v)
            else
                insert_edge!(graph, data_out_nodes[in2], compute_v)
            end

            insert_edge!(graph, compute_v, data_out)
            data_out_nodes[node] = data_out

        elseif occursin(regex_m, node)
            # assume for now that only the first particle of the three is combined and the other two are "original" ones
            capt = match(regex_m, node)
            in1 = capt.captures[1]
            in2 = capt.captures[2]
            in3 = capt.captures[3]

            # in2 + in3 with a v
            compute_v = insert_node!(graph, ComputeTaskABC_V())
            data_v = insert_node!(graph, DataTask(PARTICLE_VALUE_SIZE))

            insert_edge!(graph, data_out_nodes[in2], compute_v)
            insert_edge!(graph, data_out_nodes[in3], compute_v)
            insert_edge!(graph, compute_v, data_v)

            # combine with the v of the combined other input
            compute_S2 = insert_node!(graph, ComputeTaskABC_S2())
            data_out = insert_node!(graph, DataTask(FLOAT_SIZE)) # output of a S2 task is only a float

            insert_edge!(graph, data_v, compute_S2)
            insert_edge!(graph, data_out_nodes[in1], compute_S2)
            insert_edge!(graph, compute_S2, data_out)

            insert_edge!(graph, data_out, sum_node)
            add_child!(task(sum_node))
        elseif occursin(regex_plus, node)
            if (verbose)
                println("\rReading Nodes Complete    ")
                println("Added ", length(graph.nodes), " nodes")
            end
        else
            @assert false ("Unknown node '$node' while reading from file $filename")
        end
    end

    #put all nodes into dirty nodes set
    graph.dirty_nodes = copy(graph.nodes)

    if (verbose)
        println("Generating the graph's properties")
    end
    graph.properties = GraphProperties(graph)

    if (verbose)
        println("Done")
    end

    # TODO: validate that the graph actually fits the given proc
    # don't actually need to read the edges
    return graph
end

"""
    parse_process(string::AbstractString, model::ABCModel)

Parse a string representation of a process, such as "AB->ABBB" into the corresponding [`GenericABCProcess`](@ref).
"""
function parse_process(str::AbstractString, model::ABCModel)
    if !(contains(str, "->"))
        throw("Did not find -> while parsing process \"$str\"")
    end

    (in_str, out_str) = split(str, "->")

    if (isempty(in_str) || isempty(out_str))
        throw("Process (\"$str\") input or output part is empty!")
    end

    incoming_particles = Vector{AbstractParticleType}()
    outgoing_particles = Vector{AbstractParticleType}()

    for (particle_vector, s) in
        ((incoming_particles, in_str), (outgoing_particles, out_str))
        for c in s
            if c == 'A'
                push!(particle_vector, ParticleA())
            elseif c == 'B'
                push!(particle_vector, ParticleB())
            elseif c == 'C'
                push!(particle_vector, ParticleC())
            else
                throw("Encountered unknown characters in the process \"$str\"")
            end
        end
    end

    return GenericABCProcess(tuple(incoming_particles...), tuple(outgoing_particles...))
end
