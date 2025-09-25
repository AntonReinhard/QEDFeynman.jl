using Combinatorics

import Base.==

"""
    FeynmanParticle

Representation of a particle for use in [`FeynmanDiagram`](@ref)s. Consist of the `ParticleStateful` type and an id.
"""
struct FeynmanParticle
    particle::Type{<:ParticleStateful}
    id::Int
end

"""
    FeynmanVertex

Representation of a vertex in a [`FeynmanDiagram`](@ref). Stores two input [`FeynmanParticle`](@ref)s and one output.
"""
struct FeynmanVertex
    in1::FeynmanParticle
    in2::FeynmanParticle
    out::FeynmanParticle
end

"""
    FeynmanTie

Representation of a "tie" in a [`FeynmanDiagram`](@ref). A tie ties two virtual particles in a diagram together and thus represent an inner line of the diagram. Not all inner lines are [`FeynmanTie`](@ref)s, in fact, a connected diagram only ever has exactly one tie.
"""
struct FeynmanTie
    in1::FeynmanParticle
    in2::FeynmanParticle
end

"""
    FeynmanDiagram

Representation of a feynman diagram. It consists of its initial input/output particles, and a vector of sets of [`FeynmanVertex`](@ref)s. The vertices are to be applied level by level.
A [`FeynmanVertex`](@ref) will always be at the lowest level possible, i.e. the lowest level at which all input particles for it exist.
The [`FeynmanTie`](@ref) represents the final inner edge of the diagram.
"""
struct FeynmanDiagram
    vertices::Vector{Set{FeynmanVertex}}
    tie::Ref{Union{FeynmanTie, Missing}}
    particles::Vector{FeynmanParticle}
    type_ids::Dict{Type, Int64} # lut for number of used ids for a particle type
end

"""
    FeynmanDiagram(pd::ScatteringProcess)

Create an initial [`FeynmanDiagram`](@ref) with only its initial particles set and no vertices or ties.

Use [`gen_diagrams`](@ref) to generate all possible diagrams from this one.
"""
function FeynmanDiagram(pd::ScatteringProcess)
    parts = Vector{FeynmanParticle}()

    ids = Dict{Type, Int64}()
    for type in types(model(pd))
        for i in 1:number_particles(pd, type)
            push!(parts, FeynmanParticle(type, i))
        end
        ids[type] = number_particles(pd, type)
    end

    return FeynmanDiagram([], missing, parts, ids)
end

function particle_after_tie(p::FeynmanParticle, t::FeynmanTie)
    if p == t.in1 || p == t.in2
        return FeynmanParticle(ParticleStateful{Incoming, Electron, SFourMomentum}, -1) # placeholder particle and id for tied particles
    end
    return p
end

function vertex_after_tie(v::FeynmanVertex, t::FeynmanTie)
    return FeynmanVertex(
        particle_after_tie(v.in1, t),
        particle_after_tie(v.in2, t),
        particle_after_tie(v.out, t),
    )
end

function vertex_after_tie(v::FeynmanVertex, t::Missing)
    return v
end

function vertex_set_after_tie(vs::Set{FeynmanVertex}, t::FeynmanTie)
    return Set{FeynmanVertex}(vertex_after_tie(v, t) for v in vs)
end

function vertex_set_after_tie(vs::Set{FeynmanVertex}, t::Missing)
    return vs
end

function vertex_set_after_tie(
        vs::Set{FeynmanVertex}, t1::Union{FeynmanTie, Missing}, t2::Union{FeynmanTie, Missing}
    )
    return Set{FeynmanVertex}(vertex_after_tie(vertex_after_tie(v, t1), t2) for v in vs)
end

"""
    String(p::FeynmanParticle)

Return a string representation of the [`FeynmanParticle`](@ref) in a format that is readable by [`type_index_from_name`](@ref).
"""
function String(p::FeynmanParticle)
    return "$(String(p.particle))$(String(particle_direction(p.particle)))$(p.id)"
end

function Base.hash(v::FeynmanVertex)
    return hash(v.in1) * hash(v.in2)
end

function Base.hash(t::FeynmanTie)
    return hash(t.in1) * hash(t.in2)
end

function Base.hash(d::FeynmanDiagram)
    return hash((d.vertices, d.particles))
end

function ==(v1::FeynmanVertex, v2::FeynmanVertex)
    return (v1.in1 == v2.in1 && v1.in2 == v2.in1) || (v1.in2 == v2.in1 && v1.in1 == v2.in2)
end

function ==(t1::FeynmanTie, t2::FeynmanTie)
    return (t1.in1 == t2.in1 && t1.in2 == t2.in1) || (t1.in2 == t2.in1 && t1.in1 == t2.in2)
end

function ==(d1::FeynmanDiagram, d2::FeynmanDiagram)
    if (!ismissing(d1.tie[]) && ismissing(d2.tie[])) ||
            (ismissing(d1.tie[]) && !ismissing(d2.tie[]))
        return false
    end
    if d1.particles != d2.particles
        return false
    end
    if length(d1.vertices) != length(d2.vertices)
        return false
    end

    # TODO can i prove that this works?
    for (v1, v2) in zip(d1.vertices, d2.vertices)
        if vertex_set_after_tie(v1, d1.tie[], d2.tie[]) !=
                vertex_set_after_tie(v2, d1.tie[], d2.tie[])
            return false
        end
    end
    return true

    #=return isequal.(
        vertex_set_after_tie(d1.vertices, d1.tie, d2.tie),
        vertex_set_after_tie(d2.vertices, d1.tie, d2.tie),
    )=#
end

function Base.copy(fd::FeynmanDiagram)
    return FeynmanDiagram(
        deepcopy(fd.vertices), copy(fd.tie[]), deepcopy(fd.particles), copy(fd.type_ids)
    )
end

"""
    id_for_type(d::FeynmanDiagram, t::Type{<:ParticleStateful})

Return the highest id of any particle of the given type in the diagram + 1.
"""
function id_for_type(d::FeynmanDiagram, t::Type{<:ParticleStateful})
    return d.type_ids[t] + 1
end

"""
    can_apply_vertex(particles::Vector{FeynmanParticle}, vertex::FeynmanVertex)

Return true if the given [`FeynmanVertex`](@ref) can be applied to the given particles, i.e. both input particles of the vertex are in the vector and the output particle is not.
"""
function can_apply_vertex(particles::Vector{FeynmanParticle}, vertex::FeynmanVertex)
    return vertex.in1 in particles && vertex.in2 in particles && !(vertex.out in particles)
end

"""
    apply_vertex!(particles::Vector{FeynmanParticle}, vertex::FeynmanVertex)

Apply a [`FeynmanVertex`](@ref) to the given vector of [`FeynmanParticle`](@ref)s.
"""
function apply_vertex!(particles::Vector{FeynmanParticle}, vertex::FeynmanVertex)
    #@assert can_apply_vertex(particles, vertex)
    length_before = length(particles)
    filter!(x -> x != vertex.in1 && x != vertex.in2, particles)
    push!(particles, vertex.out)
    #@assert length(particles) == length_before - 1
    return nothing
end

"""
    can_apply_tie(particles::Vector{FeynmanParticle}, tie::FeynmanTie)

Return true if the given [`FeynmanTie`](@ref) can be applied to the given particles, i.e. both input particles of the tie are in the vector.
"""
function can_apply_tie(particles::Vector{FeynmanParticle}, tie::FeynmanTie)
    return tie.in1 in particles && tie.in2 in particles
end

"""
    apply_tie!(particles::Vector{FeynmanParticle}, tie::FeynmanTie)

Apply a [`FeynmanTie`](@ref) to the given vector of [`FeynmanParticle`](@ref)s.
"""
function apply_tie!(particles::Vector{FeynmanParticle}, tie::FeynmanTie)
    @assert length(particles) == 2
    @assert can_apply_tie(particles, tie)
    @assert can_tie(tie.in1.particle, tie.in2.particle)
    empty!(particles)
    @assert length(particles) == 0
    return nothing
end

function apply_tie!(::Vector{FeynmanParticle}, ::Missing)
    return nothing
end

"""
    get_particles(fd::FeynmanDiagram, level::Int)

Return a vector of the particles after applying the vertices and tie of the diagram up to the given level. If no level is given, apply all. The tie comes last and is its own "level".
"""
function get_particles(fd::FeynmanDiagram, level::Int = -1)
    if level == -1
        level = length(fd.vertices) + 1
    end

    working_particles = copy(fd.particles)
    for l in 1:length(fd.vertices)
        if l > level
            break
        end
        for v in fd.vertices[l]
            apply_vertex!(working_particles, v)
        end
    end

    if (level > length(fd.vertices))
        apply_tie!(working_particles, fd.tie[])
    end

    return working_particles
end

"""
    add_vertex!(fd::FeynmanDiagram, vertex::FeynmanVertex)

Add the given vertex to the diagram, at the earliest level possible.
"""
function add_vertex!(fd::FeynmanDiagram, vertex::FeynmanVertex)
    for i in eachindex(fd.vertices)
        if (can_apply_vertex(get_particles(fd, i - 1), vertex))
            push!(fd.vertices[i], vertex)
            fd.type_ids[vertex.out.particle] += 1
            return nothing
        end
    end

    if !can_apply_vertex(get_particles(fd), vertex)
        @assert false "Can't add vertex $vertex to diagram $(get_particles(fd))"
    end

    push!(fd.vertices, Set{FeynmanVertex}())
    push!(fd.vertices[end], vertex)

    fd.type_ids[vertex.out.particle] += 1

    return nothing
end

"""
    add_vertex(fd::FeynmanDiagram, vertex::FeynmanVertex)

Add the given vertex to the diagram, at the earliest level possible. Return the new diagram without muting the given one.
"""
function add_vertex(fd::FeynmanDiagram, vertex::FeynmanVertex)
    newfd = copy(fd)
    add_vertex!(newfd, vertex)
    return newfd
end

"""
    add_tie!(fd::FeynmanDiagram, tie::FeynmanTie)

Add the given tie to the diagram, always at the last level.
"""
function add_tie!(fd::FeynmanDiagram, tie::FeynmanTie)
    if !can_apply_tie(get_particles(fd), tie)
        @assert false "Can't add tie $tie to diagram"
    end

    fd.tie[] = tie
    #=
        @assert length(fd.vertices) >= 2
        #if the last vertex is involved in the tie and alone, lower it one level down
        if (length(fd.vertices[end]) != 1)
            return nothing
        end

        vert = fd.vertices[end][1]
        if (vert != vertex_after_tie(vert, tie))
            return nothing
        end

        pop!(fd.vertices)
        push!(fd.vertices[end], vert)
    =#
    return nothing
end

"""
    add_tie(fd::FeynmanDiagram, tie::FeynmanTie)

Add the given tie to the diagram, at the earliest level possible. Return the new diagram without muting the given one.
"""
function add_tie(fd::FeynmanDiagram, tie::FeynmanTie)
    newfd = copy(fd)
    add_tie!(newfd, tie)
    return newfd
end

"""
    isvalid(fd::FeynmanDiagram)

Return whether the given diagram is valid. A diagram is valid iff the following are true:
- After applying all vertices and the tie, there are no more particles left
- The diagram is connected
"""
function isvalid(fd::FeynmanDiagram)
    if ismissing(fd.tie[])
        # diagram is connected iff there is one tie
        return false
    end

    if get_particles(fd) != []
        return false
    end

    return true
end

"""
    possible_vertices(fd::FeynmanDiagram)

Return a vector of all possible vertices that can be applied to the diagram at its current state.
"""
function possible_vertices(fd::FeynmanDiagram)
    possibilities = Vector{FeynmanVertex}()
    fully_generated_particles = get_particles(fd)

    min_level = max(0, length(fd.vertices) - 1)
    for l in min_level:length(fd.vertices)
        particles = get_particles(fd, l)
        for i in 1:length(particles)
            for j in (i + 1):length(particles)
                p1 = particles[i]
                p2 = particles[j]
                if (caninteract(p1.particle, p2.particle))
                    interaction_res = propagation_result(
                        interaction_result(p1.particle, p2.particle)
                    )
                    v = FeynmanVertex(
                        p1,
                        p2,
                        FeynmanParticle(interaction_res, id_for_type(fd, interaction_res)),
                    )
                    #@assert !(v.out in particles) "$v is in $fd"
                    if !can_apply_vertex(fully_generated_particles, v)
                        continue
                    end
                    push!(possibilities, v)
                end
            end
        end
        if (!isempty(possibilities))
            return possibilities
        end
    end
    return possibilities
end

"""
    can_tie(p1::Type, p2::Type)

For two given `QEDParticle` types, return whether they can be tied together.

They can be tied iff one is the `propagation_result` of the other, or if both are photons, in which case their direction does not matter.
"""
function can_tie(p1::Type, p2::Type)
    if p1 == propagation_result(p2)
        return true
    end
    if (p1 <: PhotonStateful && p2 <: PhotonStateful)
        return true
    end
    return false
end

"""
    possible_tie(fd::FeynmanDiagram)

Return a possible tie or `missing` for the diagram at its current state.
"""
function possible_tie(fd::FeynmanDiagram)
    particles = get_particles(fd)
    if (length(particles) != 2)
        return missing
    end

    if (particles[1] in fd.particles || particles[2] in fd.particles)
        return missing
    end

    tie = FeynmanTie(particles[1], particles[2])
    if (can_apply_tie(particles, tie))
        return tie
    end
    return missing
end

function remove_duplicates(compare_set::Set{FeynmanDiagram})
    result = Set()

    while !isempty(compare_set)
        x = pop!(compare_set)
        # we know there will only be one duplicate if any, so search for that and delete it
        for y in compare_set
            if x == y
                delete!(compare_set, y)
                break
            end
        end
        push!(result, x)
    end

    return result
end

"""
    is_compton(fd::FeynmanDiagram)

Returns true iff the given feynman diagram is an (empty) diagram of a compton process like ke->k^ne
"""
function is_compton(fd::FeynmanDiagram)
    return fd.type_ids[ParticleStateful{Incoming, Electron, SFourMomentum}] == 1 &&
        fd.type_ids[ParticleStateful{Outgoing, Electron, SFourMomentum}] == 1 &&
        fd.type_ids[ParticleStateful{Incoming, Positron, SFourMomentum}] == 0 &&
        fd.type_ids[ParticleStateful{Outgoing, Positron, SFourMomentum}] == 0 &&
        fd.type_ids[ParticleStateful{Incoming, Photon, SFourMomentum}] >= 1 &&
        fd.type_ids[ParticleStateful{Outgoing, Photon, SFourMomentum}] >= 1
end

"""
    gen_compton_diagram_from_order(order::Vector{Int}, inFerm, outFerm, n::Int, m::Int)

Helper function for [`gen_compton_diagrams`](@ref). Generates a single diagram for the given order and n input and m output photons.
"""
function gen_compton_diagram_from_order(order::Vector{Int}, inFerm, outFerm, n::Int, m::Int)
    photons = vcat(
        [FeynmanParticle(ParticleStateful{Incoming, Photon, SFourMomentum}, i) for i in 1:n],
        [FeynmanParticle(ParticleStateful{Outgoing, Photon, SFourMomentum}, i) for i in 1:m],
    )

    new_diagram = FeynmanDiagram(
        [],
        missing,
        [inFerm, outFerm, photons...],
        Dict{Type, Int64}(
            ParticleStateful{Incoming, Electron, SFourMomentum} => 1,
            ParticleStateful{Outgoing, Electron, SFourMomentum} => 1,
            ParticleStateful{Incoming, Photon, SFourMomentum} => n,
            ParticleStateful{Outgoing, Photon, SFourMomentum} => m,
        ),
    )

    left_index = 1
    right_index = length(order)

    iterations = 1

    while left_index <= right_index
        # left side
        v_left = FeynmanVertex(
            FeynmanParticle(ParticleStateful{Incoming, Electron, SFourMomentum}, iterations),
            photons[order[left_index]],
            FeynmanParticle(
                ParticleStateful{Incoming, Electron, SFourMomentum}, iterations + 1
            ),
        )
        left_index += 1
        add_vertex!(new_diagram, v_left)

        if (left_index > right_index)
            break
        end

        # right side
        v_right = FeynmanVertex(
            FeynmanParticle(ParticleStateful{Outgoing, Electron, SFourMomentum}, iterations),
            photons[order[right_index]],
            FeynmanParticle(
                ParticleStateful{Outgoing, Electron, SFourMomentum}, iterations + 1
            ),
        )
        right_index -= 1
        add_vertex!(new_diagram, v_right)

        iterations += 1
    end

    @assert possible_tie(new_diagram) !== missing
    add_tie!(new_diagram, possible_tie(new_diagram))
    return new_diagram
end

"""
    gen_compton_diagram_from_order_one_side(order::Vector{Int}, inFerm, outFerm, n::Int, m::Int)

Helper function for [`gen_compton_diagrams`](@ref). Generates a single diagram for the given order and n input and m output photons.
"""
function gen_compton_diagram_from_order_one_side(
        order::Vector{Int}, inFerm, outFerm, n::Int, m::Int
    )
    photons = vcat(
        [FeynmanParticle(ParticleStateful{Incoming, Photon, SFourMomentum}, i) for i in 1:n],
        [FeynmanParticle(ParticleStateful{Outgoing, Photon, SFourMomentum}, i) for i in 1:m],
    )

    new_diagram = FeynmanDiagram(
        [],
        missing,
        [inFerm, outFerm, photons...],
        Dict{Type, Int64}(
            ParticleStateful{Incoming, Electron, SFourMomentum} => 1,
            ParticleStateful{Outgoing, Electron, SFourMomentum} => 1,
            ParticleStateful{Incoming, Photon, SFourMomentum} => n,
            ParticleStateful{Outgoing, Photon, SFourMomentum} => m,
        ),
    )

    left_index = 1
    right_index = length(order)

    iterations = 1

    while left_index <= right_index
        # left side
        v_left = FeynmanVertex(
            FeynmanParticle(ParticleStateful{Incoming, Electron, SFourMomentum}, iterations),
            photons[order[left_index]],
            FeynmanParticle(
                ParticleStateful{Incoming, Electron, SFourMomentum}, iterations + 1
            ),
        )
        left_index += 1
        add_vertex!(new_diagram, v_left)

        if (left_index > right_index)
            break
        end

        # only once on the right side
        if (iterations == 1)
            # right side
            v_right = FeynmanVertex(
                FeynmanParticle(
                    ParticleStateful{Outgoing, Electron, SFourMomentum}, iterations
                ),
                photons[order[right_index]],
                FeynmanParticle(
                    ParticleStateful{Outgoing, Electron, SFourMomentum}, iterations + 1
                ),
            )
            right_index -= 1
            add_vertex!(new_diagram, v_right)
        end

        iterations += 1
    end

    ps = get_particles(new_diagram)
    @assert length(ps) == 2
    add_tie!(new_diagram, FeynmanTie(ps[1], ps[2]))
    return new_diagram
end

"""
    gen_compton_diagrams(n::Int, m::Int)

Special case diagram generation for Compton processes, i.e., processes of the form k^ne->k^me
"""
function gen_compton_diagrams(n::Int, m::Int)
    inFerm = FeynmanParticle(ParticleStateful{Incoming, Electron, SFourMomentum}, 1)
    outFerm = FeynmanParticle(ParticleStateful{Outgoing, Electron, SFourMomentum}, 1)

    perms = [permutations([i for i in 1:(n + m)])...]

    diagrams = [Vector{FeynmanDiagram}() for i in 1:nthreads()]
    @threads for order in perms
        push!(
            diagrams[threadid()],
            gen_compton_diagram_from_order(order, inFerm, outFerm, n, m),
        )
    end

    return vcat(diagrams...)
end

"""
    gen_compton_diagrams_one_side(n::Int, m::Int)

Special case diagram generation for Compton processes, i.e., processes of the form k^ne->k^me, but generating from one end, yielding larger diagrams
"""
function gen_compton_diagrams_one_side(n::Int, m::Int)
    inFerm = FeynmanParticle(ParticleStateful{Incoming, Electron, SFourMomentum}, 1)
    outFerm = FeynmanParticle(ParticleStateful{Outgoing, Electron, SFourMomentum}, 1)

    perms = [permutations([i for i in 1:(n + m)])...]

    diagrams = [Vector{FeynmanDiagram}() for i in 1:nthreads()]
    @threads for order in perms
        push!(
            diagrams[threadid()],
            gen_compton_diagram_from_order_one_side(order, inFerm, outFerm, n, m),
        )
    end

    return vcat(diagrams...)
end

"""
    gen_diagrams(fd::FeynmanDiagram)

From a given feynman diagram in its initial state, e.g. when created through the [`FeynmanDiagram`](@ref)`(pd::ProcessDescription)` constructor, generate and return all possible [`FeynmanDiagram`](@ref)s that describe that process.
"""
function gen_diagrams(fd::FeynmanDiagram)
    if is_compton(fd)
        return gen_compton_diagrams(
            fd.type_ids[ParticleStateful{Incoming, Photon, SFourMomentum}],
            fd.type_ids[ParticleStateful{Outgoing, Photon, SFourMomentum}],
        )
    end

    throw(error("Unimplemented for non-compton!"))

    #=
    working = Set{FeynmanDiagram}()
    results = Set{FeynmanDiagram}()

    push!(working, fd)

    # we know there will be particle_number - 2 vertices, followed by 1 tie
    n_particles = length(fd.particles)
    n_vertices = n_particles - 2

    # doing this in iterations should reduce the intermediate number of diagrams by hash collisions
    for _ in 1:n_vertices
        next_iter_set = Set{FeynmanDiagram}()

        while !isempty(working)
            d = pop!(working)

            possibilities = possible_vertices(d)
            for v in possibilities
                push!(next_iter_set, add_vertex(d, v))
            end
        end

        working = next_iter_set
    end

    # add the tie
    for d in working
        tie = possible_tie(d)
        if ismissing(tie)
            continue
        end
        add_tie!(d, tie)
        if isvalid(d)
            push!(results, d)
        end
    end

    return remove_duplicates(results)
    =#
end
