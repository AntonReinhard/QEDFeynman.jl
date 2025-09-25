using StaticArrays

"""
    mass(t::Type{T}) where {T <: ABCParticle}

Return the mass (at rest) of the given particle type.
"""
QEDbase.mass(p::ABCParticle) = mass(typeof(p))

QEDbase.mass(::Type{ParticleA}) = 1.0
QEDbase.mass(::Type{ParticleB}) = 1.0
QEDbase.mass(::Type{ParticleC}) = 0.0

"""
    interaction_result(
        p1::AbstractParticleStateful{<:ParticleDirection, <:ABCParticle},
        p2::AbstractParticleStateful{<:ParticleDirection, <:ABCParticle},
    )

For 2 given (non-equal) particle types, return the third of ABC.
"""
function interaction_result(
        p1::AbstractParticleStateful{<:ParticleDirection, <:ABCParticle},
        p2::AbstractParticleStateful{<:ParticleDirection, <:ABCParticle},
    )
    PS_T = parameterless(typeof(p1))
    MOM_T = typeof(momentum(p1))
    @assert particle_species(p1) != particle_species(p2)
    if particle_species(p1) != ParticleA() && particle_species(p2) != ParticleA()
        return PS_T{Outgoing, ParticleA, MOM_T}
    elseif particle_species(p1) != ParticleB() && particle_species(p2) != ParticleB()
        return PS_T{Outgoing, ParticleB, MOM_T}
    else
        return PS_T{Outgoing, ParticleC, MOM_T}
    end
end

"""
    types(::ABCModel)

Return a Vector of the possible types of particle in the [`ABCModel`](@ref).
"""
function types(::ABCModel)
    return [
        ParticleStateful{Incoming, ParticleA, SFourMomentum},
        ParticleStateful{Incoming, ParticleB, SFourMomentum},
        ParticleStateful{Incoming, ParticleC, SFourMomentum},
        ParticleStateful{Outgoing, ParticleA, SFourMomentum},
        ParticleStateful{Outgoing, ParticleB, SFourMomentum},
        ParticleStateful{Outgoing, ParticleC, SFourMomentum},
    ]
end

"""
    square(p::AbstractParticleStateful{Dir, ABCParticle})

Return the square of the particle's momentum as a `Float` value.

Takes 7 effective FLOP.
"""
function square(p::AbstractParticleStateful{<:ParticleDirection, <:ABCParticle})
    return getMass2(momentum(p))
end

"""
    ABC_inner_edge(p::AbstractParticleStateful{Dir, ABCParticle})

Return the factor of the inner edge with the given (virtual) particle.

Takes 10 effective FLOP. (3 here + 7 in square(p))
"""
function ABC_inner_edge(p::AbstractParticleStateful{<:ParticleDirection, <:ABCParticle})
    res = 1.0 / (square(p) - mass(particle_species(p))^2)
    return res
end

"""
    ABC_outer_edge(p::AbstractParticleStateful{Dir, ABCParticle})

Return the factor of the outer edge with the given (real) particle.

Takes 0 effective FLOP.
"""
function ABC_outer_edge(::AbstractParticleStateful{D, <:ABCParticle}) where {D}
    return 1.0
end

"""
    ABC_vertex()

Return the factor of a vertex.

Takes 0 effective FLOP since it's constant.
"""
function ABC_vertex()
    i = 1.0
    lambda = 1.0 / 137.0
    return i * lambda
end

"""
    ABC_conserve_momentum(p1::ABCParticle, p2::ABCParticle)

Calculate and return a new particle from two given interacting ones at a vertex.

Takes 4 effective FLOP.
"""
function ABC_conserve_momentum(
        p1::AbstractParticleStateful{<:ParticleDirection, <:ABCParticle},
        p2::AbstractParticleStateful{<:ParticleDirection, <:ABCParticle},
    )
    t3 = interaction_result(p1, p2)
    p1_mom = momentum(p1)
    if (is_outgoing(p1))
        p1_mom *= -1
    end
    p2_mom = momentum(p2)
    if (is_outgoing(p2))
        p2_mom *= -1
    end
    p3 = t3(p1_mom + p2_mom)
    return p3
end

function Base.copy(process::GenericABCProcess)
    return GenericABCProcess(copy(process.inParticles), copy(process.outParticles))
end

model(::GenericABCProcess) = ABCModel()

function type_index_from_name(::ABCModel, name::String)
    if startswith(name, "Ai")
        return (ParticleStateful{Incoming, ParticleA, SFourMomentum}, parse(Int, name[3:end]))
    elseif startswith(name, "Ao")
        return (ParticleStateful{Outgoing, ParticleA, SFourMomentum}, parse(Int, name[3:end]))
    elseif startswith(name, "Bi")
        return (ParticleStateful{Incoming, ParticleB, SFourMomentum}, parse(Int, name[3:end]))
    elseif startswith(name, "Bo")
        return (ParticleStateful{Outgoing, ParticleB, SFourMomentum}, parse(Int, name[3:end]))
    elseif startswith(name, "Ci")
        return (ParticleStateful{Incoming, ParticleC, SFourMomentum}, parse(Int, name[3:end]))
    elseif startswith(name, "Co")
        return (ParticleStateful{Outgoing, ParticleC, SFourMomentum}, parse(Int, name[3:end]))
    else
        throw("Invalid name for a particle in the ABC model")
    end
end

function String(::Type{PS}) where {DIR, P <: ABCParticle, PS <: AbstractParticleStateful{DIR, P}}
    return String(P)
end
function String(::Type{ParticleA})
    return "A"
end
function String(::Type{ParticleB})
    return "B"
end
function String(::Type{ParticleC})
    return "C"
end
String(p::ABCParticle) = String(typeof(p))

function ComputableDAGs.input_type(p::GenericABCProcess)
    in_t = _assemble_tuple_type(incoming_particles(p), Incoming())
    out_t = _assemble_tuple_type(outgoing_particles(p), Outgoing())
    return AbstractPhaseSpacePoint{
        typeof(p),
        PerturbativeABC,
        PhasespaceDefinition{SphericalCoordinateSystem, ElectronRestFrame},
        <:Tuple{in_t...},
        <:Tuple{out_t...},
    }
end
