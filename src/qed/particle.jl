using QEDprocesses
using StaticArrays

const e = sqrt(4π / 137)

QEDbase.is_incoming(::Type{<:ParticleStateful{Incoming}}) = true
QEDbase.is_outgoing(::Type{<:ParticleStateful{Outgoing}}) = true
QEDbase.is_incoming(::Type{<:ParticleStateful{Outgoing}}) = false
QEDbase.is_outgoing(::Type{<:ParticleStateful{Incoming}}) = false

function QEDbase.particle_direction(
    ::Type{<:ParticleStateful{DIR}}
) where {DIR<:ParticleDirection}
    return DIR()
end
function QEDbase.particle_species(
    ::Type{<:ParticleStateful{DIR,SPECIES}}
) where {DIR<:ParticleDirection,SPECIES<:AbstractParticleType}
    return SPECIES()
end

function spin_or_pol(
    process::ScatteringProcess, type::Type{ParticleStateful{DIR,SPECIES,EL}}, n::Int
) where {DIR<:ParticleDirection,SPECIES<:AbstractParticleType,EL<:AbstractFourMomentum}
    i = 0
    c = n
    for p in particles(process, DIR())
        i += 1
        if p == SPECIES()
            c -= 1
        end
        if c == 0
            break
        end
    end

    if c != 0 || n <= 0
        throw(
            InvalidInputError(
                "could not get $n-th spin/pol of $(DIR()) $species, does not exist"
            ),
        )
    end

    return spin_pols(process, DIR())[i]
end

function ComputableDAGs.input_type(p::ScatteringProcess)
    in_t = _assemble_tuple_type(incoming_particles(p), Incoming())
    out_t = _assemble_tuple_type(outgoing_particles(p), Outgoing())
    return AbstractPhaseSpacePoint{
        typeof(p),
        PerturbativeQED,
        PhasespaceDefinition{SphericalCoordinateSystem,ElectronRestFrame},
        <:Tuple{in_t...},
        <:Tuple{out_t...},
    }
end

ValueType = Union{BiSpinor,AdjointBiSpinor,DiracMatrix,SLorentzVector{Float64},ComplexF64}
APS = AbstractParticleStateful

# incoming vs. outgoing of same fermion makes photon
function interaction_result(
    p1::APS{Incoming,P}, ::APS{Outgoing,P}
) where {P<:Union{Electron,Positron}}
    return parameterless(typeof(p1)){Incoming,Photon,typeof(momentum(p1))}
end
function interaction_result(
    p1::APS{Outgoing,P}, ::APS{Incoming,P}
) where {P<:Union{Electron,Positron}}
    return parameterless(typeof(p1)){Incoming,Photon,typeof(momentum(p1))}
end

# electron + positron of same direction makes photon
function interaction_result(
    p1::APS{D,Electron}, ::APS{D,Positron}
) where {D<:ParticleDirection}
    return parameterless(typeof(p1)){Incoming,Photon,typeof(momentum(p1))}
end

# electron/positron + photon makes the same fermion again in reverse direction
function interaction_result(
    p1::APS{<:ParticleDirection,P}, ::APS{<:ParticleDirection,Photon}
) where {P<:Union{Electron,Positron}}
    return parameterless(typeof(p1)){
        typeof(reverse(particle_direction(p1))),P,typeof(momentum(p1))
    }
end

# commutativity (photon always on right side, positron on right side as long as the other isn't a photon to prevent infinite recursion)
function interaction_result(
    p1::APS{<:ParticleDirection,Positron},
    p2::APS{<:ParticleDirection,<:Union{Electron,Positron}},
)
    return interaction_result(p2, p1)
end
function interaction_result(p1::APS{<:ParticleDirection,Photon}, p2::APS)
    return interaction_result(p2, p1)
end

# but prevent stack overflow
function interaction_result(
    p1::APS{<:ParticleDirection,Photon}, p2::APS{<:ParticleDirection,Photon}
)
    @assert false "Invalid interaction between particles $p1 and $p2"
end

"""
    types(::QEDModel)

Return a Vector of the possible types of particle in the [`QEDModel`](@ref).
"""
function types(::QEDModel)
    return [
        ParticleStateful{Incoming,Photon,SFourMomentum},
        ParticleStateful{Outgoing,Photon,SFourMomentum},
        ParticleStateful{Incoming,Electron,SFourMomentum},
        ParticleStateful{Outgoing,Electron,SFourMomentum},
        ParticleStateful{Incoming,Positron,SFourMomentum},
        ParticleStateful{Outgoing,Positron,SFourMomentum},
    ]
end

# type piracy?
String(::Type{Incoming}) = "Incoming"
String(::Type{Outgoing}) = "Outgoing"

String(::Type{PolX}) = "polx"
String(::Type{PolY}) = "poly"

String(::Type{SpinUp}) = "spinup"
String(::Type{SpinDown}) = "spindown"

String(::Incoming) = "i"
String(::Outgoing) = "o"

function String(::Type{<:ParticleStateful{DIR,Photon}}) where {DIR<:ParticleDirection}
    return "k"
end
function String(::Type{<:ParticleStateful{DIR,Electron}}) where {DIR<:ParticleDirection}
    return "e"
end
function String(::Type{<:ParticleStateful{DIR,Positron}}) where {DIR<:ParticleDirection}
    return "p"
end

"""
    caninteract(T1::Type{<:ParticleStateful}, T2::Type{<:ParticleStateful})

For two given `ParticleStateful` types, return whether they can interact at a vertex. This is equivalent to `!issame(T1, T2)`.

See also: [`issame`](@ref) and [`interaction_result`](@ref)
"""
function caninteract(
    T1::Type{<:ParticleStateful{D1,S1}}, T2::Type{<:ParticleStateful{D2,S2}}
) where {
    D1<:ParticleDirection,
    S1<:AbstractParticleType,
    D2<:ParticleDirection,
    S2<:AbstractParticleType,
}
    if (T1 == T2)
        return false
    end
    if (S1 == Photon && S2 == Photon)
        return false
    end

    for (P1, P2) in [(T1, T2), (T2, T1)]
        if (
            P1 <: ParticleStateful{Incoming,Electron} &&
            P2 <: ParticleStateful{Outgoing,Positron}
        )
            return false
        end
        if (
            P1 <: ParticleStateful{Outgoing,Electron} &&
            P2 <: ParticleStateful{Incoming,Positron}
        )
            return false
        end
    end

    return true
end

function type_index_from_name(::QEDModel, name::String)
    if startswith(name, "ki")
        return (ParticleStateful{Incoming,Photon,SFourMomentum}, parse(Int, name[3:end]))
    elseif startswith(name, "ko")
        return (ParticleStateful{Outgoing,Photon,SFourMomentum}, parse(Int, name[3:end]))
    elseif startswith(name, "ei")
        return (ParticleStateful{Incoming,Electron,SFourMomentum}, parse(Int, name[3:end]))
    elseif startswith(name, "eo")
        return (ParticleStateful{Outgoing,Electron,SFourMomentum}, parse(Int, name[3:end]))
    elseif startswith(name, "pi")
        return (ParticleStateful{Incoming,Positron,SFourMomentum}, parse(Int, name[3:end]))
    elseif startswith(name, "po")
        return (ParticleStateful{Outgoing,Positron,SFourMomentum}, parse(Int, name[3:end]))
    else
        throw("Invalid name for a particle in the QED model")
    end
end

"""
    issame(T1::Type{<:ParticleStateful}, T2::Type{<:ParticleStateful})

For two given `ParticleStateful` types, return whether they are equivalent for the purpose of a Feynman Diagram. That means e.g. an `Incoming` `AntiFermion` is the same as an `Outgoing` `Fermion`. This is equivalent to `!caninteract(T1, T2)`.

See also: [`caninteract`](@ref) and [`interaction_result`](@ref)
"""
function issame(T1::Type{<:ParticleStateful}, T2::Type{<:ParticleStateful})
    return !caninteract(T1, T2)
end

"""
    QED_vertex()

Return the factor of a vertex in a QED feynman diagram.
"""
@inline function QED_vertex()::SLorentzVector{DiracMatrix}
    # Peskin-Schroeder notation
    return -1im * e * gamma()
end

@inline function QED_inner_edge(p::ParticleStateful)
    if is_outgoing(p)
        return propagator(particle_species(p), momentum(p))
    else
        return propagator(particle_species(p), -momentum(p))
    end
end

"""
    QED_conserve_momentum(p1::ParticleStateful, p2::ParticleStateful)

Calculate and return a new particle from two given interacting ones at a vertex.
"""
function QED_conserve_momentum(p1::AbstractParticleStateful, p2::AbstractParticleStateful)
    P3 = interaction_result(p1, p2)
    p1_mom = momentum(p1)
    if (is_outgoing(p1))
        p1_mom *= -1
    end
    p2_mom = momentum(p2)
    if (is_outgoing(p2))
        p2_mom *= -1
    end

    p3_mom = p1_mom + p2_mom
    if (is_incoming(particle_direction(P3)))
        p3_mom *= -1
    end

    p3 = parameterless(typeof(p1))(particle_direction(P3), particle_species(P3), p3_mom)

    #=print("$(is_incoming(p1) ? "in" : "out") $(particle_species(p1)) + ")
    print("$(is_incoming(p2) ? "in" : "out") $(particle_species(p2)) -> ")
    println("$(is_incoming(p3) ? "in" : "out") $(particle_species(p3))")
    println("$(momentum(p1)[1]) + $(momentum(p2)[1]) -> $(p3_mom[1])")=#

    return p3
end

"""
    model(::AbstractProcessDescription)

Return the model of this process description.
"""
model(::ScatteringProcess) = QEDModel()
model(::PhaseSpacePoint) = QEDModel()

function get_particle(
    input::PhaseSpacePoint, t::Type{ParticleStateful{DIR,SPECIES}}, n::Int
) where {DIR<:ParticleDirection,SPECIES<:AbstractParticleType}
    i = 0
    for p in particles(input, DIR())
        if p isa t
            i += 1
            if i == n
                return p
            end
        end
    end
    @assert false "Invalid type given"
end
