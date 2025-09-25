construction_string(::Incoming) = "Incoming()"
construction_string(::Outgoing) = "Outgoing()"

reverse(::Incoming) = Outgoing()
reverse(::Outgoing) = Incoming()

"""
    propagated_particle(p::AbstractParticleStateful)

Returns the same particle but with inversed particle_direction.
"""
function propagated_particle(p::AbstractParticleStateful)
    return parameterless(typeof(p))(
        reverse(particle_direction(p)), particle_species(p), momentum(p)
    )
end

function interaction_result(p1::AbstractParticleStateful, p2::AbstractParticleStateful)
    return @assert false "Invalid interaction between particles $p1 and $p2"
end

# recursion termination: base case
@inline _assemble_tuple_type(::Tuple{}, ::ParticleDirection) = ()

# function assembling the correct type information for the tuple of ParticleStatefuls in a phasespace point constructed from momenta
@inline function _assemble_tuple_type(
        particle_types::Tuple{SPECIES_T, Vararg{AbstractParticleType}}, dir::DIR_T
    ) where {SPECIES_T <: AbstractParticleType, DIR_T <: ParticleDirection}
    return (
        AbstractParticleStateful{DIR_T, SPECIES_T},
        _assemble_tuple_type(particle_types[2:end], dir)...,
    )
end
