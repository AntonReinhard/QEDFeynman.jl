# This file contains a definition of a generic ABC process on the AbstractProcessDefinition interface from QEDbase.
# This is a bit of an interface abuse (since the ABC model is not QED, and the particles don't have spins or polarizations), but it works for the test model.

"""
    GenericABCProcess <: AbstractProcessDefinition

"""
struct GenericABCProcess{INT, OUTT} <:
    AbstractProcessDefinition where {INT <: Tuple, OUTT <: Tuple}
    incoming_particles::INT
    outgoing_particles::OUTT

    """
        GenericABCProcess(
            incoming_particles::Tuple{AbstractParticleType},
            outgoing_particles::Tuple{AbstractParticleType},
        )

    Constructor for a GenericABCProcess with the given incoming and outgoing particles.
    """
    function GenericABCProcess(
            incoming_particles::INT, outgoing_particles::OUTT
        ) where {INT <: Tuple, OUTT <: Tuple}
        _assert_particle_type_tuple(incoming_particles)
        _assert_particle_type_tuple(outgoing_particles)

        return new{INT, OUTT}(incoming_particles, outgoing_particles)
    end
end

function QEDbase.incoming_particles(proc::GenericABCProcess)
    return proc.incoming_particles
end
function QEDbase.outgoing_particles(proc::GenericABCProcess)
    return proc.outgoing_particles
end
function QEDbase.incoming_spin_pols(proc::GenericABCProcess)
    throw("abc model particles do not have spin or polarizations")
end
function QEDbase.outgoing_spin_pols(proc::GenericABCProcess)
    throw("abc model particles do not have spin or polarizations")
end

_assert_particle_type_tuple(::Tuple{}) = nothing
function _assert_particle_type_tuple(t::Tuple{ABCParticle, Vararg})
    return _assert_particle_type_tuple(t[2:end])
end
function _assert_particle_type_tuple(t::Any)
    throw(
        InvalidInputError(
            "invalid input, provide a tuple of ABCParticles to construct a GenericABCProcess",
        ),
    )
end

_particle_to_letter(::ParticleA) = "A"
_particle_to_letter(::ParticleB) = "B"
_particle_to_letter(::ParticleC) = "C"
