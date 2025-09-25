using QEDprocesses

# add type overload for number_particles function
@inline function QEDprocesses.number_particles(
        proc_def::QEDbase.AbstractProcessDefinition, ::Type{PS}
    ) where {
        DIR <: QEDbase.ParticleDirection,
        PT <: QEDbase.AbstractParticleType,
        EL <: AbstractFourMomentum,
        PS <: ParticleStateful{DIR, PT, EL},
    }
    return QEDprocesses.number_particles(proc_def, DIR(), PT())
end
