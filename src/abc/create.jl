using QEDbase
using Random
using Roots
using ForwardDiff

ComputeTaskABC_Sum() = ComputeTaskABC_Sum(0)

function _svector_from_type_in(processDescription::GenericABCProcess, type, particles)
    if haskey(incoming_particles(processDescription), type)
        return SVector{incoming_particles(processDescription)[type], type}(
            filter(x -> typeof(x) <: type, particles)
        )
    end
    return SVector{0, type}()
end

function _svector_from_type_out(processDescription::GenericABCProcess, type, particles)
    if haskey(outgoing_particles(processDescription), type)
        return SVector{outgoing_particles(processDescription)[type], type}(
            filter(x -> typeof(x) <: type, particles)
        )
    end
    return SVector{0, type}()
end

"""
    gen_process_input(proc::GenericABCProcess)

Return a ProcessInput of randomly generated [`ABCParticle`](@ref)s from a [`GenericABCProcess`](@ref). The process description can be created manually or parsed from a string using [`parse_process`](@ref).

Note: This uses RAMBO to create a valid process with conservation of momentum and energy.
"""
function gen_process_input(proc::GenericABCProcess)
    mass_sum = 0
    input_masses = Vector{Float64}()
    for particle in incoming_particles(proc)
        mass_sum += mass(particle)
        push!(input_masses, mass(particle))
    end
    output_masses = Vector{Float64}()
    for particle in outgoing_particles(proc)
        mass_sum += mass(particle)
        push!(output_masses, mass(particle))
    end

    # add some extra random mass to allow for some momentum
    mass_sum +=
        rand(rng[threadid()]) *
        (number_incoming_particles(proc) + number_outgoing_particles(proc))

    initial_momenta = generate_initial_moms(mass_sum, input_masses)
    final_momenta = generate_physical_massive_moms(rng[threadid()], mass_sum, output_masses)

    return PhaseSpacePoint(
        proc,
        PerturbativeABC(),
        PhasespaceDefinition(SphericalCoordinateSystem(), ElectronRestFrame()), # this is a bit nonsensical but also isn't currently being used anyways
        tuple(initial_momenta...),
        tuple(final_momenta...),
    )
end
