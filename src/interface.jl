import QEDbase.AbstractParticle

"""
    AbstractPhysicsModel

Base type for a model, e.g. ABC-Model or QED. This is used to dispatch many functions.
"""
abstract type AbstractPhysicsModel end

"""
    ParticleValue{ParticleType <: AbstractParticleStateful}

A struct describing a particle during a calculation of a Feynman Diagram, together with the value that's being calculated. `AbstractParticleStateful` is the type from the QEDbase package.

`sizeof(ParticleValue())` = 48 Byte
"""
struct ParticleValue{ParticleType<:AbstractParticleStateful,ValueType}
    p::ParticleType
    v::ValueType
end

"""
TBW

particle value + spin/pol info, only used on the external legs (u tasks)
"""
struct ParticleValueSP{
    ParticleType<:AbstractParticleStateful,SP<:AbstractSpinOrPolarization,ValueType
}
    p::ParticleType
    v::ValueType
    sp::SP
end

"""
    AbstractProcessDescription <: AbstractProblemInstance

Base type for particle scattering process descriptions. An object of this type of a corresponding [`AbstractPhysicsModel`](@ref) should uniquely identify a scattering process in that model.

See also: [`parse_process`](@ref)
"""
abstract type AbstractProcessDescription end

#TODO: i don't think giving this a base type is a good idea, the input type should just be returned of some function, allowing anything as an input type
"""
    AbstractProcessInput

Base type for process inputs. An object of this type contains the input values (e.g. momenta) of the particles in a process.

See also: [`gen_process_input`](@ref)
"""
abstract type AbstractProcessInput end

"""
    interaction_result(t1::AbstractParticleStateful, t2::AbstractParticleStateful)

Interface function that must be implemented for `AbstractParticleStateful`s for all pairs of particle_species that can occur in the model.
It should return the result particle (stateful) type when the two given particles interact.
"""
function interaction_result end

"""
    types(::AbstractPhysicsModel)

Interface function that must be implemented for every subtype of [`AbstractPhysicsModel`](@ref), returning a `Vector` of the available particle types in the model.
"""
function types end

"""
    get_particle(::AbstractProcessInput, t::Type, n::Int)

Interface function that must be implemented for every subtype of [`AbstractProcessInput`](@ref).
Returns the `n`th particle of type `t`.
"""
function get_particle end

"""
    parse_process(::AbstractString, ::AbstractPhysicsModel)

Interface function that must be implemented for every subtype of [`AbstractPhysicsModel`](@ref).
Returns a `ProcessDescription` object.
"""
function parse_process end

"""
    gen_process_input(::AbstractProcessDescription)

Interface function that must be implemented for every specific [`AbstractProcessDescription`](@ref).
Returns a randomly generated and valid corresponding `ProcessInput`.
"""
function gen_process_input end

"""
    model(::AbstractProcessDescription)
    model(::AbstractProcessInput)

Return the model of this process description or input.
"""
function model end

"""
    type_from_name(model::AbstractPhysicsModel, name::String)

For a name of a particle in the given `AbstractPhysicsModel`, return the particle's `Type` and index as a tuple. The input string can be expetced to be of the form \"<name><index>\".
"""
function type_index_from_name end

"""
    part_from_x(type::Type, index::Int, x::AbstractProcessInput)

Return the [`ParticleValue`](@ref) of the given type of particle with the given `index` from the given process input.

Function is wrapped into a `FunctionCall` in `gen_input_assignment_code`.
"""
part_from_x(type::Type, index::Int, x::AbstractProcessInput) =
    ParticleValue{type,ComplexF64}(get_particle(x, type, index), one(ComplexF64))
