"""
    ABCModel <: AbstractPhysicsModel

Singleton definition for identification of the ABC-Model.
"""
struct ABCModel <: AbstractPhysicsModel end

"""
    PerturbativeABC <: AbstractModelDefinition

The model being used for the ABC model.
"""
struct PerturbativeABC <: QEDbase.AbstractModelDefinition end

"""
    ABCParticle

Base type for all particles in the [`ABCModel`](@ref).
"""
abstract type ABCParticle <: AbstractParticleType end

"""
    ParticleA <: ABCParticle

An 'A' particle in the ABC Model.
"""
struct ParticleA <: ABCParticle end

"""
    ParticleB <: ABCParticle

A 'B' particle in the ABC Model.
"""
struct ParticleB <: ABCParticle end

"""
    ParticleC <: ABCParticle

A 'C' particle in the ABC Model.
"""
struct ParticleC <: ABCParticle end

"""
    ComputeTaskABC_S1 <: AbstractComputeTask

S task with a single child.
"""
struct ComputeTaskABC_S1 <: AbstractComputeTask end

"""
    ComputeTaskABC_S2 <: AbstractComputeTask

S task with two children.
"""
struct ComputeTaskABC_S2 <: AbstractComputeTask end

"""
    ComputeTaskABC_P <: AbstractComputeTask

P task with no children.
"""
struct ComputeTaskABC_P <: AbstractComputeTask end

"""
    ComputeTaskABC_V <: AbstractComputeTask

v task with two children.
"""
struct ComputeTaskABC_V <: AbstractComputeTask end

"""
    ComputeTaskABC_U <: AbstractComputeTask

u task with a single child.
"""
struct ComputeTaskABC_U <: AbstractComputeTask end

"""
    ComputeTaskABC_Sum <: AbstractComputeTask

Task that sums all its inputs, n children.
"""
mutable struct ComputeTaskABC_Sum <: AbstractComputeTask
    children_number::Int
end

"""
    ABC_TASKS

Constant vector of all tasks of the ABC-Model.
"""
ABC_TASKS = [
    ComputeTaskABC_S1,
    ComputeTaskABC_S2,
    ComputeTaskABC_P,
    ComputeTaskABC_V,
    ComputeTaskABC_U,
    ComputeTaskABC_Sum,
]
