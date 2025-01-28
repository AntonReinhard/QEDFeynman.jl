"""
    QEDModel <: AbstractPhysicsModel

Singleton definition for identification of the QED-Model.
"""
struct QEDModel <: AbstractPhysicsModel end

"""
    ComputeTaskQED_S1 <: AbstractComputeTask

S task with a single child.
"""
struct ComputeTaskQED_S1 <: AbstractComputeTask end

"""
    ComputeTaskQED_S2 <: AbstractComputeTask

S task with two children.
"""
struct ComputeTaskQED_S2 <: AbstractComputeTask end

"""
    ComputeTaskQED_V <: AbstractComputeTask

v task with two children.
"""
struct ComputeTaskQED_V <: AbstractComputeTask end

"""
    ComputeTaskQED_U <: AbstractComputeTask

u task with a single child.
"""
struct ComputeTaskQED_U <: AbstractComputeTask end

"""
    ComputeTaskQED_Sum <: AbstractComputeTask

Task that sums all its inputs, n children.
"""
mutable struct ComputeTaskQED_Sum <: AbstractComputeTask
    children_number::Int
end

"""
    QED_TASKS

Constant vector of all tasks of the QED-Model.
"""
QED_TASKS = [
    ComputeTaskQED_S1,
    ComputeTaskQED_S2,
    ComputeTaskQED_V,
    ComputeTaskQED_U,
    ComputeTaskQED_Sum,
]
