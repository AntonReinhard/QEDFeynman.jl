import ComputableDAGs: compute_effort, children

"""
    compute_effort(t::ComputeTaskABC_S1)

Return the compute effort of an S1 task.
"""
compute_effort(t::ComputeTaskABC_S1)::Float64 = 11.0

"""
    compute_effort(t::ComputeTaskABC_S2)

Return the compute effort of an S2 task.
"""
compute_effort(t::ComputeTaskABC_S2)::Float64 = 12.0

"""
    compute_effort(t::ComputeTaskABC_U)

Return the compute effort of a U task.
"""
compute_effort(t::ComputeTaskABC_U)::Float64 = 1.0

"""
    compute_effort(t::ComputeTaskABC_V)

Return the compute effort of a V task.
"""
compute_effort(t::ComputeTaskABC_V)::Float64 = 6.0

"""
    compute_effort(t::ComputeTaskABC_Sum)

Return the compute effort of a Sum task. 

Note: This is a constant compute effort, even though sum scales with the number of its inputs. Since there is only ever a single sum node in a graph generated from the ABC-Model,
this doesn't matter.
"""
compute_effort(t::ComputeTaskABC_Sum)::Float64 = 1.0

"""
    children(::ComputeTaskABC_S1)

Return the number of children of a ComputeTaskABC_S1 (always 1).
"""
children(::ComputeTaskABC_S1) = 1

"""
    children(::ComputeTaskABC_S2)

Return the number of children of a ComputeTaskABC_S2 (always 2).
"""
children(::ComputeTaskABC_S2) = 2

"""
    children(::ComputeTaskABC_U)

Return the number of children of a ComputeTaskABC_U (always 1).
"""
children(::ComputeTaskABC_U) = 1

"""
    children(::ComputeTaskABC_V)

Return the number of children of a ComputeTaskABC_V (always 2).
"""
children(::ComputeTaskABC_V) = 2

"""
    children(::ComputeTaskABC_Sum)

Return the number of children of a ComputeTaskABC_Sum.
"""
children(t::ComputeTaskABC_Sum) = t.children_number

function add_child!(t::ComputeTaskABC_Sum)
    t.children_number += 1
    return nothing
end
