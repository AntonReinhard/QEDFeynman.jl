import ComputableDAGs: compute_effort, children
# compute effort numbers were measured on a home pc system using likwid

"""
    compute_effort(t::ComputeTaskQED_S1)

Return the compute effort of an S1 task.
"""
compute_effort(t::ComputeTaskQED_S1)::Float64 = 475.0

"""
    compute_effort(t::ComputeTaskQED_S2)

Return the compute effort of an S2 task.
"""
compute_effort(t::ComputeTaskQED_S2)::Float64 = 505.0

"""
    compute_effort(t::ComputeTaskQED_U)

Return the compute effort of a U task.
"""
compute_effort(t::ComputeTaskQED_U)::Float64 = (291.0 + 467.0 + 16.0 + 17.0) / 4.0 # The exact FLOPS count depends heavily on the type of particle, take an average value here

"""
    compute_effort(t::ComputeTaskQED_V)

Return the compute effort of a V task.
"""
compute_effort(t::ComputeTaskQED_V)::Float64 = (1150.0 + 764.0 + 828.0) / 3.0

"""
    compute_effort(t::ComputeTaskQED_Sum)

Return the compute effort of a Sum task. 

Note: This is a constant compute effort, even though sum scales with the number of its inputs. Since there is only ever a single sum node in a graph generated from the QED-Model,
this doesn't matter.
"""
compute_effort(t::ComputeTaskQED_Sum)::Float64 = 1.0

"""
    children(::ComputeTaskQED_S1)

Return the number of children of a ComputeTaskQED_S1 (always 1).
"""
children(::ComputeTaskQED_S1) = 1

"""
    children(::ComputeTaskQED_S2)

Return the number of children of a ComputeTaskQED_S2 (always 2).
"""
children(::ComputeTaskQED_S2) = 2

"""
    children(::ComputeTaskQED_U)

Return the number of children of a ComputeTaskQED_U (always 1).
"""
children(::ComputeTaskQED_U) = 1

"""
    children(::ComputeTaskQED_V)

Return the number of children of a ComputeTaskQED_V (always 2).
"""
children(::ComputeTaskQED_V) = 2

"""
    children(::ComputeTaskQED_Sum)

Return the number of children of a ComputeTaskQED_Sum.
"""
children(t::ComputeTaskQED_Sum) = t.children_number

function add_child!(t::ComputeTaskQED_Sum)
    t.children_number += 1
    return nothing
end
