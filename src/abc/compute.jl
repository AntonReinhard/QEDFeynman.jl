using AccurateArithmetic
using StaticArrays

construction_string(::ParticleA) = "ParticleA()"
construction_string(::ParticleB) = "ParticleB()"
construction_string(::ParticleC) = "ParticleC()"

function ComputableDAGs.input_expr(
    instance::GenericABCProcess, name::String, psp_symbol::Symbol
)
    (type, index) = type_index_from_name(ABCModel(), name)

    return Meta.parse(
        "ParticleValue(
            $type(momentum($psp_symbol, $(construction_string(particle_direction(type))), $(construction_string(particle_species(type))), Val($index))),
            1 + 0.0im,
        )",
    )
end

"""
    compute(::ComputeTaskABC_U, data::ParticleValue)

Compute an outer edge. Return the particle value with the same particle and the value multiplied by an ABC_outer_edge factor.

1 FLOP.
"""
function ComputableDAGs.compute(
    ::ComputeTaskABC_U, data::ParticleValue{P}
)::ParticleValue{P} where {P}
    return ParticleValue(data.p, data.v * ABC_outer_edge(data.p))
end

"""
    compute(::ComputeTaskABC_V, data1::ParticleValue, data2::ParticleValue)

Compute a vertex. Preserve momentum and particle types (AB->C etc.) to create resulting particle, multiply values together and times a vertex factor.

6 FLOP.
"""
function ComputableDAGs.compute(
    ::ComputeTaskABC_V, data1::ParticleValue{P1}, data2::ParticleValue{P2}
)::ParticleValue where {P1,P2}
    p3 = ABC_conserve_momentum(data1.p, data2.p)
    dataOut = ParticleValue(p3, data1.v * ABC_vertex() * data2.v)
    return dataOut
end

"""
    compute(::ComputeTaskABC_S2, data1::ParticleValue, data2::ParticleValue)

Compute a final inner edge (2 input particles, no output particle).

For valid inputs, both input particles should have the same momenta at this point.

12 FLOP.
"""
function ComputableDAGs.compute(
    ::ComputeTaskABC_S2, data1::ParticleValue{P}, data2::ParticleValue{P}
)::Float64 where {P}
    #=
    @assert isapprox(abs(data1.p.momentum.E), abs(data2.p.momentum.E), rtol = 0.001, atol = sqrt(eps())) "E: $(data1.p.momentum.E) vs. $(data2.p.momentum.E)"
    @assert isapprox(data1.p.momentum.px, -data2.p.momentum.px, rtol = 0.001, atol = sqrt(eps())) "px: $(data1.p.momentum.px) vs. $(data2.p.momentum.px)"
    @assert isapprox(data1.p.momentum.py, -data2.p.momentum.py, rtol = 0.001, atol = sqrt(eps())) "py: $(data1.p.momentum.py) vs. $(data2.p.momentum.py)"
    @assert isapprox(data1.p.momentum.pz, -data2.p.momentum.pz, rtol = 0.001, atol = sqrt(eps())) "pz: $(data1.p.momentum.pz) vs. $(data2.p.momentum.pz)"
    =#
    inner = ABC_inner_edge(propagated_particle(data1.p))
    return data1.v * inner * data2.v
end

"""
    compute(::ComputeTaskABC_S1, data::ParticleValue)

Compute inner edge (1 input particle, 1 output particle).

11 FLOP.
"""
function ComputableDAGs.compute(::ComputeTaskABC_S1, data::ParticleValue{P}) where {P}
    return ParticleValue(data.p, data.v * ABC_inner_edge(propagated_particle(data.p)))
end

"""
    compute(::ComputeTaskABC_Sum, data...)
    compute(::ComputeTaskABC_Sum, data::AbstractArray)

Compute a sum over the vector. Use an algorithm that accounts for accumulated errors in long sums with potentially large differences in magnitude of the summands.

Linearly many FLOP with growing data.
"""
function ComputableDAGs.compute(::ComputeTaskABC_Sum, data...)::Float64
    return sum_kbn([data...])
end

function ComputableDAGs.compute(::ComputeTaskABC_Sum, data::AbstractArray)::Float64
    s = 0.0im
    for d in data
        s += d
    end
    return s
end
