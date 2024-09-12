using StaticArrays

construction_string(::Electron) = "Electron()"
construction_string(::Positron) = "Positron()"
construction_string(::Photon) = "Photon()"

construction_string(::PolX) = "PolX()"
construction_string(::PolY) = "PolY()"
construction_string(::SpinUp) = "SpinUp()"
construction_string(::SpinDown) = "SpinDown()"

function ComputableDAGs.input_expr(
    instance::ScatteringProcess, name::String, psp_symbol::Symbol
)
    (type, index) = type_index_from_name(QEDModel(), name)

    return Meta.parse(
        "ParticleValueSP(
            $type(momentum($psp_symbol, $(construction_string(particle_direction(type))), $(construction_string(particle_species(type))), Val($index))),
            0.0im,
            $(construction_string(spin_or_pol(instance, type, index))),
        )",
    )
end

"""
    compute(::ComputeTaskQED_U, data::ParticleValueSP)

Compute an outer edge. Return the particle value with the same particle and the value multiplied by an outer_edge factor.
"""
function ComputableDAGs.compute(
    ::ComputeTaskQED_U, data::ParticleValueSP{P,SP,V}
) where {P<:ParticleStateful,V<:ValueType,SP<:AbstractSpinOrPolarization}
    part::P = data.p
    state = base_state(
        particle_species(part), particle_direction(part), momentum(part), SP()
    )
    return ParticleValue{P,typeof(state)}(
        data.p,
        state, # will return a SLorentzVector{ComplexF64}, BiSpinor or AdjointBiSpinor
    )
end

"""
    compute(::ComputeTaskQED_V, data1::ParticleValue, data2::ParticleValue)

Compute a vertex. Preserve momentum and particle types (e + gamma->p etc.) to create resulting particle, multiply values together and times a vertex factor.
"""
function ComputableDAGs.compute(
    ::ComputeTaskQED_V, data1::ParticleValue{P1,V1}, data2::ParticleValue{P2,V2}
) where {P1<:ParticleStateful,P2<:ParticleStateful,V1<:ValueType,V2<:ValueType}
    p3 = QED_conserve_momentum(data1.p, data2.p)
    state = QED_vertex()
    if (typeof(data1.v) <: AdjointBiSpinor)
        state = data1.v * state
    else
        state = state * data1.v
    end
    if (typeof(data2.v) <: AdjointBiSpinor)
        state = data2.v * state
    else
        state = state * data2.v
    end

    #println("$(particle_species(p3)) with $(momentum(p3))")
    dataOut = ParticleValue(p3, state)
    return dataOut
end

"""
    compute(::ComputeTaskQED_S2, data1::ParticleValue, data2::ParticleValue)

Compute a final inner edge (2 input particles, no output particle).

For valid inputs, both input particles should have the same momenta at this point.

12 FLOP.
"""
function ComputableDAGs.compute(
    ::ComputeTaskQED_S2, data1::ParticleValue{P1,V1}, data2::ParticleValue{P2,V2}
) where {
    D1<:ParticleDirection,
    D2<:ParticleDirection,
    S1<:Union{Electron,Positron},
    S2<:Union{Electron,Positron},
    V1<:ValueType,
    V2<:ValueType,
    EL<:AbstractFourMomentum,
    P1<:ParticleStateful{D1,S1,EL},
    P2<:ParticleStateful{D2,S2,EL},
}
    inner1 = QED_inner_edge(data1.p)
    inner2 = QED_inner_edge(data2.p)

    #=println(
        "$(particle_direction(data1.p)): $(inner1[1, 1]), $(particle_direction(data2.p)): $(inner2[1, 1])",
    )=#

    # i'm pretty sure this is not universally true
    inner = is_incoming(data1.p) ? inner1 : inner2
    println("$(inner[1, 1])")

    #@assert isapprox(inner1, inner2, rtol=sqrt(eps()), atol=sqrt(eps())) "$(data1.p) vs. $(data2.p)"

    # inner edge is just a "scalar", data1 and data2 are bispinor/adjointbispinnor, need to keep correct order
    if typeof(data1.v) <: BiSpinor
        return (data2.v)::AdjointBiSpinor * inner * (data1.v)::BiSpinor
    else
        return (data1.v)::AdjointBiSpinor * inner * (data2.v)::BiSpinor
    end
end

function ComputableDAGs.compute(
    ::ComputeTaskQED_S2,
    data1::ParticleValue{ParticleStateful{D1,Photon},V1},
    data2::ParticleValue{ParticleStateful{D2,Photon},V2},
) where {D1<:ParticleDirection,D2<:ParticleDirection,V1<:ValueType,V2<:ValueType}
    # TODO: assert that data1 and data2 are opposites
    @assert isapprox(
        momentum(data1.p), momentum(data2.p), rtol=sqrt(eps()), atol=sqrt(eps())
    ) "$(momentum(data1.p)) vs. $(momentum(data2.p))"

    inner = QED_inner_edge(data1.p)
    #println("inner(s2): $(inner[1, 1])")
    # inner edge is just a scalar, data1 and data2 are photon states that are just Complex numbers here
    return data1.v * inner * data2.v
end

"""
    compute(::ComputeTaskQED_S1, data::ParticleValue)

Compute inner edge (1 input particle, 1 output particle).
"""
function ComputableDAGs.compute(
    ::ComputeTaskQED_S1, data::ParticleValue{P,V}
) where {P<:ParticleStateful,V<:ValueType}
    inner = QED_inner_edge(data.p)
    println("inner(s1): $(inner[1, 1])")
    new_p = propagated_particle(data.p)
    # inner edge is just a scalar, can multiply from either side
    if typeof(data.v) <: BiSpinor
        return ParticleValue(new_p, inner * data.v)
    else
        return ParticleValue(new_p, data.v * inner)
    end
end

"""
    compute(::ComputeTaskQED_Sum, data...)
    compute(::ComputeTaskQED_Sum, data::AbstractArray)

Compute a sum over the vector. Use an algorithm that accounts for accumulated errors in long sums with potentially large differences in magnitude of the summands.

Linearly many FLOP with growing data.
"""
function ComputableDAGs.compute(::ComputeTaskQED_Sum, data...)::ComplexF64
    # TODO: want to use sum_kbn here but it doesn't seem to support ComplexF64, do it element-wise?
    println("summing $data")
    s = 0.0im
    for d in data
        s += d
    end
    return s
end

function ComputableDAGs.compute(::ComputeTaskQED_Sum, data::AbstractArray)::ComplexF64
    # TODO: want to use sum_kbn here but it doesn't seem to support ComplexF64, do it element-wise?
    s = 0.0im
    for d in data
        s += d
    end
    return s
end
