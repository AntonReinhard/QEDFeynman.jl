using Roots
using ForwardDiff
using StaticArrays
using Random

# TODO: reliably find out how many threads we're running with (nthreads() returns 1 when precompiling :/)
rng = [Random.MersenneTwister(0) for _ in 1:128]

####################
# CODE FROM HERE BORROWED FROM SOURCE: https://codebase.helmholtz.cloud/qedsandbox/QEDphasespaces.jl/
# use qedphasespaces directly once released
#
# quick and dirty implementation of the RAMBO algorithm
#
# reference:
# * https://cds.cern.ch/record/164736/files/198601282.pdf
# * https://www.sciencedirect.com/science/article/pii/0010465586901190
####################

function generate_initial_moms(ss, masses)
    E1 = (ss^2 + masses[1]^2 - masses[2]^2) / (2 * ss)
    E2 = (ss^2 + masses[2]^2 - masses[1]^2) / (2 * ss)

    rho1 = sqrt(E1^2 - masses[1]^2)
    rho2 = sqrt(E2^2 - masses[2]^2)

    return [SFourMomentum(E1, 0, 0, rho1), SFourMomentum(E2, 0, 0, -rho2)]
end

function Random.rand(rng::AbstractRNG, ::Random.SamplerType{SFourMomentum})
    return SFourMomentum(rand(rng, 4))
end
function Random.rand(rng::AbstractRNG, ::Random.SamplerType{NTuple{N, Float64}}) where {N}
    return Tuple(rand(rng, N))
end

function _transform_uni_to_mom(u1, u2, u3, u4)
    cth = 2 * u1 - 1
    sth = sqrt(1 - cth^2)
    phi = 2 * pi * u2
    q0 = -log(u3 * u4)
    qx = q0 * sth * cos(phi)
    qy = q0 * sth * sin(phi)
    qz = q0 * cth

    return SFourMomentum(q0, qx, qy, qz)
end

function _transform_uni_to_mom!(uni_mom, dest)
    u1, u2, u3, u4 = Tuple(uni_mom)
    cth = 2 * u1 - 1
    sth = sqrt(1 - cth^2)
    phi = 2 * pi * u2
    q0 = -log(u3 * u4)
    qx = q0 * sth * cos(phi)
    qy = q0 * sth * sin(phi)
    qz = q0 * cth

    return dest = SFourMomentum(q0, qx, qy, qz)
end

_transform_uni_to_mom(u1234::Tuple) = _transform_uni_to_mom(u1234...)
_transform_uni_to_mom(u1234::SFourMomentum) = _transform_uni_to_mom(Tuple(u1234))

function generate_massless_moms(rng, n::Int)
    a = Vector{SFourMomentum}(undef, n)
    rand!(rng, a)
    return map(_transform_uni_to_mom, a)
end

function generate_physical_massless_moms(rng, ss, n)
    r_moms = generate_massless_moms(rng, n)
    Q = sum(r_moms)
    M = sqrt(Q * Q)
    fac = -1 / M
    Qx = getX(Q)
    Qy = getY(Q)
    Qz = getZ(Q)
    bx = fac * Qx
    by = fac * Qy
    bz = fac * Qz
    gamma = getT(Q) / M
    a = 1 / (1 + gamma)
    x = ss / M

    i = 1
    while i <= n
        mom = r_moms[i]
        mom0 = getT(mom)
        mom1 = getX(mom)
        mom2 = getY(mom)
        mom3 = getZ(mom)

        bq = bx * mom1 + by * mom2 + bz * mom3

        p0 = x * (gamma * mom0 + bq)
        px = x * (mom1 + bx * mom0 + a * bq * bx)
        py = x * (mom2 + by * mom0 + a * bq * by)
        pz = x * (mom3 + bz * mom0 + a * bq * bz)

        r_moms[i] = SFourMomentum(p0, px, py, pz)
        i += 1
    end
    return r_moms
end

function _to_be_solved(xi, masses, p0s, ss)
    sum = 0.0
    for (i, E) in enumerate(p0s)
        sum += sqrt(masses[i]^2 + xi^2 * E^2)
    end
    return sum - ss
end

function _build_massive_momenta(xi, masses, massless_moms)
    vec = SFourMomentum[]
    i = 1
    while i <= length(massless_moms)
        massless_mom = massless_moms[i]
        k0 = sqrt(getT(massless_mom)^2 * xi^2 + masses[i]^2)

        kx = xi * getX(massless_mom)
        ky = xi * getY(massless_mom)
        kz = xi * getZ(massless_mom)

        push!(vec, SFourMomentum(k0, kx, ky, kz))

        i += 1
    end
    return vec
end

first_derivative(func) = x -> ForwardDiff.derivative(func, float(x))

function generate_physical_massive_moms(rng, ss, masses; x0 = 0.1)
    n = length(masses)
    massless_moms = generate_physical_massless_moms(rng, ss, n)
    energies = getT.(massless_moms)
    f = x -> _to_be_solved(x, masses, energies, ss)
    xi = find_zero((f, first_derivative(f)), x0, Roots.Newton())
    return _build_massive_momenta(xi, masses, massless_moms)
end
