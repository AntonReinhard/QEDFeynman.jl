"""
    parse_process(string::AbstractString, model::QEDModel)

Parse a string representation of a process, such as "ke->ke" into the corresponding `QEDProcessDescription`.
"""
function parse_process(
        str::AbstractString,
        model::QEDModel,
        inphpol::AbstractDefinitePolarization = PolX(),
        inelspin::AbstractDefiniteSpin = SpinUp(),
        outphpol::AbstractDefinitePolarization = PolX(),
        outelspin::AbstractDefiniteSpin = SpinUp(),
    )
    if !(contains(str, "->"))
        throw("Did not find -> while parsing process \"$str\"")
    end

    (in_str, out_str) = split(str, "->")

    if (isempty(in_str) || isempty(out_str))
        throw("Process (\"$str\") input or output part is empty!")
    end

    incoming_particles = Vector{AbstractParticleType}()
    outgoing_particles = Vector{AbstractParticleType}()

    for (particle_vector, s) in
        ((incoming_particles, in_str), (outgoing_particles, out_str))
        for c in s
            if c == 'e'
                push!(particle_vector, Electron())
            elseif c == 'p'
                push!(particle_vector, Positron())
            elseif c == 'k'
                push!(particle_vector, Photon())
            else
                throw("Encountered unknown characters in the process \"$str\"")
            end
        end
    end

    in_spin_pols = tuple(
        [
            is_boson(incoming_particles[i]) ? inphpol : inelspin for
                i in eachindex(incoming_particles)
        ]...,
    )
    out_spin_pols = tuple(
        [
            is_boson(outgoing_particles[i]) ? outphpol : outelspin for
                i in eachindex(outgoing_particles)
        ]...,
    )

    return ScatteringProcess(
        tuple(incoming_particles...),
        tuple(outgoing_particles...),
        in_spin_pols,
        out_spin_pols,
    )
end
