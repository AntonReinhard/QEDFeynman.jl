"""
    QEDFeynman

A module containing QED and ABC model functionality for scattering processes in particle physics.
"""
module QEDFeynman

using QEDbase
using QEDcore
using QEDprocesses
using ComputableDAGs
using TypeUtils
using Base.Threads

# ABC model
export ParticleValue
export ParticleA, ParticleB, ParticleC
export ABCParticle, GenericABCProcess, ABCModel, PerturbativeABC
export ComputeTaskABC_S1
export ComputeTaskABC_S2
export ComputeTaskABC_V
export ComputeTaskABC_U
export ComputeTaskABC_Sum
export parse_process
export parse_dag

# QED model
export FeynmanDiagram, FeynmanVertex, FeynmanTie, FeynmanParticle
export QEDModel
export ComputeTaskQED_S1
export ComputeTaskQED_S2
export ComputeTaskQED_V
export ComputeTaskQED_U
export ComputeTaskQED_Sum
export graph

export ParticleValue, ParticleValueSP

# input generation
export gen_process_input

include("utility.jl")
include("interface.jl")
include("impl.jl")

include("abc/types.jl")
include("abc/generic_abc_process.jl")
include("abc/particle.jl")
include("abc/compute.jl")
include("abc/create.jl")
include("abc/properties.jl")
include("abc/parse.jl")
include("abc/print.jl")

include("qed/utility.jl")
include("qed/types.jl")
include("qed/particle.jl")
include("qed/diagrams.jl")
include("qed/compute.jl")
include("qed/create.jl")
include("qed/properties.jl")
include("qed/parse.jl")
include("qed/print.jl")

end # module QEDFeynman
