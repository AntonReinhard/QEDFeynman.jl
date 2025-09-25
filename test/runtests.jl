using SafeTestsets

@safetestset "Task Unit Tests                     " begin
    include("unit_tests_tasks.jl")
end
@safetestset "Node Unit Tests                     " begin
    include("unit_tests_nodes.jl")
end
@safetestset "Properties Unit Tests               " begin
    include("unit_tests_properties.jl")
end
@safetestset "Estimation Unit Tests               " begin
    include("unit_tests_estimator.jl")
end
@safetestset "ABC-Model Unit Tests                " begin
    include("unit_tests_abcmodel.jl")
end
#=@safetestset "QED-Model Unit Tests                " begin
    include("unit_tests_qedmodel.jl")
end=#
#=@safetestset "QED Feynman Diagram Generation Tests" begin
    include("unit_tests_qed_diagrams.jl")
end=#
@safetestset "Node Reduction Unit Tests           " begin
    include("node_reduction.jl")
end
#=@safetestset "Graph Unit Tests                    " begin
    include("unit_tests_graph.jl")
end=#
#=@safetestset "Execution Unit Tests                " begin
    include("unit_tests_execution.jl")
end=#
#=@safetestset "Optimization Unit Tests             " begin
    include("unit_tests_optimization.jl")
end=#
#=@safetestset "Known Graph Tests                   " begin
    include("known_graphs.jl")
end=#
