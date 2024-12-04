using QEDFeynman
using ComputableDAGs
using Random

RNG = Random.MersenneTwister(321)

function test_known_graph(name::String, n)
    @testset "Test $name Graph ($n)" begin
        proc = parse_process(name, ABCModel())
        graph = parse_dag(joinpath(@__DIR__, "..", "input", "$name.txt"), proc)
        props = get_properties(graph)

        test_random_walk(RNG, graph, n)
    end
end

function test_random_walk(RNG, g::DAG, n::Int64)
    @testset "Test Random Walk ($n)" begin
        # the purpose here is to do "random" operations and reverse them again and validate that the graph stays the same and doesn't diverge
        reset_graph!(g)

        @test is_valid(g)

        properties = get_properties(g)

        for i in 1:n
            # choose push or pop
            if rand(RNG, Bool)
                # push
                opt = get_operations(g)

                # choose one of split/reduce
                option = rand(RNG, 1:2)
                if option == 1 && !isempty(opt.node_reductions)
                    push_operation!(g, rand(RNG, collect(opt.node_reductions)))
                elseif option == 2 && !isempty(opt.node_splits)
                    push_operation!(g, rand(RNG, collect(opt.node_splits)))
                else
                    i = i - 1
                end
            else
                # pop
                if (can_pop(g))
                    pop_operation!(g)
                else
                    i = i - 1
                end
            end
        end

        reset_graph!(g)

        @test is_valid(g)

        @test properties == get_properties(g)
    end
end

test_known_graph("AB->AB", 10000)
test_known_graph("AB->ABBB", 10000)
test_known_graph("AB->ABBBBB", 1000)
