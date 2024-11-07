# This file includes all tests for the DGSEM method
include("DGSEM.jl")
using Plots
using DataFrames

function plot_numeric_analytic(x, u, test, error, title)

    u = vec(u)
    test = vec(test)
    error = vec(error)

    labels = ["numeric" "analytic" "init"]
    figure = plot(x,[u test error], label = labels, title = title, xlabel = "x", ylabel = "solution")
    display(figure)
    #readline() shows the plot in Codium. End Code by pressing enter.
    readline()

    savefig(figure,title*".pdf")
end

function test_evaluation(u, x, max_time, a, test_result_function)

    println("u")
    print_line(u)
    
    test = test_result_function(x, max_time, a)
    error = u - test
    println("error: u - analytical result")
    print_line(error)
    return [test, error]
end

# Test function for sine initialization
function analytical_result(x, max_time, a)
    return sin.(2*pi .*(x .- a* max_time))
end

# Init function for sine initialization
function init_advection(x)
    return sin.(2*pi .*x)
end

# Init function for sine initialization
function init_discontinuous(x)
    return abs.(x)
end

# Init function for sine initialization
function init_discontinuous_sin_cos(x)
    if(x <= 0)
        return sin(x)
    else
        return cos(x)
    end
end

# Init constant and result constant corresponds to the test of the DGSEM with constant result and advection velocity 0
function init_constant(x)
    return 4 .* ones(size(x))
end

function result_constant(x, time, a)
    return 4 .* ones(size(x))
end

# Test constant result with advection velocitiy 0, expectation: nothing happens
function test_DGSEM_const()
    println("Test DGSEM with const result.")
    deg_p = 3;
    mesh_size = 10;
    max_time = 2.0;

    advection_velocity = 1;

    result = DGSEM(deg_p, mesh_size, init_constant, max_time, advection_velocity)
    u = result[1]
    x = result[2]

    test = test_evaluation(u, x, max_time,advection_velocity, result_constant)
    plot_numeric_analytic(vec(x),vec(u), vec(test[1]), vec(test[2]), "DGSEM with const init")
end

function test_DGSEM_sin()
    println("Test DGSEM on one axis.")
    deg_p = 4;
    mesh_size = 20;
    max_time = 1.75;

    advection_velocity = 1.0;

    result = DGSEM(deg_p, mesh_size, init_advection, max_time, advection_velocity)
    u = result[1]
    x = result[2]

    test = test_evaluation(u, x, max_time, advection_velocity, analytical_result)
    # Plot with init function instead the error
    plot_numeric_analytic(vec(x),vec(u), vec(test[1]), init_advection(x), "DGSEM with sin init")
end

function test_DGSEM_discontinuous()
    println("DGSEM discontinuous.")
    deg_p = 4;
    mesh_size = 40;
    max_time = 1.75;

    advection_velocity = 1.5;

    result = DGSEM(deg_p, mesh_size, init_discontinuous, max_time, advection_velocity)
    u = vec(result[1])
    x = vec(result[2])
    println("test")
    init = vec(init_discontinuous.(x))

    title = "discontinuous example"
    labels = ["solution t=1.75" "init t=0"]
    figure = plot(x,[u init], label = labels, title = title, xlabel = "x", ylabel = "solution")
    display(figure)
    #readline() shows the plot in Codium. End Code by pressing enter.
    readline()

    savefig(figure,title*".svg")    
end

function test_DGSEM_discontinuous_sine_cosinus()
    println("DGSEM discontinuous.")
    deg_p = 4;
    mesh_size = 100;
    max_time = 0.75;

    advection_velocity = 1.5;

    result = DGSEM(deg_p, mesh_size, init_discontinuous_sin_cos, max_time, advection_velocity)
    u = vec(result[1])
    x = vec(result[2])
    init = vec(init_discontinuous_sin_cos.(x))

    title = "discontinuous example"
    labels = ["solution t=0.75" "init t=0"]
    figure = plot(x,[u init], label = labels, title = title, xlabel = "x", ylabel = "solution")
    display(figure)
    #readline() shows the plot in Codium. End Code by pressing enter.
    readline()

    savefig(figure,title*".svg")    
end


function main()
    println("Program start :)")

    # Test constant result with advection velocitiy 0, expectation: nothing happens
    #test_DGSEM_const()
    # Test advection along one axis with advection 1 and max_time 1.0 and initialization with sine, expectation: same result, than init with sine
    #test_DGSEM_sin()
    test_DGSEM_discontinuous_sine_cosinus()

    println("Program finished :)")
end
main()

