# some functions for better work

function print_line(x)
    for i in 1:size(x)[1]
        println(x[i,:])
    end
    println()
end

