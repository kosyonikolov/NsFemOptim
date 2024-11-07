function external()
    x = 3
    y = 2 * x
    println("y = $y")
end

function aaa()
    # println("External function call")
    # x = 5
    # println("x = $x")
    # external()
    # println("x = $x")
    # # println("y = $y") # Undefined

    println("Internal function call")
    function internal()
        x = 3
        y = 2 * x
        println("y = $y")
    end
    x = 5 # !!! X gets captured even though it's declared after the internal function

    println("x = $x")
    internal()
    println("x = $x")
    # println("y = $y") # Undefined
end

function external2()
    x = [1,3]
    y = 2 * x
    println("y = $y")
end

function aaa2()
    # println("External function call")
    # x = [2,4]
    # println("x = $x")
    # external()
    # println("x = $x")
    # # println("y = $y") # Undefined

    println("Internal function call")
    function internal()
        x = [1,3]
        y = 2 * x
        println("y = $y")
    end
    x = [2,4] # !!! X gets captured even though it's declared after the internal function

    println("x = $x")
    internal()
    println("x = $x")
    println("y = $y") # Undefined
end