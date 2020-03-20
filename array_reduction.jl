#=
Find the maximum and minimum value of an array A.
=#

using Printf
using CuArrays
using CUDAnative
using CUDAdrv
using Plots
CuArrays.allowscalar(false)

# gets the smallest divisor (used for  blocks per column)
function smallest_divisor(N)
    for i = 2:N
       if N%i == 0 && div(N,i)%2 == 0 && div(div(N,i),2) <= 1024
           return i
       end
    end
    return 1
end

# stores the minimum value of a block in block_min[block id]
function reduce_array_min!(d_A, block_min, ::Val{nthreads}, ::Val{bPc}) where {nthreads, bPc}

    T = eltype(d_A)
    
    tidx = threadIdx().x
    bidx = blockIdx().x

    i_1 = (bidx - 1) % bPc * (nthreads*2) + tidx
    i_2 = i_1 + nthreads
    j_1 = div(bidx - 1, bPc) + 1
    j_2 = j_1

    shared = @cuStaticSharedMem(T, nthreads)
    @inbounds shared[tidx] = min(d_A[i_1, j_1], d_A[i_2, j_2])

    
    s = cld(nthreads,2)
    active = nthreads
    sync_threads()
    while s > 1
        
        if tidx + s <= active
            @inbounds shared[tidx] = min(shared[tidx], shared[tidx + s])
        else
            @inbounds shared[tidx] = min(shared[tidx], typemax(T))
        end
        
        sync_threads()
        s = cld(s,2)
        active = cld(active,2)        
    end

    @inbounds block_min[bidx] = min(shared[1], shared[2])
    nothing
end

# stores the minimum value of a block in block_min[block id]
function reduce_array_min2!(d_A, block_min, ::Val{nthreads}, ::Val{bPc}) where {nthreads, bPc}

    T = eltype(d_A)
    
    tidx = threadIdx().x
    bidx = blockIdx().x

    i_1 = (bidx - 1) % bPc * (nthreads*2) + (2 * (tidx - 1)) + 1
    i_2 = i_1 + 1
    j_1 = div(bidx - 1, bPc) + 1
    j_2 = j_1


    #@cuprintln("I am thread $tidx in $bidx, looking at ($i_1, $j_1) and  ($i_2, $j_2)")
    
    shared = @cuStaticSharedMem(T, nthreads)
    @inbounds shared[tidx] = min(d_A[i_1, j_1], d_A[i_2, j_2])

    
    s = 1
    active_threads = cld(nthreads,2)
    active_indices = nthreads
    sync_threads()
    loc_i = (2 * (tidx - 1)) + 1
    while active_threads > 1
        
        if loc_i + s <= active_indices
            @inbounds shared[tidx] = min(shared[loc_i], shared[loc_i + s])
        elseif loc_i <= active_indices && loc_i + s > active_indices
            @inbounds shared[tidx] = min(shared[loc_i], typemax(T))
        else
            @inbounds shared[tidx] = typemax(T)
        end
        
        sync_threads()
        active_indices = cld(active_indices ,2)
        active_threads = cld(active_threads, 2)        
    end

    @inbounds block_min[bidx] = min(shared[1], shared[2])
    nothing
end

# stores the minimum value of a block in block_min[block id]
function reduce_array_max!(d_A, block_min, ::Val{nthreads}, ::Val{bPc}) where {nthreads, bPc}

    T = eltype(d_A)
    
    tidx = threadIdx().x
    bidx = blockIdx().x

    i_1 = (bidx - 1) % bPc * (nthreads*2) + tidx
    i_2 = i_1 + nthreads
    j_1 = div(bidx - 1, bPc) + 1
    j_2 = j_1

    shared = @cuStaticSharedMem(T, nthreads)
    @inbounds shared[tidx] = max(d_A[i_1, j_1], d_A[i_2, j_2])

    
    s = cld(nthreads,2)
    active = nthreads
    sync_threads()
    while s > 1
        
        if tidx + s <= active
            @inbounds shared[tidx] = max(shared[tidx], shared[tidx + s])
        else
            @inbounds shared[tidx] = max(shared[tidx], typemin(T))
        end
        
        sync_threads()
        s = cld(s,2)
        active = cld(active,2)        
    end

    @inbounds block_min[bidx] = max(shared[1], shared[2])
    nothing
end

# stores the minimum value of a block in block_min[block id]
function reduce_array_max2!(d_A, block_min, ::Val{nthreads}, ::Val{bPc}) where {nthreads, bPc}

    T = eltype(d_A)
    
    tidx = threadIdx().x
    bidx = blockIdx().x

    i_1 = (bidx - 1) % bPc * (nthreads*2) + (2 * (tidx - 1)) + 1
    i_2 = i_1 + 1
    j_1 = div(bidx - 1, bPc) + 1
    j_2 = j_1


    #@cuprintln("I am thread $tidx in $bidx, looking at ($i_1, $j_1) and  ($i_2, $j_2)")
    
    shared = @cuStaticSharedMem(T, nthreads)
    @inbounds shared[tidx] = max(d_A[i_1, j_1], d_A[i_2, j_2])

    
    s = 1
    active_threads = cld(nthreads,2)
    active_indices = nthreads
    sync_threads()
    loc_i = (2 * (tidx - 1)) + 1
    while active_threads > 1
        
        if loc_i + s <= active_indices
            @inbounds shared[tidx] = max(shared[loc_i], shared[loc_i + s])
        elseif loc_i <= active_indices && loc_i + s > active_indices
            @inbounds shared[tidx] = max(shared[loc_i], typemin(T))
        else
            @inbounds shared[tidx] = typemin(T)
        end
        
        sync_threads()
        active_indices = cld(active_indices ,2)
        active_threads = cld(active_threads, 2)        
    end

    @inbounds block_min[bidx] = max(shared[1], shared[2])
    nothing
end



function block_reduce_vector_min!(d_V, block_min, call, ::Val{nthreads}, ::Val{nblocks}) where {nthreads, nblocks}
    N = length(d_V)
    T = eltype(d_V)
    tidx = threadIdx().x
    bidx = blockIdx().x

    i =  2 * nthreads * (bidx - 1) + tidx
    next = i + nthreads

    shared = @cuStaticSharedMem(T, nthreads)
    if next <= N && i <= N
        @inbounds shared[tidx] = min(d_V[i], d_V[next])
    elseif i <= N && next >= N
        @inbounds shared[tidx] = min(d_V[i], typemax(T))
    else
        @inbounds shared[tidx] = typemax(T)                   
    end
    
    s = cld(nthreads,2)
    active = nthreads
    sync_threads()
    
    while s > 1
        if tidx + s <= active
            @inbounds shared[tidx] = min(shared[tidx], shared[tidx + s])
        else
            @inbounds shared[tidx] = min(shared[tidx], typemax(T))
        end
        sync_threads()
        s = cld(s,2)
        active = cld(active,2)        
    end

    @inbounds block_min[bidx] = min(shared[1], shared[2])
    
    nothing
    
end
#=
function block_reduce_vector_min2!(d_V, block_min, call, ::Val{nthreads}, ::Val{nblocks}) where {nthreads, nblocks}
    
    N = length(d_V)
    T = eltype(d_V)
    tidx = threadIdx().x
    bidx = blockIdx().x

    
    i =  2 * nthreads * (bidx - 1) + (2 * (tidx - 1)) + 1
    next = i + 1

    shared = @cuStaticSharedMem(T, nthreads)
    if next <= N && i <= N
        @inbounds shared[tidx] = min(d_V[i], d_V[next])
    elseif i <= N && next > N
        @inbounds shared[tidx] = min(d_V[i], typemax(T))
    else
        @inbounds shared[tidx] = typemax(T)                   
    end

    s = 1
    active_threads = cld(nthreads,2)
    active_indices = nthreads
    sync_threads()
    loc_i = (2 * (tidx - 1)) + 1
    while active_threads > 1
        
        if loc_i + s <= active_indices
            @inbounds shared[tidx] = min(shared[loc_i], shared[loc_i + s])
        elseif loc_i <= active_indices && loc_i + s > active_indices
            @inbounds shared[tidx] = min(shared[loc_i], typemax(T))
        else
            @inbounds shared[tidx] = typemax(T)
        end
        
        sync_threads()
        active_indices = cld(active_indices ,2)
        active_threads = cld(active_threads, 2)        
    end

    @inbounds block_min[bidx] = min(shared[1], shared[2])
    nothing
  
end
=#

function block_reduce_vector_max!(d_V, block_min, call, ::Val{nthreads}, ::Val{nblocks}) where {nthreads, nblocks}
    N = length(d_V)
    T = eltype(d_V)
    tidx = threadIdx().x
    bidx = blockIdx().x

    i =  2 * nthreads * (bidx - 1) + tidx
    next = i + nthreads

    shared = @cuStaticSharedMem(T, nthreads)
    if next <= N && i <= N
        @inbounds shared[tidx] = max(d_V[i], d_V[next])
    elseif i <= N && next >= N
        @inbounds shared[tidx] = max(d_V[i], typemin(T))
    else
        @inbounds shared[tidx] = typemin(T)                   
    end
    
    s = cld(nthreads,2)
    active = nthreads
    sync_threads()
    while s > 1
        if tidx + s <= active
            @inbounds shared[tidx] = max(shared[tidx], shared[tidx + s])
        else
            @inbounds shared[tidx] = max(shared[tidx], typemin(T))
        end
        sync_threads()
        s = cld(s,2)
        active = cld(active,2)        
    end

    @inbounds block_min[bidx] = max(shared[1], shared[2])
    
    nothing
    
end

function reduce_vector_min(d_V, nthreads)

    N = length(d_V)
    T = eltype(d_V)
    

    nblocks = cld(N, nthreads * 2)
    
    block_min =  CuArray{T, 1}(undef, nblocks)
    call = 1
    @cuda threads=nthreads blocks=nblocks block_reduce_vector_min!(d_V,
                                                                   block_min,
                                                                   call,
                                                                   Val(nthreads),
                                                                   Val(nblocks))

    nblocks = cld(nblocks, 2 * nthreads)
    while nblocks >= 2
        call += 1
        # the new length of block min is the number of blocks each time
        
        new_block_min = CuArray{T, 1}(undef, nblocks)
        @cuda threads=nthreads blocks=nblocks block_reduce_vector_min!(block_min,
                                                                       new_block_min,
                                                                       call,
                                                                       Val(nthreads),
                                                                       Val(nblocks))
        block_min = new_block_min
        nblocks = cld(nblocks, nthreads)
        
    end
    
    return minimum(block_min)
    
end

function reduce_vector_max(d_V, nthreads)

    N = length(d_V)
    T = eltype(d_V)
    

    nblocks = cld(N, nthreads * 2)
    
    block_min =  CuArray{T, 1}(undef, nblocks)
    call = 1
    @cuda threads=nthreads blocks=nblocks block_reduce_vector_max!(d_V,
                                                                   block_min,
                                                                   call,
                                                                   Val(nthreads),
                                                                   Val(nblocks))

    nblocks = cld(nblocks, 2 * nthreads)
    while nblocks >= 2
        call += 1
        # the new length of block min is the number of blocks each time
        
        new_block_min = CuArray{T, 1}(undef, nblocks)
        @cuda threads=nthreads blocks=nblocks block_reduce_vector_max!(block_min,
                                                                       new_block_min,
                                                                       call,
                                                                       Val(nthreads),
                                                                       Val(nblocks))
        block_min = new_block_min
        nblocks = cld(nblocks, nthreads)
        
    end
    return maximum(block_min)  
end



function reduce_array_min(d_A, threads2)

    N, M = size(d_A)
    bPc = smallest_divisor(N)
    nthreads = div(N, bPc*2)
    ePb = nthreads * 2
    nblocks = M * bPc

    @show N, M
    @show bPc
    @show nthreads
    @show ePb
    @show nblocks
    
    block_min = CuArray{Float64, 1}(undef, nblocks)
    @cuda threads=nthreads blocks=nblocks reduce_array_min!(d_A,
                                                            block_min,
                                                            Val(nthreads),
                                                             Val(bPc))
 

    my_minimum = reduce_vector_min(block_min, threads2)
    
    return my_minimum
end

function reduce_array_min2(d_A, threads2)

    N, M = size(d_A)
    bPc = smallest_divisor(N)
    nthreads = div(N, bPc*2)
    ePb = nthreads * 2
    nblocks = M * bPc

    @show N, M
    @show bPc
    @show nthreads
    @show ePb
    @show nblocks
    
    block_min = CuArray{Float64, 1}(undef, nblocks)
    @cuda threads=nthreads blocks=nblocks reduce_array_min2!(d_A,
                                                            block_min,
                                                            Val(nthreads),
                                                             Val(bPc))
 
    
    my_minimum = reduce_vector_min(block_min, threads2)
    
    return my_minimum
end


function reduce_array_max(d_A)

    N, M = size(d_A)
    bPc = smallest_divisor(N)
    nthreads = div(N, bPc*2)
    ePb = nthreads * 2
    nblocks = M * bPc

 
    block_min = CuArray{Float64, 1}(undef, nblocks)
    @cuda threads=nthreads blocks=nblocks reduce_array_max!(d_A,
                                                            block_min,
                                                            Val(nthreads),
                                                            Val(bPc))

    
    my_max = reduce_vector_max(block_min, nthreads)

    return my_max
end

function extremevalues(d_A)
    
    my_min = reduce_array_min(d_A)
    my_max = reduce_array_max(d_A)
    
    return(my_min, my_max)
end


function main(; N=1024, M=1024, repetitions=10, T=Int32)
    
    # amount of memory in gigabytes
    memory_size = (N*M*sizeof(T)) / 1024^3

    @printf("Dimensions of Array: (%d, %d)\n", N, M)
    @printf("Storing: %s\n", T)
    @printf("Memory: %e GiB\n\n", memory_size)

    # host Array
    h_A = rand(T, N, M)
    #display(h_A)
    @printf("\n\n")
    # minimum and maximum of host Array
    h_min_A = minimum(h_A)
    h_max_A = maximum(h_A)

    t1 = time_ns()
    for i in 1:repetitions
        minimum(h_A)
    end
    t2 = time_ns()

    htime_n = (t2 - t1)/repetitions*10^-9
    @printf("Time of cpu library function: %f s\n", htime_n)
    # GPU Array
    d_A = CuArray(h_A)
    # minimum and maximum of GPU Array
    d_min_A = minimum(d_A)
    d_max_A = maximum(d_A)
    
    
    t1 = time_ns()
    for i in 1:repetitions
        minimum(d_A)
    end
    t2 = time_ns()
    dtime_n = (t2 - t1)/repetitions*10^-9


    # error between library CPU and GPU min and max
    l_error_min = h_min_A - d_min_A
    l_error_max = h_max_A - d_max_A
    @printf("Time of gpu library function: %f s\n", dtime_n)
    @printf("Library function minimum error: %e\n", l_error_min)
    #=
    # every block will have an even number of elements, and each block will be
    # as long as possible while still dividing the column
    bPc = smallest_divisor(N)
    nthreads = div(N, bPc*2)
    nblocks = M * bPc
    =#
    #@printf("Blocks per Column: %d\nthreads per block: %d\nNumber of blocks: %d\n\n\n"
     #       , bPc
     #       , nthreads
     #       , nblocks)






    
    #=
    t1 = time_ns()
    block_min = CuArray{Float64, 1}(undef, nblocks)
    @cuda threads=nthreads blocks=nblocks reduce_array_min!(d_A,
                                                            block_min,
                                                            Val(nthreads),
                                                            Val(bPc))



    
    nthreads = 32
    my_minimum = reduce_vector_min(block_min, nthreads)
    synchronize()
    t2 = time_ns()
    =#
    my_minimum = reduce_array_min(d_A)
    t1 = time_ns()
    my_minimum = reduce_array_min(d_A)
    t2 = time_ns()
    avg_time = (t2-t1)*10^-9
    my_error_min = h_max_A - my_minimum
    @printf("My function error: %e\n", my_error_min)
    @printf("My function time: %f\n", avg_time)
    nothing
end    


function stride_comp()
    T = Float64
    N = 1024:1024:(1024*17)
    mem = []
    h_time = []
    d_time = []
    s1_time = []
    s2_time = []
    repetitions = 10
    for n in N
        h_A = rand(n,n)
        d_A = CuArray(h_A)
        t1 = time_ns()
        for i in 1:repetitions
            minimum(h_A)
        end
        t2 = time_ns()
        htime_n = (t2 - t1)/repetitions*10^-9
        
        t1 = time_ns()
        for i in 1:repetitions
            minimum(d_A)
        end
        t2 = time_ns()
        dtime_n = (t2 - t1)/repetitions*10^-9

        t1 = time_ns()
        for i in 1:repetitions
            reduce_array_min(d_A, 32)
        end
        t2 = time_ns()
        my_s1_time_n = (t2 - t1)/repetitions*10^-9

        t1 = time_ns()
        for i in 1:repetitions
            reduce_array_min2(d_A, 32)
        end
        t2 = time_ns()
        my_s2_time_n = (t2 - t1)/repetitions*10^-9

        
        
        append!(mem, (n*n*sizeof(T)) / 1024^3)
        append!(h_time, htime_n)
        append!(d_time, dtime_n)
        append!(s1_time, my_s1_time_n)
        append!(s2_time, my_s2_time_n)
    end
    plt = plot(title = "Minimum Array Reduction Times with 32 threads per block", xlabel="GiB data", ylabel="seconds")
    
    plot!(mem, h_time, label="CPU library function")
    plot!(mem, d_time, label="GPU library function")
    plot!(mem, s1_time, label="My function big stride")
    plot!(mem, s2_time, label="My function neighbor stride")
    png("stide_comp")
        
end


function thread_comp()

    T = Float64
    h_A = rand(1024*8, 1024*8)
    d_A = CuArray(h_A)
    repetitions = 10
    t_num = collect(2:1:32)
    s1 = []
    s2 = []
    for t in t_num

        t1 = time_ns()
        for i in 1:repetitions
            reduce_array_min(d_A, t)
        end
        t2 = time_ns()
        my_s1_time_n = (t2 - t1)/repetitions*10^-9
        append!(s1, my_s1_time_n)
        t1 = time_ns()
        for i in 1:repetitions
            reduce_array_min2(d_A, t)
        end
        t2 = time_ns()
        my_s2_time_n = (t2 - t1)/repetitions*10^-9
        append!(s2, my_s2_time_n)
    end
    plt = plot(title = "Minimum Array Reduction Times", xlabel="Number of threads per Block", ylabel="seconds")
    
    plot!(t_num, s1, label="My big stride function")
    plot!(t_num, s2, label="My neighor stride function")

    png("thread_comp")
end

    
