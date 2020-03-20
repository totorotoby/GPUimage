using FileIO
using Colors
using StaticArrays
using CUDAdrv
using CuArrays
using CUDAnative
using LinearAlgebra
using Printf
include("array_reduction.jl")
CuArrays.allowscalar(false)

"""
    gaussianstencil(T)

Create a Gaussian blur stencil of floating point type `T`

For more information see
<https://en.wikipedia.org/wiki/Kernel_(image_processing)>
"""
@inline function gaussianstencil(T)
  SMatrix{5, 5, T, 25}(1,  4,  6,   4,  1,
                       4, 16, 24,  16,  4,
                       6, 24, 36,  24,  6,
                       4, 16, 24,  16,  4,
                       1,  4,  6,   4,  1) / 256
end

"""
    sobel(T)

Create a Sobel gradient stencil of floating point type `T`

For more information see
<https://en.wikipedia.org/wiki/Kernel_(image_processing)>
"""
@inline function sobel(T)
  SMatrix{3, 3, T, 9}(1, 0, -1,
                      2, 0, -2,
                      1, 0, -1)
end

"""
    applyblur!(B, A, M, N, S, W, b, block_dim)

Apply the Gaussian blur stencil to the matrix `A` and store the result in the
matrix `B`

M: # of rows in image
N: # of columns in image
S: the stencil matrix
W: the half length of the stencil
b: float to store the new entry of b[i,j]
block_dim: the dimensions of each block of threads
"""
function applyblur!(d_B, d_A, S, ::Val{block_dim}) where block_dim

    # which thread are we locally
    tidx = threadIdx().x
    tidy = threadIdx().y

     # which block are we in
    bidx = blockIdx().x
    bidy = blockIdx().y

    # global indices
    i = tidx + block_dim * (bidx - 1)
    j = tidy + block_dim * (bidy - 1)

    
    
    N, M = size(d_B)
    W = div(size(S,1),2)

    if i <= N && j <= M
        # Loop over the stencil indices (k + W + 1, l + W + 1)
        for l = -W:W
            # (I, J) are the input image pixels
            # (max and min prevent going outside the image)
            J = max(1, min(M, j + l))
            for k = -W:W
                I = max(1, min(N, i + k)) 
                # apply the stencil to the pixel
                @inbounds d_B[i, j] += S[k + W + 1, l + W + 1]* d_A[I, J]
  	    end
        end
    end

    nothing
    
end



"""
    gradientmagnitude!(B, A)

Apply the gradient stencils to the matrix `A` and store the result in the
magnitude of the two calculations in the matrix `B`.
"""
function gradientmagnitude!(B, A, T, Gx, Gy, ::Val{block_dim}) where block_dim

    N, M = size(A)
    W = div(size(Gx, 1),2)

   
  
    # which thread are we locally
    tidx = threadIdx().x
    tidy = threadIdx().y

    # which block are we in
    bidx = blockIdx().x
    bidy = blockIdx().y

    # global indices
    i = tidx + block_dim * (bidx - 1)
    j = tidy + block_dim * (bidy - 1)

    
    if i <= N && j <= M
        
        # storage for application of the stencil in two direction
        bx = zero(eltype(A))
        by = zero(eltype(A))
        # Loop over the stencil indices (k + W + 1, l + W + 1)
        for l = -W:W
            # (I, J) are the input image pixels
            # (max and min prevent going outside the image)
            J = max(1, min(M, j + l))
            for k = -W:W
                I = max(1, min(N, i + k))
                
                # apply the stencil to the pixel
                @inbounds bx += Gx[W+1+k, W+1+l] * A[I, J]
                # apply the transpose stencil to the pixel
                @inbounds by += Gy[W+1+l, W+1+k] * A[I, J]
            end
        end
        
    
        # Save the magnitude of the gradient vector
        # Will need/want to use CUDAnative.hypot(bx, by) on GPU
        @inbounds B[i, j] = CUDAnative.hypot(bx, by)
    end
    nothing
end

"""
    normalize!(A, minv, maxv)

Normalize each value of `A` so that it is between `0` and `1` using the formula
`A[i,j] = (A[i,j] - minv) / (maxv - minv)`
"""
function normalize!(A, minv, maxv, ::Val{block_dim}) where block_dim
  # Get the matrix size
    N, M = size(A)
  
    # which thread are we locally
    tidx = threadIdx().x
    tidy = threadIdx().y

    # which block are we in
    bidx = blockIdx().x
    bidy = blockIdx().y
    
    # global indices
    i = tidx + block_dim * (bidx - 1)
    j = tidy + block_dim * (bidy - 1)
    
    
    if i <= N && j <= M
        @inbounds A[i, j] = (A[i, j] - minv) / (maxv - minv)
    end
    
    nothing
end

"""
    findedges(inputfile, outputfile, [T = Float64])

Find the edges in the image `inputfile` by first using a blur stencil then
computing the (normalized) gradient magnitude. The final result is saved in the
file `outputfile`. The optional parameter `T` can be used to specify the
floating point type to be used for the calculation.

# Example
```julia-repl
julia> findedges("10945_46620.jpg", "edges.png")

julia> findedges("10945_46620.jpg", "edges.png", Float32)
```
"""
function findedges(inputfile, outputfile, T=Float64)
    # load the image
    img = load(inputfile)
    
    # convert the image to a grayscale image and then a matrix
    h_A = convert(Array{T}, Gray.(img))
    
    # Convert to device array for compution
    d_A = CuArray(h_A)
    
    d_B = similar(d_A)
    
    # Call the findedges! function on the data matrix d_A
    findedges!(d_A, d_B)

    # Copy back to host for saving output
    copy!(h_A, d_A)

    #=
    minA = minimum(h_A)
    maxA = maximum(h_A)    
    len = maxA - minA
    h_A .= (h_A .- minA) ./ len
    =#
    # save the image
    save(outputfile, Gray.(h_A))
end

"""
    findedges!(A, [B = similar(A)])

Apply the find edges algorithm to the data matrix `A`. The optional argument `B`
is for preallocated scratch storage and should match `A`. (In performance
testing runs you should preallocate `B` so that allocation is not part of the
timing.)

# Example
```julia-repl
julia> using Random

julia> A = CuArray{Float64}(undef, 100, 200)

julia> rand!(A)

julia> findedges!(A)

julia> B = similar(A)

julia> findedges!(A, B)
```
"""
function findedges!(d_A, d_B=similar(d_A))

    
    T = eltype(d_A)
    
    S = gaussianstencil(T)
    S = CuArray(S)
    N, M = size(d_A)
    block_dim = 32
    nblocks = (cld(N, block_dim), cld(M, block_dim))
    
    @cuda threads=(block_dim, block_dim) blocks=nblocks applyblur!(d_B,
                                                                   d_A,
                                                                   S,
                                                                   Val(block_dim))
    
    Gx = sobel(T)
    Gy = sobel(T)
    Gx = CuArray(Gx)
    Gy = CuArray(Gy')
    @cuda threads=(block_dim, block_dim) blocks=nblocks gradientmagnitude!(d_A,
                                                                           d_B,
                                                                           T,
                                                                           Gx,
                                                                           Gy,
                                                                           Val(block_dim))
    
    
    # Compute the extrema of the data
    minA, maxA = extremevalues(d_A)
    checMax = maximum(d_A)
    @show maxA, checMax
    
    # normalize the image by the maximum value
    @cuda threads=(block_dim, block_dim) blocks=nblocks normalize!(d_A,
                                                                    minA,
                                                                   maxA,
                                                                   Val(block_dim))
    
    return d_A
end
