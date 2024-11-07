# # based on the paper: A horizontally explicit, vertically implicit (HEVI) discontinuous Galerkin scheme 
# for the 2-dimensional Euler and Navier-Stokes equations using terrain-following coordinates (M. Baldauf)
# The terrain-following parameters are stored with the abbreviation tfc and the Cartesian ones are stored with c

using LinearAlgebra
using ForwardDiff
using UnPack
using StaticArrays

function print_line(x)
    x = round.(x; digits = 4)
    for i in 1:size(x)[1]
        println(x[i,:])
    end
    println()
end

# Struct for the paramteters that characterises the meshes in Cartesian and terrain-following coordinates in 2D
struct Meshes_struct_2D{A1, A3, h_top, Cells_tfc, Cells_c, Conn, Num_elem,DIM}
    a1::A1
    a3::A3
    H_top::h_top
    cells_tfc::Cells_tfc
    cells_c::Cells_c 
    conn::Conn
    num_elem::Num_elem
    dim::DIM
end

# Short call for the function derivate of the package ForwardDiff
@inline function derivate(func, variable)
    return ForwardDiff.derivative(func, variable)
end

# Defines the grid in vertical direction, xi in [0,1]
# Stretching function based on the paper of M.Baldauf
# It focused on the lower areas of atmosphere
@inline function stretching_paper(xi)
    b = 19/21
    return b .* xi .^2 .+ (1-b).*xi
end

# Defines the grid in the vertical direction, xi in [0,1]
# Stretching function for equidistant data points in vertical direction : f(xi) = xi
@inline function stretching_equidistant(xi)
    return xi
end

# The function generates a mesh with N_hc x N_vc unit cells (Quads) on [0,2N_hc]x[0,2N_vc]. 
# The DG method is applied to this this mesh. It is stored in the same format than (cells, conn) of the unit_mesh of StartUpDG.jl 
function tfc_mesh(N_hc, N_vc)

    num_elem = N_vc * N_hc;

    # Create connectivity matrix
    conn = zeros(num_elem,4);
    ielem = 0
    for v in 1:N_vc
        for h in 1:N_hc
            edge = v + (h-1)*(N_vc+1);
            conn[ielem + h, 1] = edge;
            conn[ielem + h, 2] = edge + N_vc + 1;
            conn[ielem + h, 3] = edge + 1;
            conn[ielem + h, 4] = edge + N_vc + 2;
        end
        ielem += N_hc
    end
    conn = convert(Array{Int64}, conn)

    # Create cells [0,2N_hc]x[0,2N_vc]
    num_nodes = (N_hc+1)*(N_vc+1)
    x = zeros(num_nodes);
    y = zeros(num_nodes);
    step = 2;
    inode = 1;
    for ix in 1:N_hc+1
        for iy in 1:N_vc+1
            x[inode] = step * (ix-1)
            y[inode] = step * (iy-1)
            inode += 1
        end
    end
    cells = (x,y)
    return (cells, conn)
end

# Distributes the x3 values between the model bottom, which is described by orography(x1_c = a1*x1_tfc) and the model top H_top.
# The distribution is described by the stretching function
@inline function cart_vertical(orography, x1_tfc, x3_tfc, a, H_top; stretching = stretching_equidistant)
    return orography.(a[1] .* x1_tfc) .+ (H_top .- orography.(a[1] .* x1_tfc)) .*stretching.(a[2] .*x3_tfc ./H_top)
end

# The start mesh is a cartesian mesh. It based on equidistant x1 values. The bottom of the mesh is described by orography(x1).
# From each orography(x1) value the the distribution of the x3 values to the model top H_top are defined by the stretching function. 
# The cartesian mesh mesh is stored in the same format than the tfc-mesh.
@inline function cartesian_mesh(orography, cells_tfc, a1, a3, H_top; stretching = stretching_equidistant)
    x1_tfc = cells_tfc[1];
    x3_tfc = cells_tfc[2];
    
    x1_c = a1 .*x1_tfc
    x3_c = cart_vertical(orography, x1_tfc, x3_tfc, [a1, a3], H_top, stretching = stretching)

    return (x1_c, x3_c)
end

""" Meshes calculates the x and y values of the nodes of the Cartesian and the tfc mesh.
The struct stores the cells of the meshes and their connectivity with important scaling parameters.
The connectivity does not change by transforming the Cartesian mesh to the tfc mesh. 
Contains: a1, a3, H_top, cells_tfc, cells_c, conn, num_elem """
function Meshes_2D(orography, Lx, H_top, N_hc, N_vc, dim; stretching = stretching_equidistant)
    
    cells_tfc, conn = tfc_mesh(N_hc, N_vc)
    
    a1 = Lx/(2*N_hc); 
    a3 = H_top/(2*N_vc);

    cells_c = cartesian_mesh(orography, cells_tfc, a1, a3, H_top, stretching = stretching)

    num_elem = N_hc*N_vc;
    variables = Meshes_struct_2D(a1, a3, H_top, cells_tfc, cells_c, conn, num_elem, dim)
end

# Calculation of alpha_3_1 = dx3_cart/dx1_tfc
@inline function alpha_3_1(orography, x1_c, xi, a1; stretching = stretching_equidistant)
    return derivate(orography,x1_c)*a1*(1-stretching(xi))
end

# Calculation of alpha_3_3 = dx3_cart/dx3_tfc
@inline function alpha_3_3(orography, x1_c, xi, a3, H_top; stretching = stretching_equidistant)
    return (H_top - orography(x1_c))*derivate(stretching,xi)*(a3/H_top)
end

# Function that calculates the covariant metric tensor of a 2d transformation 
# Based on the paper of M. Baldauf
# As the covariant metric tensor is symmetric (g[2,1] = g[1,2]) we only need this information once. 
function covariant_metric_tensor_2d(a1, alpha)
    rxG = a1^2 .+ (alpha[3]) .^2    #g11
    sxG = (alpha[3]) .* (alpha[4])  #g12
    ryG = sxG                       #g21
    syG = (alpha[4]) .^2            #g22
    return SMatrix{2,2}(rxG, sxG, ryG, syG)
end

# Function for calculating the metric scalar density of a covariant metric tensor
@inline function metric_scalar_density_2d(g11, g12, g21, g22)
    return det([g11 g12;g21 g22])
end

@inline function inverse_metric(g11, g12, g21, g22)
    tmp = inv([g11 g12;g21 g22])
    return (tmp[1,1], tmp[1,2], tmp[2,1], tmp[2,2])
end

function contravariant_metric_tensor_2d(g_cov)
    g11, g12, g21, g22 = g_cov[1], g_cov[2], g_cov[3], g_cov[4]
    g_c1 = zero(g11) 
    g_c2 = zero(g11) 
    g_c3 = zero(g11) 
    g_c4 = zero(g11)  

    num_nodes, num_elem = size(g11)
    
    for inode in 1:num_nodes
        for ielem in 1:num_elem
            tmp = inverse_metric(g11[inode,ielem], g12[inode,ielem], g21[inode,ielem], g22[inode,ielem])
            g_c1[inode,ielem] = tmp[1]
            g_c2[inode,ielem] = tmp[2]
            g_c3[inode,ielem] = tmp[3]
            g_c4[inode,ielem] = tmp[4]
        end
    end
    
    return SMatrix{2,2}(g_c1,g_c2,g_c3,g_c4)
end

# The alpha elements of each node
function alpha_values_2d(orography, meshes, meshdata_tfc, meshdata_c ; stretching = stretching_equidistant, face = false)

    @unpack a1, a3, H_top, num_elem = meshes

    # num_nodes = Number of nodal points on each element if face=false, Number of face points if face = true
    if(face)
        num_nodes = size(meshdata_tfc.xf)[1]
        x3_tfc = meshdata_tfc.yf;
        x1_c = meshdata_c.xf;
    else
        num_nodes = size(meshdata_tfc.x)[1]
        x3_tfc = meshdata_tfc.y;
        x1_c = meshdata_c.x;
    end

    xi = a3 .* x3_tfc ./H_top;

    derivate1 = alpha_3_1.(orography, x1_c, xi, a1,stretching = stretching)
    derivate2 = alpha_3_3.(orography, x1_c, xi, a3, H_top , stretching = stretching)

    alpha = SMatrix{2,2}(a1 .* ones(num_nodes,num_elem),zeros(num_nodes,num_elem),derivate1,derivate2)

    return alpha
end

# The beta elements of each node/ interpolation point stored by [b^1_1' , b^1_2' ; b^2_1' , b^2_2']
function beta_values_2d(meshes, alpha, g_contra)
    
    @unpack a1, num_elem = meshes
    
    # Number of nodal points on each element
    num_nodes = size(alpha[1])[1]

    beta3 = alpha[1] .* g_contra[3] .+ alpha[2] .* g_contra[4]
    beta4 = alpha[3] .* g_contra[3] .+ alpha[4] .* g_contra[4]

    beta = SMatrix{2,2}(1/a1 .* ones(num_nodes,num_elem),zeros(num_nodes,num_elem),beta3 ,beta4)

    return beta
end