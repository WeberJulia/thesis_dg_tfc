# # based on the paper: A horizontally explicit, vertically implicit (HEVI) discontinuous Galerkin scheme 
# for the 2-dimensional Euler and Navier-Stokes equations using terrain-following coordinates (M. Baldauf)
# The terrain-following parameters are stored with the abbreviation tfc and the Cartesian ones are stored with c

using LinearAlgebra
using ForwardDiff
using StartUpDG
using StaticArrays

function print_line(x)
    x = round.(x; digits = 4)
    for i in 1:size(x)[1]
        println(x[i,:])
    end
    println()
end

# Struct for the paramteters that characterises the meshes in Cartesian and terrain-following coordinates in 2D
struct Meshes_struct_3D{A1, A2, A3, h_top, Cells_tfc, Cells_c, Conn, Num_elem, DIM}
    a1::A1
    a2::A2
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

# The function generates a mesh with N_hc1 x N_hc2 x N_vc unit cells (Hexagons) on [0,2N_hc1]x[0,2N_hc2]x[0,2N_vc]. 
# The DG method is applied to this this mesh. It is stored in the same format than (cells, conn) of the unit_mesh of StartUpDG.jl 
function tfc_mesh(N_hc1, N_hc2, N_vc)

    num_elem = N_vc * N_hc1 * N_hc2;

    # # Create connectivity matrix
    nodes_per_level = (N_hc1+1)*(N_hc2+1)
    conn = zeros(num_elem,8);
    ielem = 0
    for v in 1:N_vc
        for h2 in 1:N_hc2
            for h1 in 1:N_hc1
                edge = h1 + (h2-1)*(N_hc1+1) + (v-1)*nodes_per_level;
                conn[ielem + h1, 1] = edge;
                conn[ielem + h1, 2] = edge + N_hc1 + 1;
                conn[ielem + h1, 3] = edge + 1;
                conn[ielem + h1, 4] = edge + N_hc1 + 2;
                conn[ielem + h1, 5] = edge + nodes_per_level;
                conn[ielem + h1, 6] = edge + N_hc1 + 1 + nodes_per_level;
                conn[ielem + h1, 7] = edge + 1 + nodes_per_level;
                conn[ielem + h1, 8] = edge + N_hc1 + 2 + nodes_per_level;
            end
            ielem += N_hc1
        end
    end
    conn = convert(Array{Int64}, conn)

    # Create cells [0,2N_hc1]x[0,2N_hc2]x[0,2N_vc]
    num_nodes = (N_hc1+1)*(N_hc2+1)*(N_vc+1)
    x = zeros(num_nodes);
    y = zeros(num_nodes);
    z = zeros(num_nodes);
    step = 2;
    inode = 1;
    for iz in 1:N_vc+1
        for iy in 1:N_hc2+1
            for ix = 1:N_hc1+1
                x[inode] = step * (ix-1)
                y[inode] = step * (iy-1)
                z[inode] = step * (iz-1)
                inode += 1
            end
        end
    end
    cells = (x,y,z)

    return (cells, conn)
end
# function tfc_mesh(N_hc1, N_hc2, N_vc)

#     cells, conn = uniform_mesh(Hex(), N_hc1, N_hc2, N_vc)
#     cells[1] .= (cells[1] .+ 1) 
#     cells[2] .= (cells[2] .+ 1) 
#     cells[3] .= (cells[3] .+ 1) 

#     cells[1] .= cells[1] .*(N_hc1)
#     cells[2] .= cells[2] .*(N_hc2)
#     cells[3] .= cells[3] .*(N_vc)

#     return (cells, conn)
# end


# Distributes the x3 values between the model bottom, which is described by orography(x1_c = a1*x1_tfc) and the model top H_top.
# The distribution is described by the stretching function
@inline function cart_vertical(orography, x1_tfc, x2_tfc, x3_tfc, a, H_top; stretching = stretching_equidistant)
    return orography.(a[1] .* x1_tfc, a[2] .* x2_tfc) .+ (H_top .- orography.(a[1] .* x1_tfc, a[2] .* x2_tfc)) .*stretching.(a[3] .*x3_tfc ./H_top)
end

# The start mesh is a cartesian mesh. It based on equidistant x1 values. The bottom of the mesh is described by orography(x1).
# From each orography(x1) value the the distribution of the x3 values to the model top H_top are defined by the stretching function. 
# The cartesian mesh mesh is stored in the same format than the tfc-mesh.
@inline function cartesian_mesh(orography, cells_tfc, a, H_top; stretching = stretching_equidistant)
    x1_tfc = cells_tfc[1];
    x2_tfc = cells_tfc[2];
    x3_tfc = cells_tfc[3];
    
    x1_c = a[1] .*x1_tfc;
    x2_c = a[2] .*x2_tfc;
    x3_c = cart_vertical(orography, x1_tfc, x2_tfc, x3_tfc, a, H_top, stretching = stretching)

    return (x1_c, x2_c, x3_c)
end

""" Meshes calculates the x and y values of the nodes of the Cartesian and the tfc mesh.
The struct stores the cells of the meshes and their connectivity with important scaling parameters.
The connectivity does not change by transforming the Cartesian mesh to the tfc mesh. 
Contains: a1, a3, H_top, cells_tfc, cells_c, conn, num_elem """
function Meshes_3D(orography, Lx1, Lx2, H_top, N_hc1, N_hc2, N_vc, dim; stretching = stretching_equidistant)
    
    cells_tfc, conn = tfc_mesh(N_hc1, N_hc2, N_vc)
    
    a1 = Lx1/(2*N_hc1); 
    a2 = Lx2/(2*N_hc2);
    a3 = H_top/(2*N_vc);

    cells_c = cartesian_mesh(orography, cells_tfc, [a1, a2, a3], H_top, stretching = stretching)

    num_elem = N_hc1*N_hc2*N_vc;
    variables = Meshes_struct_3D(a1, a2, a3, H_top, cells_tfc, cells_c, conn, num_elem, dim)
end

@inline function grad(func, x, y)
    ForwardDiff.gradient(func, [x,y])
end

@inline function grad_orography(func, x,y)
    dx = grad.(func,x,y)
    return (getindex.(dx,1), getindex.(dx,2))
end

# Calculation of alpha_3_i = dx3_cart/dxi_tfc for i=1,2
@inline function alpha_3_i(dx_orography, xi, ai; stretching = stretching_equidistant)
    return dx_orography*ai*(1-stretching(xi))
end

# Calculation of alpha_3_3 = dx3_cart/dx3_tfc
@inline function alpha_3_3(orography, x1_c, x2_c, xi, a3, H_top; stretching = stretching_equidistant)
    return (H_top - orography([x1_c, x2_c]))*derivate(stretching,xi)*(a3/H_top)
end

# Function that calculates the covariant metric tensor of a 3d transformation 
# Based on the 2d idea of the paper from M. Baldauf
# As the covariant metric tensor is symmetric (g[2,1] = g[1,2]) we only need this information once. 
function covariant_metric_tensor_3d(a1, a2, alpha)
    rxG = a1^2 .+ (alpha[3]) .^2    #g11
    sxG = (alpha[3]) .* (alpha[6])  #g12
    txG = alpha[3] .* alpha[9]      #g13
    ryG = sxG                       #g21
    syG = a2^2 .+ (alpha[6]) .^2    #g22           
    tyG = alpha[6] .* alpha[9]      #g23
    rzG = txG                       #g31
    szG = tyG                       #g32
    tzG = alpha[9] .^2              #g33
    return SMatrix{3,3}(rxG, sxG, txG, ryG, syG, tyG, rzG, szG, tzG)
end

# Function for calculating the metric scalar density of a covariant metric tensor
@inline function metric_scalar_density_3d(g11, g12, g13, g21, g22, g23, g31, g32, g33)
    return det([g11 g12 g13;g21 g22 g23; g31 g32 g33])
end

@inline function inverse_metric(g11, g12, g13, g21, g22, g23, g31, g32, g33)
    tmp = inv([g11 g12 g13;g21 g22 g23; g31 g32 g33])
    return (tmp[1,1], tmp[1,2], tmp[1,3], tmp[2,1], tmp[2,2], tmp[2,3], tmp[3,1], tmp[3,2], tmp[3,3])
end

function contravariant_metric_tensor_3d(g_cov)
    g11, g12, g13 = g_cov[1], g_cov[2], g_cov[3]
    g21, g22, g23 = g_cov[4], g_cov[5], g_cov[6]
    g31, g32, g33 = g_cov[7], g_cov[8], g_cov[9]

    g_c1, g_c2, g_c3 = zero(g11), zero(g11), zero(g11) 
    g_c4, g_c5, g_c6 = zero(g11), zero(g11), zero(g11) 
    g_c7, g_c8, g_c9 = zero(g11), zero(g11), zero(g11)   

    num_nodes, num_elem = size(g11)
    
    for inode in 1:num_nodes
        for ielem in 1:num_elem
            tmp = inverse_metric(g11[inode,ielem], g12[inode,ielem], g13[inode,ielem], g21[inode,ielem], g22[inode,ielem], g23[inode,ielem], g31[inode,ielem], g32[inode,ielem],g33[inode,ielem])
            g_c1[inode,ielem], g_c2[inode,ielem], g_c3[inode,ielem] = tmp[1], tmp[2], tmp[3]
            g_c4[inode,ielem], g_c5[inode,ielem], g_c6[inode,ielem] = tmp[4], tmp[5], tmp[6]
            g_c7[inode,ielem], g_c8[inode,ielem], g_c9[inode,ielem] = tmp[7], tmp[8], tmp[9]
        end
    end
    
    return SMatrix{3,3}(g_c1,g_c2,g_c3,g_c4,g_c5,g_c6,g_c7, g_c8, g_c9)
end

# The alpha elements of each node/ interpolation point stored by [a^1'_1, a^1'_2 ; a^2'_1 , a^2'_2]
function alpha_values_3d(orography, meshes, meshdata_tfc, meshdata_c ; stretching = stretching_equidistant, face = false)

    @unpack cells_tfc, cells_c, conn = meshes
    @unpack a1, a2, a3, H_top, num_elem = meshes

    if(face)
        x1_c = meshdata_c.xf;
        x2_c = meshdata_c.yf;
        x3_tfc = meshdata_tfc.zf;
        
        # Number of nodal points on each element
        num_nodes = size(meshdata_tfc.xf)[1]
    else
        x1_c = meshdata_c.x;
        x2_c = meshdata_c.y;
        x3_tfc = meshdata_tfc.z;
        
        # Number of nodal points on each element
        num_nodes = size(meshdata_tfc.x)[1]
    end 
    xi = a3 .* x3_tfc ./H_top;
    
    dx_orography = grad_orography(orography, x1_c, x2_c);
    derivate1 = alpha_3_i.(dx_orography[1], xi, a1,stretching = stretching);
    derivate2 = alpha_3_i.(dx_orography[2], xi, a2,stretching = stretching);
    derivate3 = alpha_3_3.(orography, x1_c, x2_c, xi, a3, H_top , stretching = stretching);
    zero = zeros(num_nodes,num_elem);
    one = ones(num_nodes,num_elem);
    
    alpha = SMatrix{3,3}(a1 .* one,zero,derivate1,zero, a2.* one, derivate2, zero, zero, derivate3)

    return alpha
end

# The beta elements of each node/ interpolation point stored by [b^1_1' , b^1_2' ; b^2_1' , b^2_2']
function beta_values_3d(meshes, alpha, g_contra)
    
    @unpack a1, a2, num_elem = meshes
    
    # Number of nodal points on each element
    num_nodes = size(alpha[1])[1]

    beta7 = alpha[1] .* g_contra[7] .+ alpha[2] .* g_contra[8] .+ alpha[3] .* g_contra[9]
    beta8 = alpha[4] .* g_contra[7] .+ alpha[5] .* g_contra[8] .+ alpha[6] .* g_contra[9]
    beta9 = alpha[7] .* g_contra[7] .+ alpha[8] .* g_contra[8] .+ alpha[9] .* g_contra[9]

    one = ones(num_nodes,num_elem);
    zero = zeros(num_nodes,num_elem);

    beta = SMatrix{3,3}(1/a1 .* one, zero, zero, zero, 1/a2 .* one, zero, beta7 ,beta8, beta9)

    return beta
end