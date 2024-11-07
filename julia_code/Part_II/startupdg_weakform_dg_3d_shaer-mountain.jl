include("terrainFollowingCoordinates.jl")
using StartUpDG
using StaticArrays
using LinearAlgebra
using Plots
using TickTock
using Base.Threads

tick()
cd(dirname(@__FILE__()))
path_figures = dirname(@__FILE__())*"/Pictures/"
path_sim = dirname(@__FILE__())*"/Pictures/Simulation/"
title_save = "lin-advection"
title = "Linear advection"

println(string("NThreads=",Threads.nthreads()))

NDIMS = 3
NPOLY = 3
NNODE = NPOLY+1
N_hc1 = 35
N_hc2 = N_hc1
N_vc = 15
println(string("N_tfc = ",N_hc1))

Lx1 = 150
Lx2 = Lx1
H_top = 25

# @inline function orography_zero(x)
#     return zero.(x[1])
# end

# @inline function orography_zero(x,y)
#     return zero.(x)
# end

function initial_condition(x,y,z; xc = nothing, yc = nothing, zc = nothing #= xc,yc: element centers =#)
    # 2D Gaussian bell
    return exp(-((x-2)^2 + (y-2)^2 + (z-2)^2))
end

@inline function h_star(x, ah)
    return cos(pi * x /(2*ah))^2
end

# Orography of the shaer_mountain test
@inline function shaer_mountain(x,y)
    # Lx1 = 300     Abb. 150
    # H_top = 25    Abb. 15
    h0 = 3
    ah = 25
    l = 8
    xi = x-Lx1/2
    yi = y-Lx2/2
    if(abs(xi)< ah && abs(yi)<ah)
        return cos(pi * xi /l)^2 * h_star(xi, ah)* h0 * cos(pi * yi /l)^2 * h_star(yi, ah)
    else 
        return 0
    end
end

@inline function shaer_mountain(x)
    return shaer_mountain(x[1], x[2])
end

nodes, weights = StartUpDG.gauss_lobatto_quad(0, 0, NPOLY);
# nodes, weights = StartUpDG.gauss_quad(0, 0, NPOLY);

etype = Hex()
refdata = RefElemData(etype, Polynomial(), NPOLY)

# Build DG operators.
# volume quadrature weights, volume interpolation matrix, mass matrix, differentiation matrix
@unpack wq, Vq, M, Drst = refdata

weakDGMat = map(D -> M \ ((Vq * D)' * Diagonal(wq)), Drst);

# Preparing meshes.
meshes = Meshes(shaer_mountain, [Lx1, Lx2], H_top, [N_hc1, N_hc2, N_vc])
@unpack cells_tfc, conn = meshes

# Building mesh data: connectivity, Jacobians, ...
md_tfc = make_periodic(MeshData(cells_tfc, conn, refdata))

md_c = get_meshdata_cartesian(shaer_mountain, meshes, md_tfc)
md_c = MeshData(refdata, md_tfc, md_c[1], md_c[2], md_c[3])

transformation =  transformation_parameters(shaer_mountain, meshes, md_tfc, md_c)

# Helper routines for iterating.
@inline each_dof_global(rd,md) = Base.OneTo(rd.Np * md.num_elements)
@inline each_face_node_global(rd,md) = Base.OneTo(rd.Nfq * md.num_elements)


# Inverse Jacobian.
invJ = inv.(md_tfc.J);
invJ_test = inv.(md_c.J);

@inline function advection_shaer_mountain(x,y,z)
    a0 = 10
    z1 = 4
    z2 = 5
    if(z2 <= z)
        return a0
    elseif(z1 <= z)
        a0 * sin((pi/2)*(z-z1)/(z2-z1))^2 
    else
        return 0
    end
end

# Simple 2D advection scheme.
a = [10.0,0.0,0.0] # advection speeds
# a_test = a # test without orography

# only for shaer_mountain
a_test = [Float64.(advection_shaer_mountain.(md_c.x, md_c.y, md_c.z)), zero(md_c.y), zero(md_c.z)]
af_test = [Float64.(advection_shaer_mountain.(md_c.xf, md_c.yf, md_c.zf)), zero(md_c.yf), zero(md_c.zf)]


# Transformation of the cartesian function values to the tfc values in x-direction
@inline function transform_x_direction(a, rxB, sxB, txB)
    return (a[1] .* rxB + a[2] .* sxB + a[3] .* txB) 
end

# Transformation of the cartesian function values to the tfc values in y-direction
@inline function transform_y_direction(a, ryB, syB, tyB)
    return (a[1] .* ryB + a[2] .* syB + a[3] .* tyB) 
end

# Transformation of the cartesian function values to the tfc values in z-direction
@inline function transform_z_direction(a, rzB, szB, tzB)
    return (a[1] .* rzB + a[2] .* szB + a[3] .* tzB)
end

# Transformation of the cartesian function values to the tfc values
@inline function transform_a(a, rxB, ryB, rzB, sxB, syB, szB, txB, tyB, tzB, g)
    return [g .* transform_x_direction(a, rxB, sxB, txB), g.* transform_y_direction(a, ryB, syB, tyB), 
    g.* transform_z_direction(a, rzB, szB, tzB)]
end

@unpack g_scalar_sqrt, g_scalarf_sqrt = transformation
@unpack  rxB, ryB, rzB, sxB, syB, szB, txB, tyB, tzB = transformation
@unpack  rxBf, ryBf, rzBf, sxBf, syBf, szBf, txBf, tyBf, tzBf = transformation

# Transformation of the advection velocity
af = transform_a(a, rxBf, ryBf, rzBf, sxBf, syBf, szBf, txBf, tyBf, tzBf, g_scalarf_sqrt)
a = transform_a(a, rxB, ryB, rzB, sxB, syB, szB, txB, tyB, tzB, g_scalar_sqrt)

# Rotated max. signal speed.
@inline function max_signal_speed(am, ap, uM,uP,nn)
    return maximum([norm(am .* nn),norm(ap .* nn)])
end

# Rotated flux.
@inline function rflux(ai, u,nn)
    return sum(ai .* nn)  * u
end

# Contravariant flux in x-direction.
function fflux(u,rxJ,ryJ,rzJ)
    return @. (a[1] * rxJ + a[2] * ryJ + a[3] * rzJ) * u
end

# Contravariant flux in y-direction.
function gflux(u,sxJ,syJ,szJ)
    return @. (a[1] * sxJ + a[2] * syJ + a[3] * szJ) * u
end

# Contravariant flux in z-direction.
function hflux(u,txJ,tyJ, tzJ)
    return @. (a[1] * txJ + a[2] * tyJ + a[3] * tzJ) * u
end

# Contravariant flux in x-direction.
function fflux_t(u,rxJ,ryJ,rzJ)
    return @. (a_test[1] * rxJ + a_test[2] * ryJ + a_test[3] * rzJ) * u
end

# Contravariant flux in y-direction.
function gflux_t(u,sxJ,syJ,szJ)
    return @. (a_test[1] * sxJ + a_test[2] * syJ + a_test[3] * szJ) * u
end

# Contravariant flux in z-direction.
function hflux_t(u,txJ,tyJ, tzJ)
    return @. (a_test[1] * txJ + a_test[2] * tyJ + a_test[3] * tzJ) * u
end

# Rotated Rusanov flux.
function riemann(am, ap, uM,uP,nn) 
    return 0.5*(rflux(am, uM,nn) + rflux(ap, uP,nn)) - 0.5*max_signal_speed(am,ap, uM,uP,nn) * (uP - uM)
end

function calc_rhs(u,meshdata,refdata,weakDGMat,invJ)
    @unpack rxJ,sxJ, txJ,ryJ,syJ, tyJ, rzJ,szJ, tzJ = meshdata
    @unpack mapM,mapP,nxyzJ,Jf = meshdata
    @unpack Vf,Vq = refdata
    @unpack rstxyzJ = meshdata
    
    # Transform to quadrature space.
    uq = Vq * u
  
    # Compute the volume integral.    
    R = weakDGMat[1] * fflux(uq,Vq*rxJ,Vq*ryJ, Vq*rzJ) + weakDGMat[2] * gflux(uq,Vq*sxJ,Vq*syJ, Vq*szJ) + weakDGMat[3] * hflux(uq,Vq*txJ,Vq*tyJ, Vq*tzJ)
    
    # Prolongate to faces.
    u_faces = Vf*u    

    # Calculate interface fluxes.
    f_faces = similar(u_faces)
    
    Threads.@threads for iface in each_face_node_global(refdata,meshdata)
        iM, iP = mapM[iface], mapP[iface]
        uM = u_faces[iM]
        uP = u_faces[iP]
        nn = SVector{NDIMS}(getindex.(nxyzJ, iM)) # surface normal
        am = SVector{NDIMS}(getindex.(af, iM))
        ap = SVector{NDIMS}(getindex.(af, iP))
        
        f_faces[iface] = riemann(am, ap, uM,uP,nn)
    end

    # Add surface terms.
    @unpack LIFT = refdata
    R -= LIFT * f_faces

    # Scale to mesh size.
    return R .* invJ ./ g_scalar_sqrt
end

function calc_rhs_t(u,meshdata,refdata,weakDGMat,invJ)
    @unpack rxJ,sxJ, txJ,ryJ,syJ, tyJ, rzJ,szJ, tzJ = meshdata
    @unpack mapM,mapP,nxyzJ,Jf = meshdata
    @unpack Vf,Vq = refdata
    @unpack rstxyzJ = meshdata
    
    # Transform to quadrature space.
    uq = Vq * u
  
    # Compute the volume integral.    
    R = weakDGMat[1] * fflux_t(uq,Vq*rxJ,Vq*ryJ, Vq*rzJ) + weakDGMat[2] * gflux_t(uq,Vq*sxJ,Vq*syJ, Vq*szJ) + weakDGMat[3] * hflux_t(uq,Vq*txJ,Vq*tyJ, Vq*tzJ)
    
    # Prolongate to faces.
    u_faces = Vf*u    

    # Calculate interface fluxes.
    f_faces = similar(u_faces)
    
    Threads.@threads for iface in each_face_node_global(refdata,meshdata)
        iM, iP = mapM[iface], mapP[iface]
        uM = u_faces[iM]
        uP = u_faces[iP]
        nn = SVector{NDIMS}(getindex.(nxyzJ, iM)) # surface normal
        am = SVector{NDIMS}(getindex.(af_test, iM)) # only for shaer_mountain
        ap = SVector{NDIMS}(getindex.(af_test, iP)) # only for shaer_mountain
        

        f_faces[iface] = riemann(am, ap, uM,uP,nn) # only for shaer_mountain
        # f_faces[iface] = riemann(am, ap, uM,uP,nn) # test without orography
    end

    # Add surface terms.
    @unpack LIFT = refdata
    R -= LIFT * f_faces

    # Scale to mesh size.
    return R .* invJ
end

@inline function initial_shaer_mountain(x,y,z)
    xi = x-Lx1/2
    yi = y-Lx2/2
    xr = xi + 50
    yr = yi 
    zr = z - 9
    r = sqrt((xr/25)^2+(yr/25)^2+(zr/3)^2)
    if(r <= 1)
        return cos(pi*r/2)^2
    else
        return 0
    end
end

# Fill initial condition
u = Float64.(initial_shaer_mountain.(md_c.x, md_c.y, md_c.z));
u_test = u

# Visualize initial states.
vtu_name = MeshData_to_vtk(md_c, refdata, [u], [title], path_figures*string(title_save,"-init"), true)

@inline function min_rd(rd)
    min = 2
    for i in 1:NPOLY
        imin = abs.(rd.s[i]-rd.s[i+1])
        if (imin < min)
            min = imin 
        end
    end
    return min
end


# Temporary array for Runge-Kutta scheme.
tmp = zero(u)
tmp_test = zero(u_test)

FinalTime = 5
halTime = 2.5
dt = 0.0025

Nsteps = ceil(FinalTime / dt)
Nhalf = ceil(halTime / dt)
println(Nsteps)
println(N_hc1*N_hc2*N_vc)

# Runge-Kutta coefficients.
rk4a, rk4b, rk4c = ck45()

for i = 1:Nsteps
    #Explicit Runge-Kutta.
    for istage = 1:5
        rhs = calc_rhs(u,md_tfc,refdata,weakDGMat,invJ)
        @. tmp = rk4a[istage] * tmp + dt * rhs
        @. u =  u + rk4b[istage] * tmp

        rhs_test = calc_rhs_t(u_test,md_c,refdata,weakDGMat,invJ_test)
        @. tmp_test = rk4a[istage] * tmp_test + dt * rhs_test
        @. u_test =  u_test + rk4b[istage] * tmp_test
    end
    if(i==Nhalf)
        vtu_name_half = MeshData_to_vtk(md_c, refdata, [u], [title], path_figures*string(title_save,"-half"), true)
        vtu_name_half = MeshData_to_vtk(md_c, refdata, [u_test], [title], path_figures*string(title_save,"-test-half"), true)
    end
end

vtu_name = MeshData_to_vtk(md_c, refdata, [u], [title], path_figures*string(title_save,"-final"), true)
vtu_name = MeshData_to_vtk(md_c, refdata, [u_test], [title], path_figures*string(title_save,"-test-final"), true)
@unpack cells_c = meshes

nz = 1
nnode = nz*(N_hc2+1)*(N_hc1+1)+(N_hc2+1)*(N_hc1+1)
nelem = N_hc1*N_hc2*nz
cells_x = cells_tfc[1][1:nnode]
cells_y = cells_tfc[2][1:nnode]
cells_z = cells_tfc[3][1:nnode]
c, conn_new = uniform_mesh(Hex(),N_hc1,N_hc2,nz)
md_low = MeshData((cells_x, cells_y, cells_z), conn_new, refdata)
md_c = get_meshdata_cartesian(shaer_mountain, meshes, md_low, stretching = stretching_paper)
md_c = MeshData(refdata, md_low, md_c[1], md_c[2], md_c[3])
vtu_name = MeshData_to_vtk(md_c, refdata, [zero.(md_low.x)], [string(title,"orography")], path_figures*string(title_save,"orography"), true)

error = u-u_test
sq_error = norm(error)/length(vec(u))
println(minimum(abs.(error)))
println(maximum(abs.(error)))
println(sq_error)

tock()