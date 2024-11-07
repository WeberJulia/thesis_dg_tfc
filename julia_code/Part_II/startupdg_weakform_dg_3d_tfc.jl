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
title_save = "lin-advection-3d-tfc"
title = "Linear advection with terrain-following mesh"

println(string("NThreads=",Threads.nthreads()))

NDIMS = 3
NPOLY = 3
NNODE = NPOLY+1
N_hc1 = 10
N_hc2 = N_hc1
N_vc = N_hc1

Lx1 = 4
Lx2 = Lx1
H_top = Lx1
center = Lx1/2      #center of the computational domain, for inital condition
amplitude = H_top/4 #hight of the obstacle, variable for orography_sine


@inline function orography_zero(x,y)
    return zero.(x)
end
@inline function orography_zero(x)
    return orography_zero(x[1], x[2])
end

@inline function orography_sine(x)
    #b = 2*pi /period
    b = 2*pi /Lx1
    return amplitude .* sin.(x[1] .*b) .+ amplitude
end

@inline function orography_sine(x,y)
    b = 2*pi /Lx1
    return amplitude .* sin.(x .*b) .+ amplitude
end

nodes, weights = StartUpDG.gauss_lobatto_quad(0, 0, NPOLY);

etype = Hex()
refdata = RefElemData(etype, Polynomial(), NPOLY)

# Build DG operators.
# volume quadrature weights, volume interpolation matrix, mass matrix, differentiation matrix
@unpack wq, Vq, M, Drst = refdata

weakDGMat = map(D -> M \ ((Vq * D)' * Diagonal(wq)), Drst);

# Preparing meshes.
meshes = Meshes(orography_sine, [Lx1, Lx2], H_top, [N_hc1, N_hc2, N_vc])
@unpack cells_tfc, conn = meshes

# Building mesh data: connectivity, Jacobians, ...
md_tfc = make_periodic(MeshData(cells_tfc, conn, refdata))

md_c = get_meshdata_cartesian(orography_sine, meshes, md_tfc)
md_c = MeshData(refdata, md_tfc, md_c[1], md_c[2], md_c[3])

#get transformation parameters for the PDE
transformation =  transformation_parameters(orography_sine, meshes, md_tfc, md_c)

# Helper routines for iterating.
@inline each_dof_global(rd,md) = Base.OneTo(rd.Np * md.num_elements)
@inline each_face_node_global(rd,md) = Base.OneTo(rd.Nfq * md.num_elements)


# Inverse Jacobian.
invJ = inv.(md_tfc.J);

# 3D advection scheme in Cartesian coordinates.
a = [1.0,0.0,0.0] # advection speeds

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

function initial_condition(x,y,z; xc = nothing, yc = nothing, zc = nothing #= xc,yc: element centers =#)
    # 3D Gaussian bell
    return exp(-((x-center)^2 + (y-center)^2 + (z-center)^2))
end


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
        
        f_faces[iface] = riemann(am, ap, uM,uP,nn)  # advection velocity is location-dependant
    end

    # Add surface terms.
    @unpack LIFT = refdata
    R -= LIFT * f_faces

    # Scale to mesh size.
    return R .* invJ ./ g_scalar_sqrt
end

# Fill initial states.
u = initial_condition.(md_c.x, md_c.y, md_c.z);
u_init = u

# Visualize initial states.
vtu_name = MeshData_to_vtk(md_c, refdata, [u], [title], path_figures*string(title_save,"-init"), true)

# minimum distance inbetween the reerence element for CFL-condition
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

FinalTime = 4.0
HalfTime = 2.0
CFL = 0.01
dt = 1e-3 #CFL * min_rd(refdata) / maximum(a[1]+a[2]+a[3])

Nsteps = ceil(FinalTime / dt)
Nhalf = ceil(HalfTime / dt)
println(Nsteps)

# Runge-Kutta coefficients.
rk4a, rk4b, rk4c = ck45()

for i = 1:Nsteps
    # Explicit Runge-Kutta.
    #j = (i-1)*Nsteps
    for istage = 1:5
        rhs = calc_rhs(u,md_tfc,refdata,weakDGMat,invJ)
        @. tmp = rk4a[istage] * tmp + dt * rhs
        @. u =  u + rk4b[istage] * tmp
    end
    if(i==Nhalf)
        vtu_name_half = MeshData_to_vtk(md_c, refdata, [u], [title], path_figures*string(title_save,"-half"), true)
    end
   #vtu_sim = MeshData_to_vtk(md_c, refdata, [u], [title], path_sim*string(title_save,"_u",Int32(j)), true)
   #j +=1
end

vtu_name = MeshData_to_vtk(md_c, refdata, [u], [title], path_figures*string(title_save,"-final"), true)

nz = 1
nnode = nz*(N_hc2+1)*(N_hc1+1)+(N_hc2+1)*(N_hc1+1)
nelem = N_hc1*N_hc2*nz
cells_x = cells_tfc[1][1:nnode]
cells_y = cells_tfc[2][1:nnode]
cells_z = cells_tfc[3][1:nnode]
c, conn_new = uniform_mesh(Hex(),N_hc1,N_hc2,nz)
md_low = MeshData((cells_x, cells_y, cells_z), conn_new, refdata)
md_c = get_meshdata_cartesian(orography_sine, meshes, md_low)
md_c = MeshData(refdata, md_low, md_c[1], md_c[2], md_c[3])
vtu_name = MeshData_to_vtk(md_c, refdata, [zero.(md_low.x)], [string(title,"orography")], path_figures*string(title_save,"orography"), true)
error = u-u_init
sq_error = norm(error)/length(vec(u))
println(minimum(abs.(error)))
println(maximum(abs.(error)))
println(sq_error)

tock()