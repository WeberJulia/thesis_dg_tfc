include("terrainFollowingCoordinates.jl")
using StartUpDG, UnPack
using StaticArrays
using LinearAlgebra
using Plots
cd(dirname(@__FILE__()))
path_figures = dirname(@__FILE__())*"/Pictures/"
path_sim = dirname(@__FILE__())*"/Pictures/Simulation/"
title_save = "lin-advection-tfc"
title = "Linear advection on a terrain-following mesh"

NDIMS = 2;
NPOLY = 3;
NNODE = NPOLY+1;
# Number of cells
N_hc = 10;
N_vc = 10;

#length of the Cartesian mesh
Lx = 4;
H_top = 4;
# test Identity
# Lx = 2*N_hc;
# H_top = 2*N_vc;

# Define diffenent orographys
@inline function orography_zero(x)
    return zero.(x)
end

@inline function orography_sin(x)
        # b = 2*pi / Lx
        b = 2*pi /4
        return sin.(x .*b) .+ 1
end

@inline function orography_dwd_paper(x)
    # H0*exp(-x^2/(a_e^2))*cos(pi * x/a_c)^2
    return 10*exp(-x^2/25)*cos(pi * x/4)^2
end

nodes, weights = StartUpDG.gauss_lobatto_quad(0, 0, NPOLY);

etype = Quad()
refdata = RefElemData(etype, Polynomial(), NPOLY)

# Build DG operators.
# volume quadrature weights, volume interpolation matrix, mass matrix, differentiation matrix
@unpack wq, Vq, M, Drst = refdata

weakDGMat = map(D -> M \ ((Vq * D)' * Diagonal(wq)), Drst);

# Preparing mesh.
meshes = Meshes(orography_zero, Lx, H_top, [N_hc, N_vc])
@unpack cells_tfc, conn = meshes

# Building mesh data: connectivity, Jacobians, ...
meshdata_tfc = make_periodic(MeshData(cells_tfc, conn, refdata))

# As the solution is given in the Cartesian system, the initial function holds in the Cartesian system too.
meshdata_c = get_meshdata_cartesian(orography_zero, meshes, meshdata_tfc)
meshdata_c = MeshData(refdata, meshdata_tfc, meshdata_c[1], meshdata_c[2])

# Get transformationparameters of the meshdata, at each interpolation point the DG method has to solve another pde
transformation =  transformation_parameters(orography_zero, meshes, meshdata_tfc, meshdata_c)

meshplotter = MeshPlotter(refdata,meshdata_tfc)
plot(meshplotter, title = "tfc-mesh")
savefig(path_figures*"tfc-mesh.png")

# Helper routines for iterating.
@inline each_dof_global(rd,md) = Base.OneTo(rd.Np * md.num_elements)
@inline each_face_node_global(rd,md) = Base.OneTo(rd.Nfq * md.num_elements)

# Inverse Jacobian.
invJ = inv.(meshdata_tfc.J);

@inline function initial_condition(x,y; xc = nothing, yc = nothing #= xc,yc: element centers =#)
    # 2D Gaussian bell
    return exp(-((x-2.0)^2 + (y-2.0)^2))
end

# Fill initial condition
u_c = initial_condition.(meshdata_c.x, meshdata_c.y);

# Simple advection scheme.
a = [1.0,0.0] # advection speeds 

# Transformation of the cartesian velocity values to the tfc values in x-direction
@inline function transform_x_direction(a, rxB, sxB)
    return (a[1] .* rxB + a[2] .* sxB)
end

# Transformation of the cartesian function values to the tfc values in y-direction
@inline function transform_y_direction(a, ryB, syB)
    return (a[1].* ryB + a[2] .* syB) 
end

# Transformation of the cartesian function values to the tfc values
@inline function transform_a(a, rxB, ryB, sxB, syB, g)
    return [g .* transform_x_direction(a, rxB, sxB), g .*transform_y_direction(a, ryB, syB)]
end

@unpack g_scalar_sqrt, g_scalarf_sqrt = transformation
@unpack rxB, sxB, ryB, syB = transformation
@unpack rxBf, sxBf, ryBf, syBf = transformation

# Transformation of the advection velocity
af = transform_a(a, rxBf, ryBf, sxBf, syBf, g_scalarf_sqrt)
a = transform_a(a, rxB, ryB, sxB, syB, g_scalar_sqrt)

# Rotated max. signal speed.
@inline function max_signal_speed(a,uM,uP,nn)
    return norm(a .* nn)
end

# Rotated flux.
@inline function rflux(a,u,nn)
    return sum(a .* nn)  * u
end

# Contravariant flux in x-direction.
@inline function fflux(u,rxJ,ryJ)
    return (rxJ + ryJ) * u
end

# Contravariant flux in y-direction.
@inline function gflux(u,sxJ,syJ)
    return (sxJ + syJ) * u
end

# Rotated Rusanov flux.
@inline function riemann(a, uM,uP,nn)
    return 0.5*(rflux(a, uM,nn) + rflux(a, uP,nn)) - 0.5*max_signal_speed(a,uM,uP,nn) * (uP - uM)
end

function calc_rhs(u,meshdata_tfc,refdata,weakDGMat,invJ, transformation)
    @unpack rxJ,sxJ,ryJ,syJ = meshdata_tfc
    @unpack mapM,mapP,nxyzJ,Jf = meshdata_tfc
    @unpack Vf,Vq = refdata
    @unpack rstxyzJ = meshdata_tfc
    @unpack g_scalar_sqrt = transformation

    # Transform to quadrature space.
    uq = Vq * u

    # Compute the volume integral.
    R = weakDGMat[1] * fflux.(uq,Vq*(a[1] .*rxJ),Vq*(a[2] .*ryJ)) + weakDGMat[2] * gflux.(uq,Vq*(a[1] .*sxJ),Vq*(a[2] .* syJ))

    # Prolongate to faces.
    u_faces = Vf*u

    # Calculate interface fluxes.
    f_faces = similar(u_faces)
    for iface in each_face_node_global(refdata,meshdata_tfc)
        iM, iP = mapM[iface], mapP[iface]
        uM = u_faces[iM]
        uP = u_faces[iP]
        nn = SVector{NDIMS}(getindex.(nxyzJ, iM)) # surface normal
        ia = SVector{NDIMS}(getindex.(af, iM))    # advection velocity is location dependant

        f_faces[iface] = riemann(ia,uM,uP,nn)
    end
    
    # Add surface terms.
    @unpack LIFT = refdata

    R -= LIFT * f_faces

    # Scale to mesh size.
    return R .* invJ ./ g_scalar_sqrt
end

meshplotter = MeshPlotter(refdata,meshdata_c)
plot(meshplotter, title = "cartesian-mesh")
savefig(path_figures*"cartesian-mesh.png")

# Visualize initial states.
#@unpack Vp = refdata

# Interpolation to plotting points
#xp, yp, up = Vp * meshdata_c.x, Vp * meshdata_c.y, Vp * u_c

#scatter(xp, yp, up, zcolor=up, msw=0, leg=false, markersize=6.5*[1.0,1.0,1.0],ratio=1.0, cam=(0, 90),size = (600, 600))
# display(scatter(xp, yp, up, zcolor=up, msw=0, leg=false,ratio=1.0, cam=(0, 90),size = (600, 600), title = "init"))
# sleep(5)
# savefig(path_figures*string(title_save, "-init.png"))
vtu_name = MeshData_to_vtk(meshdata_c, refdata, [u_c], [title], path_figures*string(title_save,"-init"), true)

# Temporary array for Runge-Kutta scheme.
tmp = zero(u_c)

FinalTime = 2.0
dt = 1e-3

Nsteps = ceil(FinalTime / dt)
println(Nsteps)

# Runge-Kutta coefficients.
rk4a, rk4b, rk4c = ck45()

for i = 1:Nsteps
    # Explicit Runge-Kutta.
    for istage = 1:5
        rhs = calc_rhs(u_c,meshdata_tfc,refdata,weakDGMat,invJ, transformation)
        @. tmp = rk4a[istage] * tmp + dt * rhs
        @. u_c =  u_c + rk4b[istage] * tmp
    end

    if(i%2 == 0)
        vtu_sim = MeshData_to_vtk(meshdata_c, refdata, [u_c], [title], path_sim*string(title_save,"_",Int32(i/2)), true)
    end
end

#Visualize final state
# @unpack Vp = refdata
# xp, yp, up = Vp * meshdata_c.x, Vp * meshdata_c.y, Vp * u_c
# display(scatter(xp, yp, up, zcolor=up, msw=0, leg=false,ratio=1.0, cam=(0, 90),size = (600, 600), title = "final"))
# sleep(5)
# savefig(path_figures*string(title_save,"-final.png"))

vtu_name = MeshData_to_vtk(meshdata_c, refdata, [u_c], [title], path_figures*string(title_save,"-final"), true)

println(string("finished ",title))