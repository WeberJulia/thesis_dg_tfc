using StartUpDG
using StaticArrays
using LinearAlgebra
using Plots
using Base.Threads
using TickTock

#set Threadnumber before running the Code global
#export JULIA_NUM_THREADS=4 #Linux 
#set JULIA_NUM_THREADS=6 #Windows: settings of VS (settings.json)
println(string("NThreads=",Threads.nthreads()))

tick()

# set the pahts for saving the visualizations
cd(dirname(@__FILE__()))
path_figures = dirname(@__FILE__())*"/Pictures/"
path_sim = dirname(@__FILE__())*"/Pictures/Simulation/"
title_save = "lin-advection-3d"
title = "Linear advection"

NDIMS = 3
NPOLY = 3
NNODE = NPOLY+1
NCELL = 10

nodes, weights = StartUpDG.gauss_lobatto_quad(0, 0, NPOLY);

etype = Hex()
refdata = RefElemData(etype, Polynomial(), NPOLY)

# Build DG operators.
# volume quadrature weights, volume interpolation matrix, mass matrix, differentiation matrix
@unpack wq, Vq, M, Drst = refdata

weakDGMat = map(D -> M \ ((Vq * D)' * Diagonal(wq)), Drst);

# Preparing mesh.
cells, conn = uniform_mesh(etype,NCELL,NCELL,NCELL);

# Building mesh data: connectivity, Jacobians, ...
meshdata = make_periodic(MeshData(cells, conn, refdata))

# Helper routines for iterating.
@inline each_dof_global(rd,md) = Base.OneTo(rd.Np * md.num_elements)
@inline each_face_node_global(rd,md) = Base.OneTo(rd.Nfq * md.num_elements)


# Inverse Jacobian.
invJ = inv.(meshdata.J);

# Simple 3D advection scheme.
a = [1.0,0.0,0.0] # advection speeds

@inline function initial_condition(x,y,z; xc = nothing, yc = nothing, zc = nothing #= xc,yc: element centers =#)
    # 3D Gaussian bell
    return exp(-((x-0)^2 + (y-0)^2 + (z-0)^2))
end

# Rotated max. signal speed.
@inline function max_signal_speed(uM,uP,nn)
    return norm(a .* nn)
end

# Rotated flux.
@inline function rflux(u,nn)
    return sum(a .* nn)  * u
end

# Contravariant flux in x-direction.
@inline function fflux(u,rxJ,ryJ,rzJ)
    return (a[1]*rxJ + a[2]*ryJ + a[3]*rzJ) * u
end

# Contravariant flux in y-direction.
@inline function gflux(u,sxJ,syJ,szJ)
    return (a[1]*sxJ + a[2]*syJ + a[3]*szJ) * u
end

# Contravariant flux in z-direction.
@inline function hflux(u,txJ,tyJ, tzJ)
    return (a[1]*txJ + a[2]*tyJ + a[3]*tzJ) * u
end

# Rotated Rusanov flux.
@inline function riemann(uM,uP,nn) 
    return 0.5*(rflux(uM,nn) + rflux(uP,nn)) - 0.5*max_signal_speed(uM,uP,nn) * (uP - uM)
end

function calc_rhs(u,meshdata,refdata,weakDGMat,invJ)
    @unpack rxJ,sxJ, txJ,ryJ,syJ, tyJ, rzJ,szJ, tzJ = meshdata
    @unpack mapM,mapP,nxyzJ,Jf = meshdata
    @unpack Vf,Vq = refdata
    @unpack rstxyzJ = meshdata
    
    # Transform to quadrature space.
    uq = Vq * u
  
    # Compute the volume integral.    
    R = weakDGMat[1] * fflux.(uq,Vq*rxJ,Vq*ryJ, Vq*rzJ) + weakDGMat[2] * gflux.(uq,Vq*sxJ,Vq*syJ, Vq*szJ) + weakDGMat[3] * hflux.(uq,Vq*txJ,Vq*tyJ, Vq*tzJ)

    # Prolongate to faces.
    u_faces = Vf*u    

    # Calculate interface fluxes.
    f_faces = similar(u_faces)
    Threads.@threads for iface in each_face_node_global(refdata,meshdata)
        iM, iP = mapM[iface], mapP[iface]
        uM = u_faces[iM]
        uP = u_faces[iP]
        nn = SVector{NDIMS}(getindex.(nxyzJ, iM)) # surface normal
        
        f_faces[iface] = riemann(uM,uP,nn)
    end

    # Add surface terms.
    @unpack LIFT = refdata
    R -= LIFT * f_faces

    # Scale to mesh size.
    return R .* invJ
end

# Fill initial states.
u = initial_condition.(meshdata.x, meshdata.y, meshdata.z);

# Visualize initial states.
vtu_name = MeshData_to_vtk(meshdata, refdata, [u], [title], path_figures*string(title_save,"-init"), true)

# Temporary array for Runge-Kutta scheme.
tmp = zero(u)

FinalTime = 1.0
dt = 1e-3

Nsteps = ceil(FinalTime / dt)

# Runge-Kutta coefficients.
rk4a, rk4b, rk4c = ck45()

for i = 1:Nsteps
    # Explicit Runge-Kutta.
    #j = (i-1)*Nsteps
    for istage = 1:5
        rhs = calc_rhs(u,meshdata,refdata,weakDGMat,invJ)
        @. tmp = rk4a[istage] * tmp + dt * rhs
        @. u =  u + rk4b[istage] * tmp
    end
    #vtu_sim = MeshData_to_vtk(meshdata, refdata, [u], [title], path_sim*string(title_save,"_u",Int32(j)), true)
    #j +=1
end

@inline function analytical_solution(x,y,z,t)
    x, y, z = x - a[1]*t, y - a[2]*t, z - a[3]*t
    if(x <= -1)
        x += 2
    end
    if(y <= -1)
        y += 2
    end
    if(z <= -1)
        z += 2
    end
    return initial_condition(x,y,z)
end

u_ana = analytical_solution.(meshdata.x, meshdata.y, meshdata.z, FinalTime)
# error = u-u_ana
# sq_error = norm(error)/length(vec(u_ana))
# println(minimum(abs.(error)))
# println(maximum(abs.(error)))
# println(sq_error)

vtu_name = MeshData_to_vtk(meshdata, refdata, [u], [title], path_figures*string(title_save,"-final"), true)
vtu_name = MeshData_to_vtk(meshdata, refdata, [u_ana], [title], path_figures*string(title_save,"-final-ana"), true)
tock()

# Plotting the errors for NCELL=2,4,8,16,32
# using Plots
# x = [2, 4, 8, 16, 32]
# y = [0.000797, 0.000129, 1.924*10^(-5), 2.747*10^(-6), 3.842*10^(-7)]
# plot(x,y, title="Error of the 3d advection problem \n in the DG approach", xlabel="number of elements in each direction", ylabel = "error", legend = false, yaxis=:log10)
# savefig(path_figures*"error-3d-no-tfc")