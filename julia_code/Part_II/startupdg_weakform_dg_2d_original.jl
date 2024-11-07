using StartUpDG
using StaticArrays
using LinearAlgebra
using Plots

NDIMS = 2
NPOLY = 3
NNODE = NPOLY+1
NCELL = 10

nodes, weights = StartUpDG.gauss_lobatto_quad(0, 0, NPOLY);
# nodes, weights = StartUpDG.gauss_quad(0, 0, NPOLY);

println(nodes);
println(weights);

# etype = Tri()
etype = Quad()
refdata = RefElemData(etype, Polynomial(), NPOLY)

# Build DG operators.
# volume quadrature weights, volume interpolation matrix, mass matrix, differentiation matrix
@unpack wq, Vq, M, Drst = refdata

weakDGMat = map(D -> M \ ((Vq * D)' * Diagonal(wq)), Drst);

# Preparing mesh.
cells, conn = uniform_mesh(etype,NCELL,NCELL);

# Building mesh data: connectivity, Jacobians, ...
# meshdata = make_periodic(MeshData(cells, conn, refdata))
meshdata = MeshData(cells, conn, refdata)

meshplotter = MeshPlotter(refdata,meshdata)
plot(meshplotter)
savefig("mesh.png")

# Helper routines for iterating.
@inline each_dof_global(rd,md) = Base.OneTo(rd.Np * md.num_elements)
@inline each_face_node_global(rd,md) = Base.OneTo(rd.Nfq * md.num_elements)

## Uncomment this for curvilinear mesh.
#
# # Interpolate linearly between left and right value where s should be between -1 and 1
# linear_interpolate(s, left_value, right_value) = 0.5 * ((1 - s) * left_value + (1 + s) * right_value)
# 
# function bilinear_mapping(x, y, faces)
#   x1 = faces[1](-1) # Bottom left
#   x2 = faces[2](-1) # Bottom right
#   x3 = faces[1]( 1) # Top left
#   x4 = faces[2]( 1) # Top right
# 
#   return 0.25 * (x1 * (1 - x) * (1 - y) +
#                  x2 * (1 + x) * (1 - y) +
#                  x3 * (1 - x) * (1 + y) +
#                  x4 * (1 + x) * (1 + y))
# end
# 
# function transfinite_mapping(faces::NTuple{4, Any})
#   mapping(x, y) = (linear_interpolate(x, faces[1](y), faces[2](y)) +
#                    linear_interpolate(y, faces[3](x), faces[4](x)) -
#                    bilinear_mapping(x, y, faces))
# end
# 
# # Deformed rectangle that looks like a waving flag,
# xscale = 0.1
# yscale = 0.1
# 
# f1(s) = SVector(-1.0 + xscale * sin( pi * s), s)
# f2(s) = SVector( 1.0 + xscale * sin( pi * s), s)
# f3(s) = SVector(s, -1.0 + yscale * cos( pi * s))
# f4(s) = SVector(s,  1.0 + yscale * cos( pi * s))
# faces = (f1, f2, f3, f4)
# 
# mapping = transfinite_mapping(faces)
# 
# # Apply mapping to `meshdata`.
# @unpack x, y = meshdata
# x_ = similar(x)
# y_ = similar(y)
# 
# for i in each_dof_global(refdata,meshdata)
#     # x_[i],y_[i] = mapping(x[i],y[i])
#     x_[i] = (1.0 + x[i])*0.5
#     y_[i] = (1.0 + y[i])*0.5
# end
#     
# meshdata = MeshData(refdata, meshdata, x_, y_)

# Inverse Jacobian.
invJ = inv.(meshdata.J);

# Simple 2D advection scheme.
a = [1.0,0.2] # advection speeds

function initial_condition(x,y; xc = nothing, yc = nothing #= xc,yc: element centers =#)
    # 2D Gaussian bell
    #return exp(-(x^2 + (y-0.0)^2)/0.2^2)
    return sin(2*pi*x) 
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
function fflux(u,rxJ,ryJ)
    return (a[1]*rxJ + a[2]*ryJ) * u
end

# Contravariant flux in y-direction.
function gflux(u,sxJ,syJ)
    return (a[1]*sxJ + a[2]*syJ) * u
end

# Rotated Rusanov flux.
function riemann(uM,uP,nn) 
    return 0.5*(rflux(uM,nn) + rflux(uP,nn)) - 0.5*max_signal_speed(uM,uP,nn) * (uP - uM)
end

function calc_rhs(u,meshdata,refdata,weakDGMat,invJ)
    @unpack rxJ,sxJ,ryJ,syJ = meshdata
    @unpack mapM,mapP,nxyzJ,Jf = meshdata
    @unpack Vf,Vq = refdata
    @unpack rstxyzJ = meshdata
    
    # Transform to quadrature space.
    uq = Vq * u
  
    # Compute the volume integral.    
    R = weakDGMat[1] * fflux.(uq,Vq*rxJ,Vq*ryJ) + weakDGMat[2] * gflux.(uq,Vq*sxJ,Vq*syJ)
    
    # Prolongate to faces.
    u_faces = Vf*u    

    # Calculate interface fluxes.
    f_faces = similar(u_faces)
    for iface in each_face_node_global(refdata,meshdata)
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
u = initial_condition.(meshdata.x, meshdata.y);

# Visualize initial states.
@unpack Vp = refdata
@unpack x,y = meshdata

xp, yp, up = Vp * x, Vp * y, Vp * u

scatter(xp, yp, up, zcolor=up, msw=0, leg=false, markersize=6.5*[1.0,1.0,1.0],ratio=1.0, cam=(0, 90),size = (600, 600))
savefig("init.png")

# Temporary array for Runge-Kutta scheme.
tmp = zero(u)

FinalTime = 0.1
dt = 1e-3

Nsteps = ceil(FinalTime / dt)

# Runge-Kutta coefficients.
rk4a, rk4b, rk4c = ck45()

for i = 1:Nsteps
    # Explicit Runge-Kutta.
    for istage = 1:5
        rhs = calc_rhs(u,meshdata,refdata,weakDGMat,invJ)
        @. tmp = rk4a[istage] * tmp + dt * rhs
        @. u =  u + rk4b[istage] * tmp
    end
    
    # # Explicit Euler.
    # rhs = calc_rhs(u,meshdata,refdata,weakDGMat,invJ)
    # @. u = u + dt*rhs
    
    if i % 10 == 0
        print(i,",")
    end
end

@unpack Vp = refdata
@unpack x,y = meshdata

xp, yp, up = Vp * x, Vp * y, Vp * u

scatter(xp, yp, up, zcolor=up, msw=0, leg=false, markersize=2.5*[1.0,1.0,1.0],ratio=1.0, cam=(0, 90),size = (600, 600))
savefig("final.png")