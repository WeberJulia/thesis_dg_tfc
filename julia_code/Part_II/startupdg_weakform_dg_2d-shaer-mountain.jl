include("terrainFollowingCoordinates.jl")
using StartUpDG, UnPack
using StaticArrays
using LinearAlgebra
using Plots
cd(dirname(@__FILE__()))
path_figures = dirname(@__FILE__())*"/Pictures/"
path_sim = dirname(@__FILE__())*"/Pictures/Simulation/"
title_save = "lin-advection-bubble"
title = "Linear advection bubble"

NDIMS = 2;
NPOLY = 3;
NNODE = NPOLY+1;
N_hc = 50;
N_vc = 25;
# test Identity
# Lx = 2*N_hc;
# H_top = 2*N_vc;

Lx = 150;
H_top = 25;

@inline function orography_zero(x)
    return zero.(x)
end
@inline function orography_sin(x)
        # b = 2*pi / Lx
        b = 2*pi /4
        return sin.(x .*b) .+ 1
end
@inline function orography_paper(x)
    # H0*exp(-x^2/(a_e^2))*cos(pi * x/a_c)^2
    return 10*exp(-x^2/25)*cos(pi * x/4)^2
end

@inline function h_star(x, ah, h0)
    return h0 * cos(pi * x /(2*ah))^2
end

@inline function shaer_mountain(x)
    # Lx1 = 300     Abb. 150
    # H_top = 25    Abb. 15
    h0 = 3
    ah = 25
    l = 8
    xi = x-Lx/2
    if(abs(xi)< ah)
        return cos(pi * xi /l)^2 * h_star(xi, ah, h0)
    else 
        return 0
    end
end

nodes, weights = StartUpDG.gauss_lobatto_quad(0, 0, NPOLY);

etype = Quad()
refdata = RefElemData(etype, Polynomial(), NPOLY)

# Build DG operators.
# volume quadrature weights, volume interpolation matrix, mass matrix, differentiation matrix
@unpack wq, Vq, M, Drst = refdata

weakDGMat = map(D -> M \ ((Vq * D)' * Diagonal(wq)), Drst);

# Preparing mesh.
meshes = Meshes(shaer_mountain, Lx, H_top, [N_hc, N_vc])
@unpack cells_tfc, conn = meshes

# Building mesh data: connectivity, Jacobians, ...
meshdata_tfc = make_periodic(MeshData(cells_tfc, conn, refdata))

# As the solution is given in the Cartesian system, the initial function holds in the Cartesian system too.
meshdata_c = get_meshdata_cartesian(shaer_mountain, meshes, meshdata_tfc)

meshdata_c = MeshData(refdata, meshdata_tfc, meshdata_c[1], meshdata_c[2])

# Get transformationparameters of the meshdata, at each interpolation point the DG method has to solve another pde
transformation =  transformation_parameters(shaer_mountain, meshes, meshdata_tfc, meshdata_c)

meshplotter = MeshPlotter(refdata,meshdata_tfc)
plot(meshplotter, title = "tfc-mesh")
savefig(path_figures*"tfc-mesh.png")

# Helper routines for iterating.
@inline each_dof_global(rd,md) = Base.OneTo(rd.Np * md.num_elements)
@inline each_face_node_global(rd,md) = Base.OneTo(rd.Nfq * md.num_elements)

# Inverse Jacobian.
invJ = inv.(meshdata_tfc.J);
invJ_test = inv.(meshdata_c.J);

@inline function initial_condition(x,y; xc = nothing, yc = nothing #= xc,yc: element centers =#)
    # 2D Gaussian bell
    return exp(-((x-2.0)^2 + (y-2.0)^2))
end

@inline function advection_shaer_mountain(x,y)
    u0 = 10
    z1 = 4
    z2 = 5
    if(z2 <= y)
        return u0
    elseif(z1 <= y)
        u0 * sin((pi/2)*(y-z1)/(z2-z1))^2 
    else
        return 0
    end
end

@inline function initial_shaer_mountain(x,z)
    xi = x-Lx/2
    xr = xi + 50
    zr = z - 9
    r = sqrt((xr/25)^2+(zr/3)^2)
    if(r <= 1)
        return cos(pi*r/2)^2
    else
        return 0
    end
end

# Fill initial condition
u_c = Float64.(initial_shaer_mountain.(meshdata_c.x, meshdata_c.y));
u_test = u_c
#u_init = u_c;

# Simple advection scheme.
a = [10,0.0] # advection speeds only x-direction
a_test = [Float64.(advection_shaer_mountain.(meshdata_c.x, meshdata_c.y)), zero(meshdata_c.y)]

# Transformation of the cartesian function values to the tfc values in x-direction
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
af_test = [Float64.(advection_shaer_mountain.(meshdata_c.xf, meshdata_c.yf)), zero(meshdata_c.yf)]


# Rotated max. signal speed.
@inline function max_signal_speed(am, ap,uM,uP,nn)
    return maximum([norm(am .* nn), norm(ap .* nn)])
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
@inline function riemann(am, ap, uM,uP,nn)
    return 0.5*(rflux(am, uM,nn) + rflux(ap, uP,nn)) - 0.5*max_signal_speed(am, ap,uM,uP,nn) * (uP - uM)
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
        am = SVector{NDIMS}(getindex.(af, iM))
        ap = SVector{NDIMS}(getindex.(af, iP))

        f_faces[iface] = riemann(am, ap,uM,uP,nn)
    end
    
    # Add surface terms.
    @unpack LIFT = refdata

    R -= LIFT * f_faces

    # Scale to mesh size.
    return R .* invJ ./ g_scalar_sqrt
end

function calc_rhs(u,meshdata,refdata,weakDGMat,invJ)
    @unpack rxJ,sxJ,ryJ,syJ  = meshdata
    @unpack mapM,mapP,nxyzJ,Jf = meshdata
    @unpack Vf,Vq = refdata
    @unpack rstxyzJ = meshdata
    
    # Transform to quadrature space.
    uq = Vq * u
  
    # Compute the volume integral.    
    R = weakDGMat[1] * fflux.(uq,Vq*(a_test[1] .*rxJ),Vq*(a_test[2] .*ryJ)) + weakDGMat[2] * gflux.(uq,Vq*(a_test[1] .*sxJ),Vq*(a_test[2] .* syJ))

    # Prolongate to faces.
    u_faces = Vf*u    

    # Calculate interface fluxes.
    f_faces = similar(u_faces)
    Threads.@threads for iface in each_face_node_global(refdata,meshdata)
        iM, iP = mapM[iface], mapP[iface]
        uM = u_faces[iM]
        uP = u_faces[iP]
        nn = SVector{NDIMS}(getindex.(nxyzJ, iM)) # surface normal
        am = SVector{NDIMS}(getindex.(af_test, iM))
        ap = SVector{NDIMS}(getindex.(af_test, iP))
        
        f_faces[iface] = riemann(am, ap, uM,uP,nn)
    end
    #vtu_sim = MeshData_to_vtk(meshdata, refdata, [f_faces], [title], path_sim*string(title_save,"_faces",Int32(jtest)), true)
    

    # Add surface terms.
    @unpack LIFT = refdata
    R -= LIFT * f_faces

    # Scale to mesh size.
    return R .* invJ
end

meshplotter = MeshPlotter(refdata,meshdata_c)
plot(meshplotter, title = "cartesian-mesh")
savefig(path_figures*"cartesian-mesh.png")

# Visualize initial states.
@unpack Vp = refdata

# Interpolation to plotting points
xp, yp, up = Vp * meshdata_c.x, Vp * meshdata_c.y, Vp * u_c

#scatter(xp, yp, up, zcolor=up, msw=0, leg=false, markersize=6.5*[1.0,1.0,1.0],ratio=1.0, cam=(0, 90),size = (600, 600))
#display(scatter(xp, yp, up, zcolor=up, msw=0, leg=false,ratio=1.0, cam=(0, 90),size = (600, 600), title = "init"))
#sleep(5)
#savefig(path_figures*string(title_save, "-init.png"))
vtu_name = MeshData_to_vtk(meshdata_c, refdata, [u_c], [title], path_figures*string(title_save,"-init"), true)

@inline function min_rd(rd)
    min = 2
    for i in 1:NPOLY
        imin = abs.(rd.r[i]-rd.r[i+1])
        if (imin < min)
            min = imin 
        end
    end
    return min
end

# Temporary array for Runge-Kutta scheme.
tmp = zero(u_c)
tmp_test = zero(u_test)

FinalTime = 5.0
halfTime = 2.5
dt = 0.025

Nsteps = ceil(FinalTime / dt)
Nhalf = ceil(halfTime / dt)

# Runge-Kutta coefficients.
rk4a, rk4b, rk4c = ck45()

for i = 1:Nsteps
    # Explicit Runge-Kutta.
    for istage = 1:5
        rhs = calc_rhs(u_c,meshdata_tfc,refdata,weakDGMat,invJ, transformation)
        @. tmp = rk4a[istage] * tmp + dt * rhs
        @. u_c =  u_c + rk4b[istage] * tmp

        rhs_test = calc_rhs(u_test,meshdata_c,refdata,weakDGMat,invJ_test)
        @. tmp_test = rk4a[istage] * tmp_test + dt * rhs_test
        @. u_test =  u_test + rk4b[istage] * tmp_test
    end
    if(i == Nhalf)
        vtu_name_half = MeshData_to_vtk(meshdata_c, refdata, [u_test], [title], path_figures*string(title_save,"-test-half"), true)
        vtu_name_half = MeshData_to_vtk(meshdata_c, refdata, [u_c], [title], path_figures*string(title_save,"-half"), true)
    end
end

# println("Test Identity: Exception: init - final = 0")
# print_line(u_init .- u_c)

vtu_name = MeshData_to_vtk(meshdata_c, refdata, [u_c], [title], path_figures*string(title_save,"-final"), true)
vtu_name = MeshData_to_vtk(meshdata_c, refdata, [u_test], [title], path_figures*string(title_save,"-test-final"), true)


println(string("finished ",title))