# Implementation of the DG Collocation Spectral Element Method (DGSEM) for the linear advection equation d/dt u + a d/dx u = 0

#Import StartUpDG: a library for nodal approaches based on: "Nodal Dicountinuous Galerkin Methods" by Hesthaven and Wartburton
using StartUpDG
using LinearAlgebra

include("runge-kutta.jl")
include("supplements.jl")

function adv_func(advection_coeff, u)
    return advection_coeff*u
end

function lax_friedrich_flux(u_plus, u_minus,a)
    return (adv_func(a, u_plus) + adv_func(a, u_minus)) ./2 - abs(a)/2 .*(u_plus - u_minus)
end

function flux_update(u_flux, u, advection_coeff, num_elem)

    u_flux[1,1] = lax_friedrich_flux(u[1,1], u[end,num_elem], advection_coeff)
    u_flux[end,1] = lax_friedrich_flux(u[1,2], u[end,1], advection_coeff)

    for ielem = 2:num_elem-1
        u_flux[1, ielem] = lax_friedrich_flux(u[1,ielem], u[end,ielem-1], advection_coeff)
        u_flux[end, ielem] = lax_friedrich_flux(u[1, ielem+1], u[end, ielem], advection_coeff)
    end
    
    u_flux[1, num_elem] = lax_friedrich_flux(u[1, num_elem], u[end, num_elem-1], advection_coeff)
    u_flux[end,num_elem] = lax_friedrich_flux(u[1,1], u[end, num_elem],advection_coeff)

    return u_flux
end

function Right_side_ODE(u, u_flux, term_surface, term_volume, transform, advection_coeff) 

    num_elem = size(u)[2]
    u_flux = flux_update(u_flux, u, advection_coeff, num_elem) 
    right = zero(u)
    for i in 1:num_elem
        right[:,i] = transform[:,i] .* (term_surface * u_flux[:,i] .+ advection_coeff .*term_volume * u[:,i])
    end

    return right
end

function DGSEM(deg_p, mesh_size, init_function, max_time, advection_coeff)

    # Init ref_element with Lobatto-Gauss-Quadrature and mesh 
    ref_elem = RefElemData(Line(), deg_p; quad_rule_vol = gauss_lobatto_quad(0,0,deg_p));
    VXY, EToV = uniform_mesh(Line(),mesh_size);
    mesh_data = MeshData(VXY,EToV, ref_elem);
    num_elem = mesh_data.num_elements;

    # Use periodic boundary conditions
    mesh_data = make_periodic(mesh_data)

    # Extract the LGL-mass matrix from the ref_elem
    Mass_Matrix = ref_elem.M;
    Mass_Matrix_inv = Base.inv(Mass_Matrix);

    # Extract differention matrix from the ref_elem
    Diff_Matrix = ref_elem.Drst[1];

    # Define B-Matrix
    B = LinearAlgebra.diagm([-1; zeros(deg_p-1);1]);

    # Extract knots and faces of dim 1: faces[iface, ielem]
    faces = mesh_data.xyzf[1];
    x = mesh_data.xyz[1];
    dx = abs.(faces[2,:] - faces[1,:])

    # Transformation matrix which projects from the ref element to mesh
    invJ = inv.(mesh_data.J)

    # Compute dt 
    if advection_coeff != 0
        xmin =minimum(dx)
        CFL = 0.1
        dt = CFL*xmin/abs(advection_coeff)
    else 
        dt = 2*10^(-10)
    end

    # Start conditions 
    u = init_function.(x)

    # Start element_dependency

    u_flux = zeros(deg_p+1,num_elem)
    # Calculation of the numerical flux with start conditions
    for ielem = 1:num_elem
        u_flux[:,ielem] = [u[1,ielem]; zeros(deg_p-1);u[end, ielem]];
    end

    term_surface = .- Mass_Matrix_inv * B;
    term_volume = Mass_Matrix_inv * Diff_Matrix' * Mass_Matrix;

    time = range(0.0,step = dt, max_time-dt)

    u = runge_kutta_45(u, u_flux, time, dt, term_surface, term_volume, invJ, advection_coeff, Right_side_ODE)

    return [u,x]
end