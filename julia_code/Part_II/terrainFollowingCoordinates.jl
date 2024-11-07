# # based on the paper: A horizontally explicit, vertically implicit (HEVI) discontinuous Galerkin scheme 
# for the 2-dimensional Euler and Navier-Stokes equations using terrain-following coordinates (M. Baldauf)
# The terrain-following parameters are stored with the abbreviation tfc and the Cartesian ones are stored with c
include("terrainFollowingCoordinates_3D.jl")
include("terrainFollowingCoordinates_2D.jl")

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

# Struct which contains all transformation paramteters, that may be important for the transformation of the PDE
struct Transformation_parameters{G_cov, G_contra, G_scalar, G_scalar_sqrt, Alpha, Beta, G_covf, G_contraf, G_scalarf, G_scalarf_sqrt, Alphaf, Betaf}
    g_cov::G_cov
    g_contra::G_contra
    g_scalar::G_scalar
    g_scalar_sqrt::G_scalar_sqrt
    alpha::Alpha
    beta::Beta
    g_covf::G_covf
    g_contraf::G_contraf
    g_scalarf::G_scalarf
    g_scalarf_sqrt::G_scalarf_sqrt
    alphaf::Alphaf
    betaf::Betaf
end

# ersetzt 2D function - in main part integrieren
# Easier access to components of transformation_parameters.beta 
@inline function transformation_getproperty(x::Transformation_parameters, s::Symbol)
    if  s==:rxB
        return getfield(x, :beta)[1,1]
    elseif s==:sxB
        return getfield(x, :beta)[1,2]
    elseif s==:txB
        return getfield(x, :beta)[1,3]
    elseif s==:ryB
        return getfield(x, :beta)[2,1]
    elseif s==:syB
        return getfield(x, :beta)[2,2]
    elseif s==:tyB
        return getfield(x, :beta)[2,3]
    elseif s==:rzB
        return getfield(x, :beta)[3,1]
    elseif s==:szB
        return getfield(x, :beta)[3,2]
    elseif s==:tzB
        return getfield(x, :beta)[3,3]
    elseif  s==:rxBf
        return getfield(x, :betaf)[1,1]
    elseif s==:sxBf
        return getfield(x, :betaf)[1,2]
    elseif s==:txBf
        return getfield(x, :betaf)[1,3]
    elseif s==:ryBf
        return getfield(x, :betaf)[2,1]
    elseif s==:syBf
        return getfield(x, :betaf)[2,2]
    elseif s==:tyBf
        return getfield(x, :betaf)[2,3]
    elseif s==:rzBf
        return getfield(x, :betaf)[3,1]
    elseif s==:szBf
        return getfield(x, :betaf)[3,2]
    elseif s==:tzBf
        return getfield(x, :betaf)[3,3]
    else
        return getfield(x, s)
    end
end

# in main
Base.getproperty(x::Transformation_parameters, s::Symbol) = transformation_getproperty(x, s)


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

""" Meshes calculates the x and y values of the nodes of the Cartesian and the tfc mesh.
The struct stores the cells of the meshes and their connectivity with important scaling parameters.
The connectivity does not change by transforming the Cartesian mesh to the tfc mesh. 
Contains: a1, a3, H_top, cells_tfc, cells_c, conn, num_elem """
function Meshes(orography, Lx, H_top, NCELL; stretching = stretching_equidistant)
    dim = length(NCELL);

    if(dim == 2)
        return Meshes_2D(orography, Lx, H_top, NCELL[1], NCELL[2], dim; stretching = stretching)
    end
    if(dim == 3)
        return Meshes_3D(orography, Lx[1], Lx[2], H_top, NCELL[1], NCELL[2], NCELL[3], dim; stretching = stretching)
    end
end

""" The DG method needs interpolation/ quardature points on each element.
 It is necessary to calculate the corresponding points on the Cartesain mesh."""
 @inline function get_meshdata_cartesian(orography, meshes, meshdata_tfc; stretching = stretching_equidistant)
    @unpack dim = meshes

    if(dim == 2)
        @unpack a1, a3, H_top = meshes

        x_c = a1 .* meshdata_tfc.x
        y_c = cart_vertical(orography, meshdata_tfc.x, meshdata_tfc.y, [a1, a3], H_top, stretching = stretching)
        return (x_c,y_c)
    end
    if(dim == 3)
        @unpack a1, a2, a3, H_top = meshes

        x_c = a1 .* meshdata_tfc.x
        y_c = a2 .* meshdata_tfc.y
        z_c = cart_vertical(orography, meshdata_tfc.x, meshdata_tfc.y, meshdata_tfc.z,[a1, a2, a3], H_top, stretching = stretching)
        return (x_c, y_c, z_c)
    end
end

@inline function metric_scalar_density(g_cov,dim)
    if(dim == 2)
        return metric_scalar_density_2d.(g_cov[1], g_cov[2], g_cov[3], g_cov[4])
    end
    if(dim == 3)
        return metric_scalar_density_3d.(g_cov[1], g_cov[2], g_cov[3], g_cov[4],g_cov[5],g_cov[6], g_cov[7],g_cov[8],g_cov[9])
    end
end

""" Because of the distortion of the tfc-mesh we need to calculate the metric tensors of the transformation,
which depends on the orography. They are used to formulate the PDE.
The metric variables get calculated at each node/ interpolation point.
Contains:  g_cov, g_contra, g_scalar, g_scalar_sqrt, alpha, beta for volumes and faces"""
function transformation_parameters(orography, meshes, meshdata_tfc, meshdata_c; stretching = stretching_equidistant)
    
    @unpack dim = meshes
    g_cov, g_contra, g_scalar, g_scalar_sqrt, alpha, beta = 0,0,0,0,0,0

    if(dim == 2)
        @unpack a1 = meshes
        
        alpha = alpha_values_2d(orography, meshes, meshdata_tfc, meshdata_c, stretching = stretching)
        alphaf = alpha_values_2d(orography, meshes, meshdata_tfc, meshdata_c, stretching = stretching, face = true)
        
        g_cov = covariant_metric_tensor_2d(a1, alpha)
        g_covf = covariant_metric_tensor_2d(a1, alphaf)
        
        g_contra = contravariant_metric_tensor_2d(g_cov)
        g_contraf = contravariant_metric_tensor_2d(g_covf)

        beta = beta_values_2d(meshes, alpha, g_contra)
        betaf = beta_values_2d(meshes, alphaf, g_contraf)
    end

    if(dim == 3)
        @unpack a1, a2 = meshes

        #meshdata_c = get_meshdata_cartesian(orography, meshes, meshdata_tfc)
        
        alpha = alpha_values_3d(orography, meshes, meshdata_tfc, meshdata_c, stretching = stretching)
        alphaf = alpha_values_3d(orography, meshes, meshdata_tfc, meshdata_c, stretching = stretching, face = true)
        
        g_cov = covariant_metric_tensor_3d(a1, a2, alpha)
        g_covf = covariant_metric_tensor_3d(a1, a2, alphaf)
        
        g_contra = contravariant_metric_tensor_3d(g_cov)
        g_contraf = contravariant_metric_tensor_3d(g_covf)

        beta = beta_values_3d(meshes, alpha, g_contra)
        betaf = beta_values_3d(meshes, alphaf, g_contraf)

    end

    g_scalar = metric_scalar_density(g_cov, dim)
    g_scalar_sqrt = sqrt.(g_scalar)
    g_scalarf = metric_scalar_density(g_covf, dim)
    g_scalarf_sqrt = sqrt.(g_scalarf)

    variables = Transformation_parameters(g_cov, g_contra, g_scalar, g_scalar_sqrt, alpha, beta, g_covf, g_contraf, g_scalarf, g_scalarf_sqrt, alphaf, betaf)
end