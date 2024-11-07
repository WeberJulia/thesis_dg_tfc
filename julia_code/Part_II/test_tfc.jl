include("terrainFollowingCoordinates.jl")
using Plots
using DataFrames
using StartUpDG, UnPack
using StaticArrays
using LinearAlgebra

function orography_zero(x,y) 
    return zero.(x)
end
function orography_zero(x) 
    return zero.(x[1])
end

function tfc_mesh_new(N_hc, N_vc)

    cells, conn = uniform_mesh(Quad(), N_hc, N_vc)
    cells[1] .= (cells[1] .+ 1) 
    cells[2] .= (cells[2] .+ 1) 

    cells[1] .= cells[1] .*(N_hc)
    cells[2] .= cells[2] .*(N_vc)

    return (cells, conn)
end

function test_tfc_mesh_Quad()
    NPOLY = 3;
    etype = Quad();
    refdata = RefElemData(etype, Polynomial(), NPOLY);
    N_cell = 50

    for N_hc in 1:N_cell
        for N_vc in 1:N_cell
            cells, conn = tfc_mesh(N_hc, N_vc);
            cells_new, conn_new = tfc_mesh_new(N_hc, N_vc);
            if( conn_new != conn)
                println(string("conns are not equal: ",N_hc1," ",N_hc2," ",N_vc))
            end
            for i in 1:2
                if( round.(cells_new[i]) != cells[i])
                    println(string("cells[",i,"] are not equal: ",N_hc," ",N_vc))
                    fprint(round.(cells[i]), string("cells",i))
                    fprint(round.(cells_new[i]), string("cells_new",i))
                end
            end
        end
    end
    println("finished test_tfc_mesh_Quad")
end

function tfc_mesh_new(N_hc1, N_hc2, N_vc)

    cells, conn = uniform_mesh(Hex(), N_hc1, N_hc2, N_vc)
    cells[1] .= (cells[1] .+ 1) 
    cells[2] .= (cells[2] .+ 1) 
    cells[3] .= (cells[3] .+ 1) 

    cells[1] .= cells[1] .*(N_hc1)
    cells[2] .= cells[2] .*(N_hc2)
    cells[3] .= cells[3] .*(N_vc)

    return (cells, conn)
end

function test_tfc_mesh_Hex()
    NPOLY = 3;
    etype = Hex();
    refdata = RefElemData(etype, Polynomial(), NPOLY);
    N_cell = 3

    for N_hc1 in 1:N_cell
        for N_hc2 in 1:N_cell
            for N_vc in 1:N_cell
                cells, conn = tfc_mesh(N_hc1, N_hc2, N_vc);
                cells_new, conn_new = tfc_mesh_new(N_hc1, N_hc2, N_vc);
                if( conn_new != conn)
                    println(string("conns are not equal: ",N_hc1," ",N_hc2," ",N_vc))
                end
                for i in 1:3
                    if( round.(cells_new[i]) != cells[i])
                    println(string("cells[",i,"] are not equal: ",N_hc1," ",N_hc2," ",N_vc))
                    fprint(round.(cells[i]), string("cells",i))
                    fprint(round.(cells_new[i]), string("cells_new",i))
                    end
                end
            end
        end
    end
    println("finished test_tfc_mesh_Hex")
end

@inline function check_ones(matrix, name)
    if(matrix != ones(size(matrix)))
        println(string(name, "is not one."))
    end
end

@inline function check_zeros(matrix, name)
    if(matrix != zero(matrix))
        println(string(name, "is not zero."))
    end
end

@inline function check(check_function,transformation, indizes)
    @unpack g_cov, g_contra = transformation
    @unpack alpha, beta = transformation

    for i in indizes
        check_function(g_cov[i], string("g_cov ",i))
        check_function(g_contra[i], string("g_contra ",i))
        check_function(alpha[i], string("alpha ",i))
        check_function(beta[i], string("beta ",i))
    end
end

function test_metric_tensors_zero_Quad()

    NPOLY = 2;
    etype = Quad()
    refdata = RefElemData(etype, Polynomial(), NPOLY)

    for NCELL = 1:5
        Lx = 2*NCELL;
        H_top = 2*NCELL;

        # Preparing mesh and meshdata
        meshes = Meshes(orography_zero, Lx, H_top, [NCELL, NCELL])
        @unpack cells_tfc, conn = meshes
        meshdata_tfc = make_periodic(MeshData(cells_tfc, conn, refdata))
        meshdata_c = get_meshdata_cartesian(orography_zero, meshes, meshdata_tfc)
        meshdata_c = MeshData(refdata, meshdata_tfc, meshdata_c[1], meshdata_c[2])

        # #get transformationparameters of the meshdata, at each interpolation point the DG method has to solve another pde
        transformation =  transformation_parameters(orography_zero, meshes, meshdata_tfc,meshdata_c);
        @unpack g_scalar = transformation
        check_ones(g_scalar, "g_scalar")
        
        check(check_ones,transformation, [1,4])
        check(check_zeros,transformation, [2,3])
    end

    println("finished test_metric_tensors_zero_Quad")
end

function test_metric_tensors_zero_Hex()

    NPOLY = 2;
    etype = Hex()
    refdata = RefElemData(etype, Polynomial(), NPOLY)

    for NCELL = 1:5
        Lx1 = 2*NCELL;
        Lx2 = 2*NCELL;
        H_top = 2*NCELL;

        # Preparing mesh and meshdata
        meshes = Meshes(orography_zero, [Lx1,Lx2], H_top, [NCELL, NCELL,NCELL])
        @unpack cells_tfc, conn = meshes
        meshdata_tfc = make_periodic(MeshData(cells_tfc, conn, refdata))
        meshdata_c = get_meshdata_cartesian(orography_zero,meshes,meshdata_tfc)
        meshdata_c = MeshData(refdata, meshdata_tfc, meshdata_c[1], meshdata_c[2], meshdata_c[3])

        # #get transformationparameters of the meshdata, at each interpolation point the DG method has to solve another pde
        transformation =  transformation_parameters(orography_zero, meshes, meshdata_tfc, meshdata_c);
        @unpack g_scalar = transformation
        check_ones(g_scalar, "g_scalar")
        
        check(check_ones,transformation, [1,5,9])
        check(check_zeros,transformation, [2,3,4,6,7,8])
    end

    println("finished test_metric_tensors_zero_Hex")
end

function main()
    test_tfc_mesh_Quad()
    test_metric_tensors_zero_Quad()
    test_tfc_mesh_Hex()
    test_metric_tensors_zero_Hex()
end

main()