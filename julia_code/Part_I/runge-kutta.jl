# Implementation of the runge kutta algorithm of order four: d/dt u = R(u(t),t)
using StartUpDG

# Function for runge_kutta algorithm of order four: d/dt u = R(u(t),t)
# with start_time = t, start_value from u = u and Linear Operator A, describing R(u(t))
function runge_kutta_45_one(u, u_flux, surface, volume, dx, a, t, dt, R,tmp)
    # Coefficients for the explicit five steps Runge-Kutta method with order four
    # The rows i=1,...,5 contain the coefficants using startUpDG
    rk4a, rk4b, rk4c = ck45()
    for istage = 1:5
        g = R(u, u_flux, surface, volume, dx, a)
        tmp = rk4a[istage] .* tmp + dt .* g
        u = u + rk4b[istage] .* tmp
    end
    return [u,tmp]
end

# Apply the Runge-Kutta method for a specific number of timesteps
function runge_kutta_45(u, u_flux, time, dt, surface, volume, dx, a, R)
    tmp = zero(u)
    for t in time
        sol = runge_kutta_45_one(u, u_flux, surface, volume, dx, a, t, dt, R,tmp)
        u = sol[1]
        tmp = sol[2]
    end
    return u
end