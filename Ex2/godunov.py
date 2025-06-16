import numpy as np

def U2W(U, gamma ):
    """
    This function converts the primitive variables U = [rho, u, p] to the
    conserved variables W = [rho, rho*u, E] for a 2D array grid U, where gamma is
    the ratio of specific heats.
    """
    W = np.zeros_like(U)

    for i in range(len(U)):
        rho = U[i,0]
        u = U[i,1]
        p = U[i,2]

        W[i,0] = rho # Density
        W[i,1] = rho * u # Momentum
        W[i,2] = p / (gamma - 1) + 0.5 * rho * u ** 2 # Energy
        if W[i,2] < 0:
            print("Negative energy detected")
            W[i,2] = 0
    
    return W

def W2U(W, gamma):
    """
    This function converts the conserved variables W = [rho, rho*u, E] to the
    primitive variables U = [rho, u, p] for a 2D array W, where gamma is
    the ratio of specific heats.
    """
    U = np.zeros_like(W)

    for i in range(len(W)):
        rho = W[i,0]
        rho_u = W[i,1]
        E = W[i,2]

        u = rho_u / rho
        p = (gamma - 1) * (E - 0.5 * rho * u ** 2)
        if p < 0:
            print("Negative pressure detected")
            p = 0

        U[i,0] = rho
        U[i,1] = u
        U[i,2] = p
    
    return U

def godunov_flux(U1, U4, gm1, gm4, eps = 1e-3):


    """
    Godunov's method for the Euler equations of gas dynamics.
    This function computes the fluxes at the interface between two cells.

    It uses the Riemann solver to compute the fluxes at the interface.
    The Riemann solver is based on the method of characteristics.

    - U1 and U4 are the primitive variables [rho, u, p] at the left and right states.
    - gm1 and gm4 are the ratio of specific heats c_p/c_v for the left and right states.
    - eps is the tolerance for the Newton-Raphson iteration.
    """

    rho1, u1, p1 = U1
    rho4, u4, p4 = U4
    
    d1 = 0.5*(gm1 - 1)
    d4 = 0.5*(gm4 - 1)

    k1 = 0.5*(gm1 + 1)
    k4 = 0.5*(gm4 + 1)
    beta1 = gm1/d1
    beta4 = gm4/d4

    a1 = np.sqrt(gm1*p1/rho1)
    a4 = np.sqrt(gm4*p4/rho4)

    # Mean properties
    gam = 0.5*(gm1 + gm4)
    dm = 0.5*(gam - 1)
    alfam = dm/gam



    # Compute the wave speeds - first guess
    z = (a4/a1)*((p1/p4)**alfam)
    vel = (z*(a1/dm + u1) - (a4/dm + u4))/(z + 1)


    # Newton-Raphson iteration
    err = 1
    while err > eps:
        # u - a wave
        if vel <= u1:
            # Shock
            x1 = k1*(u1 - vel)/(2*a1)
            M1r = x1 + np.sqrt(x1**2 + 1)
            M1rq = M1r**2
            p2 = p1*(1 + gm1*(M1rq - 1)/k1)
            dp2 = -2*gm1*p1*M1r/a1/(1/M1rq + 1)
        else:
            # Rarefaction
            a2 = a1 - d1*(vel - u1)
            p2 = p1*(a2/a1)**beta1
            dp2 = -gm1*p2/a2
        
        # u + a wave
        if vel >= u4:
            # Shock
            x4 = k4*(u4 - vel)/(2*a4)
            M4r = x4 - np.sqrt(x4**2 + 1)
            M4rq = M4r**2
            p3 = p4*(1 + gm4*(M4rq - 1)/k4)
            dp3 = -2*gm4*p4*M4r/a4/(1/M4rq + 1)

        else:
            # Rarefaction
            a3 = a4 + d4*(vel - u4)
            p3 = p4*(a3/a4)**beta4
            dp3 = gm4*p3/a3

        # Compute the residual
        err = np.abs(1 - p2/p3)
        if err < eps:
            break
        # Update the guess
        vel = vel - (p2 - p3)/(dp2 - dp3)
    
    # Interface variables

    # Il caso v = 0 non è considerato e manda in errore

    if vel > 0:
        gmf = gm1
        uf = u1
        pf = p1
        rhof = rho1

        if vel < u1:
            # Shock
            w2 = u1 - a1*M1r
            if w2 < 0:
                pf = p2
                uf = vel
                rhof = rho1*(k1*(d1 + 1/M1rq))
        else:
            # vel > u1, rarefaction
            lmbd1 = u1 - a1
            if lmbd1 < 0:
                pf = p2
                uf = vel
                rhof = -dp2/a2
                
                lmbd2 = vel - a2
                if lmbd2 > 0:
                    af = (a1 + d1*u1)/(1 + d1)
                    uf = af
                    pf = p1*(af/a1)**beta1
                    rhof = gm1*pf/af**2
    else:
        # vel < 0
        gmf = gm4
        uf = u4
        pf = p4
        rhof = rho4

        if vel > u4:
            # Shock 
            w3 = u4 - a4*M4r
            if w3 > 0:
                pf = p3
                uf = vel
                rhof = rho4*(k4/(d4 + 1/M4rq))
        else:
            # vel < u4, rarefaction
            lmbd4 = u4 + a4
            if lmbd4 > 0:
                pf = p3
                uf = vel
                rhof = dp3/a3

                lmbd3 = vel + a3
                if lmbd3 < 0:
                    af = (a4 - d4*u4)/(1 + d4)
                    uf = -af
                    pf = p4*(af/a4)**beta4
                    rhof = gm4*pf/af**2
    # Compute the fluxes
    fm = rhof*uf # rho*u
    fqdm = pf + fm*uf # rho*u^2 + p

    # E = p/(gm - 1) + 0.5*rho*u^2
    e = pf/(gmf - 1) + 0.5*rhof*uf**2
    fe = uf*(e + pf) # u*(E + p)

    Uf = np.array([rhof, uf, pf])
    Ff = np.array([fm, fqdm, fe])

    return Uf, Ff
            

def resize_array_mean(array, dt_array, T):
    """
    Perform a heuristic to resize the array to store more timesteps
    Compute the mean of the timesteps until now
    """
    mean_dt = np.mean(dt_array)
    # Compute the number of timesteps left to reach T
    n_steps_left = int(np.ceil((T - np.sum(dt_array)) / mean_dt))
    # Resize the array to store more timesteps
    array = np.resize(array, (array.shape[0] + n_steps_left, array.shape[1], array.shape[2]))

    return array


def godunov_step(U_grid, dx, c, gamma = 1.4, eps = 1e-3):
    """
    Godunov's method for the Euler equations of gas dynamics.
    This function computes the fluxes at the interface between two cells.

    It uses the Riemann solver to compute the fluxes at the interface.
    The Riemann solver is based on the method of characteristics.

     - U_grid is a 2D array of shape (n_samples, 3), where each row is a sample
     of the primitive variables [rho, u, p]
     - dx is the grid spacing
     - c is the Courant number
     - gamma is the ratio of specific heats c_p/c_v


     - U = [rho, u, p]
     - W = [rho, rho*u, E]
     - F = [rho*u, rho*u^2 + p, u*(E + p)]
    """
    n_samples = len(U_grid)
    # Add ghost cells to the left and right of the grid
    U_grid_gc = np.concatenate((U_grid[1].reshape(1, -1), U_grid, U_grid[-1].reshape(1, -1)), axis = 0)
    # Convert to conservative variables
    W_grid_gc = U2W(U_grid_gc, gamma = gamma) 
    # Compute the sound speeds
    a_grid_gc = np.sqrt(gamma * U_grid_gc[:,2] / U_grid_gc[:,0]) # a = sqrt(gamma * p / rho)

    Lp = abs(U_grid_gc[:,1]) + a_grid_gc # Eigenvalues on u + a
    Lm = abs(U_grid_gc[:,1]) - a_grid_gc # Eigenvalues on u - a

    Lmax = np.maximum(np.max(Lp), np.max(Lm))
    # Compute the timestep
    dt = c * dx / Lmax

    # Compute the fluxes
    F_grid = np.zeros_like(U_grid)


    # Populate the fluxes


    for i in range(n_samples):
        U1 = U_grid_gc[i]
        U4 = U_grid_gc[i + 1]

        gm1 = gamma
        gm4 = gamma

        _, Ff = godunov_flux(U1, U4, gm1, gm4, eps = eps)
        F_grid[i] = Ff
    # Compute the step
    
    W_grid_dt = W_grid_gc[1:-1] - dt/dx * (F_grid[:] - F_grid[i])
    U_grid_dt = W2U(W_grid_dt, gamma = gamma)
    a_grid_dt = np.sqrt(gamma * U_grid_dt[:,2] / U_grid_dt[:,0]) # a = sqrt(gamma * p / rho)
    
    return U_grid_dt, dt

        # Appl
    

def Godunov(U0, dx, c, T, gamma = 1.4):
    """
    This function performs the Lax-Friedrichs scheme on a 2D array U0 of
    initial conditions, with grid spacing dx, Courant number c, and final time T.
    The scheme is run until the final time T is reached, and the solution is
    returned as a 3D array of shape (n_steps, n_samples, 3), where n_steps is the
    number of timesteps, and n_samples is the number of samples in U0.
    The primitive variables are [rho, u, p].

    - U0 is a 2D array of shape (n_samples, 3), where each row is a sample
                                  of the primitive variables [rho, u, p]
    - dx is the grid spacing
    - c is the Courant number
    - T is the simulation time
    - gamma is the ratio of specific heats c_p/c_v
    """
    U_grid = U0.copy()
    grid_samples = len(U_grid)

    t_total = 0
    
    guessed_steps = 0
    
    ## Due to the timestep being variable, an initial allocation is 
    #   attempted with a guessed timestep from the initial conditions.
    #   If the array fills, a mean of the timesteps is performed and 
    #   a resize is performed based on how much far away we are from T

    # Compute for the first time the speed of sound and the timestep required, then allocate accordingly
    a = np.sqrt(gamma * U_grid[:,2] / U_grid[:,0]) # a = sqrt(gamma * p / rho)
    
    Lp = abs(U_grid[:,1]) + a # Eigenvalues on u + a
    max_Lp = np.max(Lp)
    dt = c * dx / max_Lp

    # Preallocate the result array and a step_array to store the timesteps
    guessed_steps = int(np.ceil(T/dt))
    U_grid_result = np.zeros((guessed_steps, grid_samples, 3))
    a_grid_result = np.zeros((guessed_steps, grid_samples, 1))
    dt_array = np.zeros(guessed_steps)

    # Store the initial condition
    U_grid_result[0] = U_grid
    dt_array[0] = dt
    step_counter = 1 # Counter for the number of timesteps, starting at 1 
                    # since the initial condition is already stored 

    # Simulation loop
    while (t_total < T):
        U_grid, a_grid, dt = godunov_step(U_grid, dx, c, gamma = gamma)
        t_total += dt
        U_grid_result[step_counter] = U_grid
        a_grid_result[step_counter] = a_grid.reshape(-1, 1) # Fucking numpy 
        dt_array[step_counter] = dt
        step_counter += 1

        # Check if the array is full, if so, resize it
        if step_counter == guessed_steps:
            mean_dt = np.mean(dt_array)
            # 
            steps_left = int(np.ceil((T - t_total) / mean_dt))
            U_grid_result.resize((U_grid_result.shape[0] + steps_left, U_grid_result.shape[1], U_grid_result.shape[2]))
            a_grid_result.resize((a_grid_result.shape[0] + steps_left, a_grid_result.shape[1], a_grid_result.shape[2]))
            dt_array.resize(dt_array.shape[0] + steps_left)
            guessed_steps += steps_left


    # As a last step, resize the array to the actual number of timesteps
    U_grid_result = U_grid_result[:step_counter]
    a_grid_result = a_grid_result[:step_counter]
    dt_array = dt_array[:step_counter]

    return U_grid_result, a_grid_result, dt_array
