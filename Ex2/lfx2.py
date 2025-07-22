import numpy as np

### Functions for the Lax-Friedrichs scheme 

def assert_unphysicalU(U):
    """
    This function checks if the primitive variables U = [rho, u, p] are physical
    and raises an error if they are not.
    U is a 2D array of shape (n_samples, 3), where each row is a sample
    of the primitive variables [rho, u, p].
    """
    if np.any(U[:,0] <= 0):
        raise ValueError("Density is negative or zero")
    if np.any(U[:,2] < 0):
        raise ValueError("Pressure is negative")

def assert_unphysicalW(W):
    """
    This function checks if the conserved variables W = [rho, rho*u, E] are physical
    and raises an error if they are not.
    W is a 2D array of shape (n_samples, 3), where each row is a sample
    of the conserved variables [rho, rho*u, E].
    """
    if np.any(W[:,0] <= 0):
        raise ValueError("Density is negative or zero")
    if np.any(W[:,2] < 0):
        raise ValueError("Energy is negative")

def Flux(W, gamma):
    """
    This function computes the flux vector F = [rho*u, rho*u^2 + p, u*(E + p)] for a 2D array W, where gamma is
    the ratio of specific heats.
    """
    F = np.zeros_like(W)

    rho = W[:,0]
    rho_u = W[:,1]
    E = W[:,2]

    u = rho_u / rho
    p = (gamma - 1) * (E - 0.5 * rho * u ** 2)

    F[:,0] = rho * u
    F[:,1] = rho * u ** 2 + p
    F[:,2] = u * (E + p)
    
    return F

## U is the primitive variables [rho, u, p], 
# W is the conserved variables [rho, rho*u, E] E = rho*e + 0.5*rho*u^2

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

def lfx2_step(U_grid, dx, c, gamma = 1.4):
    """
     This function performs a single temporal step of the Lax-Friedrichs scheme
     on a 2D array U_grid, with grid spacing dx, and Courant number c.
     The timestep is computed from the Courant number and the CFL condition

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

    W_grid_gc = U2W(U_grid_gc, gamma = gamma)
    
    
    a_grid_gc = np.sqrt(gamma * U_grid_gc[:,2] / U_grid_gc[:,0]) # a = sqrt(gamma * p / rho)
                                                               # Speed of sound
    Lp = abs(U_grid_gc[:,1]) + a_grid_gc # Eigenvalues on u + a
    Lm = abs(U_grid_gc[:,1]) - a_grid_gc # Eigenvalues on u - a
    
    Lmax = np.maximum(np.max(Lp), np.max(Lm))
    # Compute the timestep
    dt = c * dx / Lmax
    
    # Compute the fluxes
    F_grid_gc = Flux(W_grid_gc, gamma = gamma)
    
    # Prediction step
    

    # Compute the half-steps
    W_half_p = 0.5 * (W_grid_gc[2:] + W_grid_gc[1:-1]) - 0.5 * dt / dx * (F_grid_gc[2:] - F_grid_gc[1:-1])
    W_half_m = 0.5 * (W_grid_gc[1:-1] + W_grid_gc[:-2]) - 0.5 * dt / dx * (F_grid_gc[1:-1] - F_grid_gc[:-2])

    # Compute the fluxes at the half-steps
    F_half_p = Flux(W_half_p, gamma = gamma)
    F_half_m = Flux(W_half_m, gamma = gamma)

    
    # Correction step
    W_grid_dt = (W_half_m + W_half_p)/2 - 0.5 *dt / dx * (F_half_p - F_half_m)
    
    # Convert the conserved variables back to the primitive variables
 
    U_grid_dt = W2U(W_grid_dt, gamma = gamma)
    a_grid =  a_grid_gc[1:-1]
    
    return U_grid_dt, a_grid , dt

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

def LFx2(U0, dx, c, T, gamma = 1.4):
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
    a_grid_result[0] = a.reshape(-1, 1) # Store the initial speed of sound

    # Initialize the a grid
    dt_array = np.zeros(guessed_steps)

    # Store the initial condition
    U_grid_result[0] = U_grid
    dt_array[0] = dt
    step_counter = 1 # Counter for the number of timesteps, starting at 1 
                    # since the initial condition is already stored 

    # Simulation loop
    while (t_total < T):
        U_grid, a_grid, dt = lfx2_step(U_grid, dx, c, gamma = gamma)
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
