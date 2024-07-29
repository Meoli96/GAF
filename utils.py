import numpy as np

def L1_norm(u_prop: np.ndarray) -> float:
    # Computes the L1 norm of the solution, being a (n_steps, space_dim) np.array
    n_steps = u_prop.shape[0]
    n_dx = u_prop.shape[1]

    L1 = np.zeros(n_steps)
    for i in range(n_steps):
        L1[i] = np.sum(np.abs(u_prop[i, :])) / n_dx   
    
    return L1


def TV_norm(u_prop: np.ndarray) -> float:
    # Compute the total variation of the solution
    n_steps = u_prop.shape[0]
    n_dx = u_prop.shape[1]

    
    TV = np.sum(np.abs(np.diff(u_prop, axis=1)), axis=1)
    return TV

def plot_results(_x: np.ndarray, u_prop: np.ndarray, title: str = None):
    # Plot the results of the solution
    import matplotlib.pyplot as plt 
    plt.figure()
    # Plot the initial configuration
    plt.plot(_x, u_prop[0,:], label="t=0")
    # Plot 4 intermediate configurations
    for i in range (1, 5):
        plt.plot(_x, u_prop[i*10,:], label="t="+str(i*10), linestyle="--", color="orange", alpha=0.5)
    # Plot the final configuration
    plt.plot(_x, u_prop[-1,:], label="t=50", color="red")
    if (title):
        plt.title(title)
    plt.ylabel("Amplitude")
    plt.xlabel("x")
    plt.legend()
    plt.show()



class Result:
    # This class acts as a container for the results of the simulation
    # It is also responsible for computing L1 and TV norms, and plotting the results
    def __init__(self, _x:np.ndarray, u_prop:np.ndarray, c, title: str) -> None:
        self._x = _x
        self.u_prop = u_prop
        self.c = c

        if (c > 1):
            print("The Courant number is greater than 1. The simulation may not be stable")
        # Compute the L1 and TV norms
        self.L1 = L1_norm(u_prop)
        self.TV = TV_norm(u_prop)
        self.title = title + " - c = " + str(c)
    
    def plot(self):
        import matplotlib.pyplot as plt 
        plot_results(self._x,  self.u_prop, self.title)
        # Plot the L1 and TV norms
        plt.figure()
        plt.plot(self.L1, label="L1 norm")
        plt.plot(self.TV, label="TV norm")
        plt.legend()
        

     

        