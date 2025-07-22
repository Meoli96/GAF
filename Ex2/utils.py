import numpy as np
image_path = "Images/"
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

def plot_results(_x: np.ndarray, u_prop: np.ndarray, title: str = None, save: bool = False):
    # Plot the results of the solution
    import matplotlib.pyplot as plt 
    plt.figure()
    
    # Plot the initial configuration
    plt.plot(_x, u_prop[0,:], label="t=0")
    # Plot 4 intermediate configurations

    # Find a way to plot the intermediate configurations
    n_steps = u_prop.shape[0]
    for i in range (n_steps//10, n_steps, n_steps//10):
        plt.plot(_x, u_prop[i,:], label="n="+str(i), linestyle="--", alpha=0.5)
    # Plot the final configuration
    plt.plot(_x, u_prop[-1,:], label="n=" + str(n_steps))
    if (title):
        plt.title(title)
    plt.ylabel("Amplitude")
    plt.xlabel("x")
    if (save):
        plt.savefig(image_path + title + "_prop" + ".png")

    plt.show()



class Result:
    # This class acts as a container for the results of the simulation
    # It is also responsible for computing L1 and TV norms, and plotting the results
    def __init__(self, _x:np.ndarray, u_prop:np.ndarray, c, title: str) -> None:
        self._x = _x
        self.data = u_prop
        self.c = c

        if (c > 1):
            print("The Courant number is greater than 1. The simulation may not be stable")
        # Compute the L1 and TV norms
        self.L1 = L1_norm(u_prop)
        self.TV = TV_norm(u_prop)
        self.title = title 
    
    def plot(self, save:bool = False):
        import matplotlib.pyplot as plt 
        plot_results(self._x,  self.data, self.title + " - c = " + str(self.c), save=save)
        # Plot the L1 and TV norms
        plt.figure()
        plt.plot(self.L1, label="L1 norm")
        plt.plot(self.TV, label="TV norm")
        plt.legend()
        if (save):
            plt.savefig(image_path + self.title + "_norms" + ".png")
        

    def animate(self, fps: int = 60):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig, ax = plt.subplots()
        fig.suptitle(self.title)
        line, = ax.plot(self._x, self.data[0,:])
        ax.set_ylim(0, 3)
        ax.set_xlim(0, 10)

        def animate(i):
            line.set_ydata(self.data[i,:])
            return line,

        ani = animation.FuncAnimation(
            fig, animate, frames=self.data.shape[0], interval=1/fps
        )
        return ani

        

def get_invariants(rho, p, u, a, gamma):
    """
    This function computes the invariants of the system.
    The invariants are:
    - a - du
    - a + du
    where a is the speed of sound, du is the velocity perturbation and d = (gamma - 1)/2.

    variable are a nxm array, where n is the length of the grid and m is the number of temporal steps.
    """
   
    
    return invariant1, invariant2