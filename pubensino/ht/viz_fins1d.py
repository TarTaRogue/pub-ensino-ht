# pubensino/ht/viz_fins1d.py
import matplotlib.pyplot as plt

def plot_profile(x, T, title):
    plt.figure()
    plt.plot(x, T)
    plt.xlabel("x [m]")
    plt.ylabel("T [K]")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_vs_Nf(Ns, Q_tot, eta_o):
    plt.figure()
    plt.plot(Ns, Q_tot)
    plt.xlabel("N_f")
    plt.ylabel("Q_total [W]")
    plt.title("Q_total vs N_f")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(Ns, eta_o)
    plt.xlabel("N_f")
    plt.ylabel("η_o [-]")
    plt.title("η_o vs N_f")
    plt.grid(True)
    plt.show()
