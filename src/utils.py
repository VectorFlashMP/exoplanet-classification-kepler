import matplotlib.pyplot as plt

def plot_light_curve(curve, title="Light Curve"):
    plt.figure()
    plt.plot(curve)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Brightness")
    plt.show()
