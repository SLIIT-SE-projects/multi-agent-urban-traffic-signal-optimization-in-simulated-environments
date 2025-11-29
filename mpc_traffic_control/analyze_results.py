import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_PATH = "data/logs/simulation_data.csv"

def plot_results():
    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Queues
    ax1.plot(df['time'], df['max_queue'], label='Max Queue', color='red')
    ax1.plot(df['time'], df['avg_queue'], label='Avg Queue', color='blue', linestyle='--')
    ax1.set_ylabel('Vehicles')
    ax1.set_title('Network Congestion')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Control Action
    # Filter out 0.0s where no optimization happened
    control_data = df[df['control_u'] > 0]
    if not control_data.empty:
        ax2.step(control_data['time'], control_data['control_u'], where='post', color='green', label='u (Green Fraction)')
    ax2.set_ylabel('Control Input (u)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    output_img = "data/logs/performance_plot.png"
    plt.savefig(output_img)
    print(f"Saved plot to {output_img}")
    plt.show()

if __name__ == "__main__":
    plot_results()