import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Sample DataFrame creation
# Assuming 'df' is your DataFrame after loading the CSV
np.random.seed(0)  # For consistent random data
df = pd.DataFrame(np.random.rand(100, 4), columns=['A', 'B', 'C', 'D'])  # Example DataFrame

class CsvLoaderApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Loader")
        self.root.geometry("300x200")  # Adjusted to accommodate the success message
        self.df = None  # Initialize DataFrame as None

        self.load_button = tk.Button(self.root, text="Load CSV", command=self.load_csv)
        self.load_button.pack(pady=10)

        self.diagram_button = tk.Button(self.root, text="Make Diagram", command=self.make_diagram)
        self.diagram_button.pack(pady=10)

        self.status_label = tk.Label(self.root, text="", fg="green")  # Label for displaying the success message
        self.status_label.pack(pady=5)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:  # Proceed if a file was selected
            self.df = pd.read_csv(file_path)
            filename = file_path.split("/")[-1]  # Extract the file name
            self.status_label.config(text=f"Successfully loaded {filename}", fg="green")  # Update the label with the success message

    def make_diagram(self):
        if self.df is not None:
            numeric_df = self.df.select_dtypes(include=['number'])

            if not numeric_df.empty:
                row_means = numeric_df.mean(axis=1)
                
                fig, ax = plt.subplots()
                ax.set_title('Arithmetic Mean of Each Row Over Numeric Columns')
                ax.set_xlabel('Row Index')
                ax.set_ylabel('Arithmetic Mean')
                ax.grid(True)

                x_data, y_data = [], []
                ln, = plt.plot([], [], 'r-', animated=True)

                def init():
                    max_bound = max(abs(row_means.min()), abs(row_means.max()))
                    ax.set_xlim(0, len(row_means) - 1)
                    ax.set_ylim(-max_bound, max_bound)  # Set y-axis limits to be symmetrical around zero
                    return ln,

                def update(frame):
                    if len(x_data) > 0:
                        if x_data[-1] > frame: return ln,
                    x_data.append(frame)
                    y_data.append(row_means.iloc[frame])
                    ln.set_data(x_data, y_data)
                    return ln,

                step_size = 10  # Adjust step size as needed
                ani = FuncAnimation(fig, update, frames=range(0, len(row_means), step_size),
                    init_func=init, blit=True, interval=1)

                plt.show()
            else:
                self.status_label.config(text="No numeric columns found.", fg="red")
        else:
            self.status_label.config(text="No CSV file loaded.", fg="red")




# Create the Tkinter window
root = tk.Tk()

# Create the application instance
app = CsvLoaderApp(root)

# Start the application
root.mainloop()
