import tkinter as tk
from tkinter import filedialog, Frame
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import random

np.random.seed(0)
df = pd.DataFrame(np.random.rand(100, 4), columns=['A', 'B', 'C', 'D'])  

class CsvLoaderApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Loader")
        self.root.geometry("800x640")  # Adjust size as needed
        self.animation= None
        # Frame for the buttons and labels
        control_frame = Frame(self.root)
        control_frame.pack(pady=10)

        # Load CSV button
        self.load_button = tk.Button(control_frame, text="Load CSV", command=self.load_csv)
        self.load_button.grid(row=0, column=0, padx=5)

        # Make Diagram button
        self.diagram_button = tk.Button(control_frame, text="Make Diagram", command=self.make_diagram)
        self.diagram_button.grid(row=0, column=1, padx=5)

        # Status label
        self.status_label = tk.Label(control_frame, text="")
        self.status_label.grid(row=1, column=0, columnspan=2)

        # Emotion label and field
        self.emotion_label = tk.Label(control_frame, text="Emotion: ")
        self.emotion_label.grid(row=0, column=2, sticky='e')

        self.emotion_field = tk.Label(control_frame, text="")
        self.emotion_field.grid(row=0, column=3)

        # Frame for the plot
        plot_frame = Frame(self.root)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Placeholder figure and canvas
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path: 
            # Stop any existing animation
            if self.animation is not None and self.animation.event_source is not None:
                self.animation.event_source.stop()
            
            self.animation = None  # Reset the animation reference
        
            # Clear the plot
            self.figure.clear()
            
            # Check if canvas exists then draw
            if self.canvas and hasattr(self.canvas, 'draw') and callable(self.canvas.draw):
                self.canvas.draw()

            self.df = pd.read_csv(file_path)
            filename = file_path.split("/")[-1]
            self.status_label.config(text=f"Successfully loaded {filename}", fg="green") 
            self.emotion_field.config(text="")

    def make_diagram(self):
        if self.df is not None:
            numeric_df = self.df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                # Perform Fourier transform on the first numeric column
                data_to_transform = numeric_df.iloc[:, 0].values
                fourier_transformed = np.fft.fft(data_to_transform)
                n = len(data_to_transform)
                freq_bins = np.fft.fftfreq(n, d=1)  # Assuming a sample rate of 1 for 'd'
                
                # Filter out only the positive frequencies for visualization
                positive_freq_indices = np.where(freq_bins >= 0)
                freq_bins_positive = freq_bins[positive_freq_indices]
                fourier_transformed_positive = fourier_transformed[positive_freq_indices]

                # Clear the figure and set up new axes
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.set_title('Frequency Spectrum Animation')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Magnitude')
                ax.grid(True)

                # Prepare for animation
                line, = ax.plot([], [], 'r-', animated=True)
                ax.set_xlim(0, freq_bins_positive.max())
                ax.set_ylim(0, np.abs(fourier_transformed_positive).max())

                # Initialize the line
                def init():
                    line.set_data([], [])
                    return line,

                # Update function for the animation
                def update(frame):
                    # Use the frame as an index slice up to current frame with step_size adjustment
                    line.set_data(freq_bins_positive[:frame * step_size], np.abs(fourier_transformed_positive[:frame * step_size]))
                    if len(freq_bins_positive) - frame * step_size <= step_size:
                        self.update_emotion()
                    return line,

                # Setup animation parameters
                step_size = 10
                frames = range(0, (len(freq_bins_positive) + step_size - 1) // step_size)  # Adjust frame count based on step size
                interval = 10  # Milliseconds between frames

                # Configure animation
                self.animation = FuncAnimation(self.figure, update, frames=frames,
                                            init_func=init, blit=True, interval=interval, repeat=False)

                # Draw the canvas
                self.canvas.draw()

            else:
                self.status_label.config(text="No numeric columns found for Fourier transformation.", fg="red")
        else:
            self.status_label.config(text="No CSV file loaded.", fg="red")


    def update_emotion(self):
        emotions = ['Happy', 'Sad', 'Fear', 'Neutral']
        emotion_color = {
            'Happy': 'green',
            'Sad': 'blue',
            'Fear': 'purple',
            'Neutral': 'gray'
        }
        chosen_emotion = random.choice(emotions)
        self.emotion_field.config(text=chosen_emotion, fg=emotion_color[chosen_emotion])

# Create the Tkinter window
root = tk.Tk()

# Create the application instance
app = CsvLoaderApp(root)

# Start the application
root.mainloop()
