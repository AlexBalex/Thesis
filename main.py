from tkinter import filedialog, Frame, messagebox, simpledialog, Tk, Button, Label, BOTH
from pandas import read_csv, concat, DataFrame
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.io import loadmat
from numpy import abs, where
from numpy.fft import fft
import random
import os

class CsvLoaderApp:
    def convert_mat_to_csv(self):
        mat_files_dir = '/home/alex/UVT/thesis/mat_files'
        csv_files_dir = '/home/alex/UVT/thesis/csv_files'

        file_path = filedialog.askopenfilename(initialdir=mat_files_dir,
                                            filetypes=[("MAT-files", "*.mat")],
                                            title="Open MAT File")
        if not file_path:
            return  # User cancelled the file selection

        # Load the .mat file
        mat_contents = loadmat(file_path)

        # Extract session names that follow the pattern '_eeg1' to '_eeg24'
        session_keys = [key for key in mat_contents if 'eeg' in key]
        session_names = sorted(session_keys)  # Sort sessions if needed

        if not session_names:
            messagebox.showerror("Error", "No EEG sessions found in the selected .mat file.")
            return

        # Sort session names numerically based on the number part of the session name
        session_names = sorted(session_keys, key=lambda name: int(name.split('eeg')[1]))
        session_number_to_name = {i + 1: name for i, name in enumerate(session_names)}


        # Ask the user to select a session number
        session_number = simpledialog.askinteger("Select Session Number", "Enter a session number to convert (1 to 24):",
                                                parent=self.root, minvalue=1, maxvalue=len(session_names))
        if session_number is None or session_number not in session_number_to_name:
            return  # User cancelled or entered an invalid number

        session_name = session_number_to_name[session_number]

        # Get the data for the specified session
        session_data = mat_contents[session_name]

        if session_data is None:
            messagebox.showerror("Error", f"No data found for session '{session_name}'")
            return

        # Proceed with the conversion process
        dataframes_list = []
        for i, array in enumerate(session_data):
            array_df = DataFrame(array)
            array_df.columns = [f"{i}_{col}" for col in array_df.columns]
            dataframes_list.append(array_df)



        initial_filename = f"{os.path.basename(session_name)}_data.csv"  # Default filename for saving
        csv_filename = filedialog.asksaveasfilename(initialdir=csv_files_dir,
                                                    defaultextension=".csv",
                                                    filetypes=[("CSV files", "*.csv")],
                                                    title="Save the session as CSV",
                                                    initialfile=initial_filename)

        if not csv_filename:
            return  # User cancelled the save file dialog
        
        all_arrays_df = concat(dataframes_list, axis=1)
        all_arrays_df.to_csv(csv_filename, index=False)

        # Calculate the size of the CSV file
        file_size_bytes = os.path.getsize(csv_filename)
        file_size_megabytes = file_size_bytes / (1024 * 1024)

        csv_filename_only = os.path.basename(csv_filename)
        messagebox.showinfo("Success", f"The session data has been successfully saved as '{csv_filename_only}'.\nSize: {file_size_megabytes:.2f} MB")

    def __init__(self, root):
        self.root = root
        self.root.title("Emotion detection")
        self.root.geometry("800x640")  # Adjust size as needed
        self.animation= None

        # Frame for the buttons and labels
        control_frame = Frame(self.root)
        control_frame.pack(pady=10)

        # Load CSV button
        self.load_button = Button(control_frame, text="Load CSV", command=self.load_csv)
        self.load_button.grid(row=0, column=0, padx=5)

        # Make Diagram button
        self.diagram_button = Button(control_frame, text="Make Diagram", command=self.make_diagram)
        self.diagram_button.grid(row=0, column=1, padx=5)

        # Convert MAT to CSV button
        self.convert_button = Button(control_frame, text="Convert MAT to CSV", command=self.convert_mat_to_csv)
        self.convert_button.grid(row=0, column=3, padx=5)  # Notice column=3 for the new button

        # Status label
        self.status_label = Label(control_frame, text="")
        self.status_label.grid(row=1, column=0, columnspan=4)

        # Emotion label and field
        self.emotion_label = Label(control_frame, text="Emotion: ")
        self.emotion_label.grid(row=0, column=4, sticky='e')  # Moved to column=4

        self.emotion_field = Label(control_frame, text="")
        self.emotion_field.grid(row=0, column=5)  # Moved to column=5


        # Frame for the plot
        plot_frame = Frame(self.root)
        plot_frame.pack(fill=BOTH, expand=True)

        # Placeholder figure and canvas
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
   
    def load_csv(self):
        file_path = filedialog.askopenfilename(initialdir='/home/alex/UVT/thesis/csv_files',filetypes=[("CSV files", "*.csv")],title="Open CSV File")
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

            self.df = read_csv(file_path)
            filename = file_path.split("/")[-1]
            self.status_label.config(text=f"Successfully loaded {filename}", fg="green") 
            self.emotion_field.config(text="")

    def make_diagram(self):
        if self.df is not None:
            numeric_df = self.df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                # Perform Fourier transform on the first numeric column
                data_to_transform = numeric_df.iloc[:, 0].values
                fourier_transformed = fft.fft(data_to_transform)
                n = len(data_to_transform)
                freq_bins = fft.fftfreq(n, d=1)  # Assuming a sample rate of 1 for 'd'
                
                # Filter out only the positive frequencies for visualization
                positive_freq_indices = where(freq_bins >= 0)
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
                ax.set_ylim(0, abs(fourier_transformed_positive).max())

                # Initialize the line
                def init():
                    line.set_data([], [])
                    return line,

                # Update function for the animation
                def update(frame):
                    # Use the frame as an index slice up to current frame with step_size adjustment
                    line.set_data(freq_bins_positive[:frame * step_size], abs(fourier_transformed_positive[:frame * step_size]))
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
root = Tk()

# Create the application instance
app = CsvLoaderApp(root)

# Start the application
root.mainloop()
