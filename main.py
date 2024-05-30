import warnings
warnings.filterwarnings('ignore')
from tkinter import Text, Toplevel, filedialog, Frame, messagebox, simpledialog, Tk, Button, Label, BOTH
from pandas import read_csv, concat, DataFrame
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.io import loadmat
from numpy import abs, where, fft
import numpy
import os
from model_training import EmotionDetector
from feature_extraction import process_eeg_data
import json


class CsvLoaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion detection")
        window_width = 800
        window_height = 640
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.root.resizable(False, False)
        self.animation = None
        self.DataFrame = None
        self.detector = EmotionDetector("processed_eeg_data.pkl")

        # Frame for the buttons and labels
        control_frame = Frame(self.root)
        control_frame.pack(pady=10)

        # Load CSV button
        self.load_button = Button(control_frame, text="Load CSV", command=self.load_csv)
        self.load_button.grid(row=0, column=1, padx=5)

        # Make Diagram button
        self.diagram_button = Button(control_frame, text="Make Diagram", command=self.make_diagram)
        self.diagram_button.grid(row=0, column=2, padx=5)

        # Convert MAT to CSV button
        self.convert_button = Button(control_frame, text="Convert MAT to CSV", command=self.convert_mat_to_csv)
        self.convert_button.grid(row=0, column=0, padx=5) 

        # Accuracy label
        self.accuracy_label = Label(control_frame, text="Accuracy:")
        self.accuracy_label.grid(row=1, column=1, sticky='e')

        with open('accuracy.json', 'r') as f:
            data = json.load(f)
            accuracy = data['accuracy']    
        accuracy_text = f"{accuracy:.2%}"

        self.accuracy_value_label = Label(control_frame, text=accuracy_text)
        self.accuracy_value_label.grid(row=1, column=2, sticky='w')

        # Status label
        self.status_label = Label(control_frame, text="")
        self.status_label.grid(row=2, column=0, columnspan=4)


        # Emotion label and field
        self.emotion_label = Label(control_frame, text="Emotion: ")
        self.emotion_label.grid(row=0, column=3, sticky='e') 

        self.emotion_field = Label(control_frame, text="")
        self.emotion_field.grid(row=0, column=4) 

        # Help button
        self.help_button = Button(self.root, text="?", command=self.show_help, font=('Arial', 10, 'bold'), padx=3, pady=0)
        self.help_button.pack(side='top', anchor='ne', padx=10, pady=5) 

        # Frame for the plot
        plot_frame = Frame(self.root)
        plot_frame.pack(fill=BOTH, expand=True)

        # Placeholder figure and canvas
        self.figure = Figure()
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

            self.DataFrame = read_csv(file_path)
            filename = file_path.split("/")[-1]
            self.status_label.config(text=f"Successfully loaded {filename}", fg="green") 
            self.emotion_field.config(text="")

    def make_diagram(self):
        if self.DataFrame is None:  # Check if the DataFrame has not been loaded
            self.status_label.config(text="No CSV file loaded. Please load a CSV file first.", fg="red")
        else:
            numeric_df = self.DataFrame.select_dtypes(include=['number'])
            if not numeric_df.empty:
                # Perform Fourier transform
                data_to_transform = numeric_df.iloc[:, 0].values
                fourier_transformed = fft.fft(data_to_transform)
                n = len(data_to_transform)
                freq_bins = fft.fftfreq(n, d=1)
                
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
                step_size = 100
                frames = range(0, (len(freq_bins_positive) + step_size - 1) // step_size)  # Adjust frame count based on step size
                interval = 10  # Milliseconds between frames

                # Configure animation
                self.animation = FuncAnimation(self.figure, update, frames=frames,
                                            init_func=init, blit=True, interval=interval, repeat=False)

                # Draw the canvas
                self.canvas.draw()

            else:
                self.status_label.config(text="No numeric columns found for Fourier transformation.", fg="red")

    def update_emotion_multiple(self):
        selected_sensors = [3, 4, 32, 40] 
        original_frequency = 1000 
        new_frequency = 200 
        max_cols = 102

        emotion_label_map = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
        dr = "/home/alex/UVT/Thesis/csv_files/2/1"
        files = sorted([f for f in os.listdir(dr) if f.endswith('.csv')])
        predictions = []

        for file in files[:24]:  
            file_path = os.path.join(dr, file)
            df = read_csv(file_path)

            # Process the EEG data, including resampling, shifting, windowing, and extracting features
            df_statistics = process_eeg_data(df, selected_sensors, original_frequency, new_frequency)
            numeric_df = DataFrame(numpy.array(df_statistics).T)
            processed_data = self.detector.preprocess_data_for_prediction(numeric_df, max_cols)

            # Use the detector to predict emotion based on the extracted features
            emotion = self.detector.predict_emotion(processed_data)
            chosen_emotion = emotion_label_map.get(emotion, 'Unknown')

            # Store prediction for the file
            predictions.append((file, chosen_emotion))

        # Print or display predictions for each file
        for file, emotion in predictions:
            print(f"File: {file} - Predicted Emotion: {emotion}")



    def update_emotion(self):
        # self.update_emotion_multiple()
        # return

        if self.DataFrame is not None:
            selected_sensors = [3, 4, 32, 40]  # Update as necessary based on your DataFrame columns
            original_frequency = 1000  # The original frequency of the EEG data
            new_frequency = 200  # The frequency to which the data should be resampled
            # expected_feature_count = 24276
            max_cols = 102

            # Process the EEG data, including resampling, shifting, windowing, and extracting features
            df_statistics = process_eeg_data(self.DataFrame, selected_sensors, original_frequency, new_frequency)

            # print("Number of features:\n", df_statistics.shape[1])

            numeric_df = DataFrame(numpy.array(df_statistics).T)

            # flattened_features = self.detector.preprocess_data([df_statistics])

            # print("Number of features:\n", flattened_features.shape[1])
            # print(flattened_features)

            # padded_features = numeric_df.join(DataFrame(0, index=numeric_df.index, columns=range(numeric_df.shape[1], max_cols)))

            # flattened_features = padded_features.values.flatten()

            processed_data = self.detector.preprocess_data_for_prediction(numeric_df, max_cols)

            # Use the detector to predict emotion based on the extracted features
            emotion = self.detector.predict_emotion(processed_data)

            # Map and update the GUI or output component with the emotion
            emotion_label_map = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
            emotion_color = {'Happy': 'green', 'Sad': 'blue', 'Fear': 'purple', 'Neutral': 'gray'}
            chosen_emotion = emotion_label_map.get(emotion, 'Unknown')
            self.emotion_field.config(text=chosen_emotion, fg=emotion_color.get(chosen_emotion, 'black'))

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
        session_names = sorted(session_keys)  # Sort sessions

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

    def show_help(self):
        # Create a new top-level window
        help_window = Toplevel(self.root)
        help_window.title("Help Information")
        # Set the size of the help window and center it on the parent window
        help_window_width = 800
        help_window_height = 250
        self.center_window_on_parent(self.root, help_window, help_window_width, help_window_height)

        help_window.resizable(False, False)  # Make the help window unresizable
        # Add a text widget to the help window
        text = Text(help_window, wrap='word')
        text.insert('1.0', (
            "How to use the application:\n\n"
            "Press the buttons in the following order:\n\n"
            "1. Convert MAT to CSV: Convert a chosen session from a .mat file to a .csv file.\n\n"
            "2. Load CSV: Click 'Load CSV' then select a csv file from a folder to load.\n\n"
            "3. Make Diagram: With a CSV file loaded, click the 'Make Diagram' button to generate a fast fourier diagram and to get the emotion for the respective .csv file.\n"
        ))
        text.config(state='disabled')  # Make the text widget read-only
        text.pack(expand=True, fill='both', padx=10, pady=10)

        # Optionally, add a close button
        close_button = Button(help_window, text="Close", command=help_window.destroy)
        close_button.pack(pady=5)

    def center_window_on_parent(self, parent_window, child_window, width, height):
        """
        Center the child_window on the parent_window with the specified width and height.
        """
        # Get the parent window's geometry
        parent_x = parent_window.winfo_x()
        parent_y = parent_window.winfo_y()
        parent_width = parent_window.winfo_width()
        parent_height = parent_window.winfo_height()

        # Calculate the position to center the child window on the parent window
        x = parent_x + (parent_width - width) // 2
        y = parent_y + (parent_height - height) // 2

        # Apply the calculated position and size to the child window
        child_window.geometry(f"{width}x{height}+{x}+{y}")


# Create the Tkinter window
root = Tk()

# Create the application instance
app = CsvLoaderApp(root)

# Start the application
root.mainloop()
