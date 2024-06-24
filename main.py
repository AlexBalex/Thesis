import warnings
warnings.filterwarnings('ignore')
from tkinter import Text, Toplevel, filedialog, Frame, messagebox, simpledialog, Tk, Button, Label, BOTH
from pandas import read_csv, concat, DataFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.io import loadmat
from scipy.signal import welch
from numpy import abs
from screeninfo import get_monitors
from scipy import fft
import numpy
import os
from model_training import EmotionDetector
from feature_extraction import process_eeg_data
import json


class CsvLoaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FeelFinder")
        monitors = get_monitors()
        main_monitor = monitors[0]
        desired_width = 1200
        desired_height = 800
        center_x = main_monitor.x + (main_monitor.width - desired_width) // 2
        center_y = main_monitor.y + (main_monitor.height - desired_height) // 2
        self.root.geometry(f"{desired_width}x{desired_height}+{center_x}+{center_y}")
        self.root.resizable(False, False)
        self.animation = None
        self.DataFrame = None
        self.detector = EmotionDetector("processed_eeg_data.pkl")

        # Frame for the buttons and labels
        control_frame = Frame(self.root)
        control_frame.pack(pady=10)

        # Retrain Model button
        self.retrain_button = Button(control_frame, text="Retrain Model", command=self.retrain_model)
        self.retrain_button.grid(row=0, column=0, padx=5)

        # Load CSV button
        self.load_button = Button(control_frame, text="Load CSV", command=self.load_csv)
        self.load_button.grid(row=0, column=2, padx=5)

        # Detect emotion button
        self.emotion_button = Button(control_frame, text="Detect Emotion", command=self.update_emotion)
        self.emotion_button.grid(row=0, column=3, padx=5)

        # Convert MAT to CSV button
        self.convert_button = Button(control_frame, text="Convert MAT to CSV", command=self.convert_mat_to_csv)
        self.convert_button.grid(row=0, column=1, padx=5) 

       # Accuracy label
        self.accuracy_label = Label(control_frame, text="Accuracy:")
        self.accuracy_label.grid(row=1, column=0, columnspan=2, sticky='e')

        # Check if the file exists
        if not os.path.isfile('accuracy.json'):
            default_data = {"accuracy": 0.0}
            with open('accuracy.json', 'w') as f:
                json.dump(default_data, f)

        with open('accuracy.json', 'r') as f:
            data = json.load(f)
            accuracy = data['accuracy']    
        accuracy_text = f"{accuracy:.2%}"

        self.accuracy_value_label = Label(control_frame, text=accuracy_text)
        self.accuracy_value_label.grid(row=1, column=2, columnspan=2, sticky='w')

        # Status label
        self.status_label = Label(control_frame, text="")
        self.status_label.grid(row=2, column=0, columnspan=6)

        # Emotion label and field
        self.emotion_label = Label(control_frame, text="Emotion: ")
        self.emotion_label.grid(row=0, column=4, sticky='e') 

        self.emotion_field = Label(control_frame, text="")
        self.emotion_field.grid(row=0, column=5) 

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
        file_path = filedialog.askopenfilename(initialdir='/home/alex/UVT/thesis/',filetypes=[("CSV files", "*.csv")],title="Open CSV File")
        if file_path: 
            # Stop existing animation
            if self.animation is not None and self.animation.event_source is not None:
                self.animation.event_source.stop()
            
            self.animation = None  
        
            
            self.figure.clear()
            
            
            if self.canvas and hasattr(self.canvas, 'draw') and callable(self.canvas.draw):
                self.canvas.draw()

            self.DataFrame = read_csv(file_path)
            filename = file_path.split("/")[-1]
            self.status_label.config(text=f"Successfully loaded {filename}", fg="green") 
            self.emotion_field.config(text="")

    def plot_fft(self, combined_data, ax, sampling_rate):
        
        fourier_transformed = fft.fft(combined_data)
        n = len(combined_data)
        freq_bins = fft.fftfreq(n, d=1.0/sampling_rate)

        
        positive_freq_indices = numpy.where(freq_bins >= 0)
        freq_bins_positive = freq_bins[positive_freq_indices]
        fourier_transformed_positive = abs(fourier_transformed[positive_freq_indices])

        # Plot the Fourier-transformed combined data
        ax.plot(freq_bins_positive, fourier_transformed_positive, 'r-')
        ax.set_title('Fast Fourier Transform Diagram')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.grid(True)

        
        ax.set_xlim(0, 90) 

    def plot_psd(self, combined_data, ax, sampling_rate):
        
        freqs, psd = welch(combined_data, fs=sampling_rate, nperseg=1024)

        # Plot the PSD data
        ax.plot(freqs, 10 * numpy.log10(psd), 'b-')
        ax.set_title('Power Spectral Density Diagram')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power/Frequency (dB/Hz)')
        ax.grid(True)

        
        ax.set_xlim(0, 90)

    def update_emotion(self):

        if self.DataFrame is None:  
            self.status_label.config(text="No CSV file loaded. Please load a CSV file first.", fg="red")
        else:
            selected_sensors = [3, 4, 32, 40]  # Update as necessary based on your DataFrame columns in this case they represent FP1, FP2, TP7, TP8
            original_frequency = 1000  
            new_frequency = 200  
            
            max_cols = 102

            
            df_statistics = process_eeg_data(self.DataFrame, selected_sensors, original_frequency, new_frequency)
            numeric_df = DataFrame(numpy.array(df_statistics).T)

            
            combined_data = numeric_df.sum(axis=1).values

            
            self.figure.clear()
            ax1 = self.figure.add_subplot(121)  
            ax2 = self.figure.add_subplot(122)  

            # Plot the FFT on the left
            self.plot_fft(combined_data, ax1, sampling_rate=new_frequency)

            # Plot the PSD on the right
            self.plot_psd(combined_data, ax2, sampling_rate=new_frequency)

            self.canvas.draw()

            processed_data = self.detector.preprocess_data_for_prediction(numeric_df, max_cols)

            emotion = self.detector.predict_emotion(processed_data)

            emotion_label_map = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}
            emotion_color = {'Happy': 'green', 'Sad': 'blue', 'Fear': 'purple', 'Neutral': 'gray'}
            chosen_emotion = emotion_label_map.get(emotion, 'Unknown')
            self.emotion_field.config(text=chosen_emotion, fg=emotion_color.get(chosen_emotion, 'black'))

    def convert_mat_to_csv(self):
        mat_files_dir = '/home/alex/UVT/thesis/'
        csv_files_dir = '/home/alex/UVT/thesis/'
        
        file_path = filedialog.askopenfilename(initialdir=mat_files_dir,
                                            filetypes=[("MAT-files", "*.mat")],
                                            title="Open MAT File")
        if not file_path:
            return  

        mat_contents = loadmat(file_path)

        session_keys = [key for key in mat_contents if 'eeg' in key]
        session_names = sorted(session_keys) 

        if not session_names:
            messagebox.showerror("Error", "No EEG sessions found in the selected .mat file.")
            return

        session_names = sorted(session_keys, key=lambda name: int(name.split('eeg')[1]))
        session_number_to_name = {i + 1: name for i, name in enumerate(session_names)}

        session_number = simpledialog.askinteger("Select Session Number", "Enter a session number to convert (1 to 24):",
                                                parent=self.root, minvalue=1, maxvalue=len(session_names))
                                                
        if session_number is None or session_number not in session_number_to_name:
            return 

        session_name = session_number_to_name[session_number]

        session_data = mat_contents[session_name]

        if session_data is None:
            messagebox.showerror("Error", f"No data found for session '{session_name}'")
            return

        dataframes_list = []
        for i, array in enumerate(session_data):
            array_df = DataFrame(array)
            array_df.columns = [f"{i}_{col}" for col in array_df.columns]
            dataframes_list.append(array_df)

        initial_filename = f"{os.path.basename(session_name)}_data.csv" 
        csv_filename = filedialog.asksaveasfilename(initialdir=csv_files_dir,
                                                    defaultextension=".csv",
                                                    filetypes=[("CSV files", "*.csv")],
                                                    title="Save the session as CSV",
                                                    initialfile=initial_filename)

        if not csv_filename:
            return  
        
        all_arrays_df = concat(dataframes_list, axis=1)
        all_arrays_df.to_csv(csv_filename, index=False)

        # Size of the CSV file
        file_size_bytes = os.path.getsize(csv_filename)
        file_size_megabytes = file_size_bytes / (1024 * 1024)

        csv_filename_only = os.path.basename(csv_filename)
        messagebox.showinfo("Success", f"The session data has been successfully saved as '{csv_filename_only}'.\nSize: {file_size_megabytes:.2f} MB")

    def show_help(self):
        help_window = Toplevel(self.root)
        help_window.title("Help Information")
        help_window_width = 800
        help_window_height = 250
        self.center_window_on_parent(self.root, help_window, help_window_width, help_window_height)

        help_window.resizable(False, False) 
        text = Text(help_window, wrap='word')
        text.insert('1.0', (
            "How to use the application:\n\n"
            "Retrain Model: Retrain the emotion detection model with the latest training data.\n\n"
            "Convert MAT to CSV: Convert a chosen session from a MAT file to a CSV file.\n\n"
            "Load CSV: Click 'Load CSV' then select a CSV file from a folder to load.\n\n"
            "Detect Emotion: With a CSV file loaded, click the 'Detect Emotion' button to generate a fast fourier diagram togheter with a power spectral density diagram and to get the emotion for the respective CSV file.\n"
        ))
        text.config(state='disabled')
        text.pack(expand=True, fill='both', padx=10, pady=10)

        close_button = Button(help_window, text="Close", command=help_window.destroy)
        close_button.pack(pady=5)

    def center_window_on_parent(self, parent_window, child_window, width, height):
        parent_x = parent_window.winfo_x()
        parent_y = parent_window.winfo_y()
        parent_width = parent_window.winfo_width()
        parent_height = parent_window.winfo_height()

        x = parent_x + (parent_width - width) // 2
        y = parent_y + (parent_height - height) // 2

        child_window.geometry(f"{width}x{height}+{x}+{y}")

    def retrain_model(self):
            data = self.detector.load_data('feature_extraction/processed_eeg_data_opt_old.pkl')
            features = [DataFrame(numpy.array(feature).T) for feature in data['features']]
            labels = data['labels']
            X = numpy.array(self.detector.preprocess_data(features))
            y = numpy.array(labels)
            
            self.detector.train_model(X, y)
            self.detector.load_model("processed_eeg_data.pkl")

            with open('accuracy.json', 'r') as f:
                data = json.load(f)
                accuracy = data['accuracy']    
            accuracy_text = f"{accuracy:.2%}"
            self.accuracy_value_label.config(text=accuracy_text)
            
            messagebox.showinfo("Success", "Model retrained successfully.")

# Tkinter window
root = Tk()

# Application instance
app = CsvLoaderApp(root)

# Start the application
root.mainloop()
