import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial


def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def load_signal_file():
    try:
        signal_path = os.path.join(get_base_path(), "signal.txt")
        with open(signal_path, "r") as f:
            lines = f.readlines()[1:]  # skip header
            x_vals, y_vals = [], []
            for line in lines:
                t_val, y_val = map(float, line.strip().split(','))
                x_vals.append(t_val)
                y_vals.append(y_val)
        return np.array(x_vals), np.array(y_vals)
    except Exception as e:
        messagebox.showerror("Xəta", f"Fayl oxunarkən xəta baş verdi:\n\n{e}")
        return None, None

# Sequential version (original)
def calculate_fourier_sequential(x, y, omega_0, nmax=50):
    T = x[-1] - x[0]
    a0 = (2 / T) * np.trapz(y, x)
    an = []
    bn = []
    for n in range(1, nmax + 1):
        cos_term = np.cos(n * omega_0 * x)
        sin_term = np.sin(n * omega_0 * x)
        a_n = (2 / T) * np.trapz(y * cos_term, x)
        b_n = (2 / T) * np.trapz(y * sin_term, x)
        an.append(a_n)
        bn.append(b_n)
    return a0, an, bn

# Function to calculate single coefficient pair (for multiprocessing)
def calculate_coefficient_pair(args):
    n, x, y, omega_0, T = args
    cos_term = np.cos(n * omega_0 * x)
    sin_term = np.sin(n * omega_0 * x)
    a_n = (2 / T) * np.trapz(y * cos_term, x)
    b_n = (2 / T) * np.trapz(y * sin_term, x)
    return n, a_n, b_n

# Multiprocessing version
def calculate_fourier_multiprocessing(x, y, omega_0, nmax=50):
    T = x[-1] - x[0]
    a0 = (2 / T) * np.trapz(y, x)
    
    # Prepare arguments for parallel processing
    args_list = [(n, x, y, omega_0, T) for n in range(1, nmax + 1)]
    
    # Use all available CPU cores
    num_cores = multiprocessing.cpu_count()
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(calculate_coefficient_pair, args_list))
    
    # Sort results by n and extract coefficients
    results.sort(key=lambda x: x[0])
    an = [result[1] for result in results]
    bn = [result[2] for result in results]
    
    return a0, an, bn

# Threading version (may not be as effective due to GIL in Python)
def calculate_fourier_threading(x, y, omega_0, nmax=50):
    T = x[-1] - x[0]
    a0 = (2 / T) * np.trapz(y, x)
    
    # Prepare arguments for parallel processing
    args_list = [(n, x, y, omega_0, T) for n in range(1, nmax + 1)]
    
    # Use threading (limited by GIL but still may show some improvement)
    num_threads = multiprocessing.cpu_count()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(calculate_coefficient_pair, args_list))
    
    # Sort results by n and extract coefficients
    results.sort(key=lambda x: x[0])
    an = [result[1] for result in results]
    bn = [result[2] for result in results]
    
    return a0, an, bn

def reconstruct_signal(x, a0, an, bn, omega_0):
    y_approx = np.full_like(x, a0 / 2)
    for n in range(1, len(an) + 1):
        y_approx += an[n - 1] * np.cos(n * omega_0 * x) + bn[n - 1] * np.sin(n * omega_0 * x)
    return y_approx

def show_plot(x, y_original, y_approx):
    fig = plt.Figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(x, y_original, label="Orijinal Siqnal", linewidth=2)
    ax.plot(x, y_approx, label="Fourier Təqribi", linestyle="--", linewidth=2)
    ax.set_title("Harmonik Siqnal")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend()

    if hasattr(show_plot, 'canvas') and show_plot.canvas:
        show_plot.canvas.get_tk_widget().destroy()

    show_plot.canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    show_plot.canvas.draw()
    show_plot.canvas.get_tk_widget().pack()

def run_analysis(method_name, calculate_func):
    omega_str = omega_var.get()
    if not omega_str:
        timing_label.config(text="Omega₀ daxil edin.", fg="red")
        return
    try:
        omega_0 = float(omega_str)
        if omega_0 <= 0:
            raise ValueError("Omega₀ müsbət olmalıdır.")
    except Exception as e:
        timing_label.config(text=f"Xəta: {e}", fg="red")
        return

    x, y = load_signal_file()
    if x is None or y is None:
        timing_label.config(text="Siqnal faylı oxunmadı.", fg="red")
        return

    # Measure execution time
    start_time = time.time()
    a0, an, bn = calculate_func(x, y, omega_0)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    # Display coefficients
    text = f"a₀ = {a0:.6f}\n"
    for i in range(min(10, len(an))):  # Show first 10 coefficients
        text += f"a{i+1} = {an[i]:.6f},  b{i+1} = {bn[i]:.6f}\n"
    if len(an) > 10:
        text += f"... və daha {len(an)-10} əmsal\n"
    
    result_label.config(text=text)
    
    # Display timing results
    num_cores = multiprocessing.cpu_count()
    timing_text = f"{method_name}\n"
    timing_text += f"İcra müddəti: {execution_time:.4f} saniyə\n"
    timing_text += f"CPU nüvələri: {num_cores}\n"
    
    # Store timing for comparison
    if not hasattr(run_analysis, 'times'):
        run_analysis.times = {}
    run_analysis.times[method_name] = execution_time
    
    # Calculate speedup if we have sequential time
    if 'Sequential' in run_analysis.times and method_name != 'Sequential':
        speedup = run_analysis.times['Sequential'] / execution_time
        efficiency = speedup / num_cores * 100
        timing_text += f"Sürət artımı: {speedup:.2f}x\n"
        timing_text += f"Səmərəlilik: {efficiency:.1f}%"
    
    timing_label.config(text=timing_text, fg="green")

    # Show plot
    y_approx = reconstruct_signal(x, a0, an, bn, omega_0)
    show_plot(x, y, y_approx)

def run_sequential():
    # Use larger nmax for better parallel testing
    run_analysis("Sequential (Ardıcıl)", lambda x, y, w: calculate_fourier_sequential(x, y, w, nmax=1000))

def run_multiprocessing():
    run_analysis("Multiprocessing (Paralel)", lambda x, y, w: calculate_fourier_multiprocessing(x, y, w, nmax=1000))

def run_threading():
    run_analysis("Threading (Çox axınlı)", lambda x, y, w: calculate_fourier_threading(x, y, w, nmax=1000))

def create_gui():
    global omega_var, result_label, timing_label, plot_frame

    window = tk.Tk()
    window.title("Paralel Furye Analizi - Ləman Nəzirli")
    window.geometry("900x700")

    # Input section
    input_frame = tk.Frame(window)
    input_frame.pack(pady=10)

    label_frame = tk.Frame(input_frame)
    label_frame.pack()

    label_bold = tk.Label(label_frame, text="Omega₀", font=("Arial", 12, "bold"))
    label_bold.pack(side="left")

    label_normal = tk.Label(label_frame, text=" dəyərini daxil edin:", font=("Arial", 12))
    label_normal.pack(side="left")

    omega_var = tk.StringVar()
    omega_entry = ttk.Entry(input_frame, textvariable=omega_var, font=("Arial", 12))
    omega_entry.pack(pady=5)

    # Buttons section
    button_frame = tk.Frame(window)
    button_frame.pack(pady=10)

    btn1 = tk.Button(button_frame, text="Sequential\n(Tək nüvə)", 
                     command=run_sequential, bg="lightblue", font=("Arial", 10, "bold"))
    btn1.pack(side="left", padx=5)

    btn2 = tk.Button(button_frame, text="Multiprocessing\n(Çox nüvə)", 
                     command=run_multiprocessing, bg="lightgreen", font=("Arial", 10, "bold"))
    btn2.pack(side="left", padx=5)

    btn3 = tk.Button(button_frame, text="Threading\n(Çox axın)", 
                     command=run_threading, bg="lightyellow", font=("Arial", 10, "bold"))
    btn3.pack(side="left", padx=5)

    # Results section
    results_frame = tk.Frame(window)
    results_frame.pack(pady=10, fill="both", expand=True)

    # Coefficients display
    result_label = tk.Label(results_frame, text="Nəticələr burada göstəriləcək", 
                           font=("Arial", 10), justify="left", anchor="nw")
    result_label.pack(side="left", padx=10)

    # Timing display
    timing_label = tk.Label(results_frame, text="Zaman ölçmələri burada göstəriləcək", 
                           font=("Arial", 10, "bold"), justify="left", anchor="nw")
    timing_label.pack(side="right", padx=10)

    # Plot section
    plot_frame = tk.Frame(window)
    plot_frame.pack(pady=10, fill="both", expand=True)

    # Instructions
    info_label = tk.Label(window, 
                         text="Omega₀ daxil edin və müxtəlif metodları test edin. Multiprocessing ən yaxşı performans göstərəcək.",
                         font=("Arial", 9), fg="blue")
    info_label.pack(pady=5)

    window.mainloop()

if __name__ == "__main__":
    # Ensure multiprocessing works properly on Windows
    multiprocessing.freeze_support()
    create_gui()