import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Qlobal dəyişənlər
x = None
y = None

# Siqnal faylını oxuma funksiyası
def load_signal_file():
    filepath = filedialog.askopenfilename(title="Siqnal Faylını Seç", filetypes=[("Text Files", "*.txt")])
    if filepath:
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()[1:]  # başlığı atırıq
                x_vals = []
                y_vals = []
                for line in lines:
                    t_val, y_val = map(float, line.strip().split(','))
                    x_vals.append(t_val)
                    y_vals.append(y_val)
            return np.array(x_vals), np.array(y_vals)
        except Exception as e:
            messagebox.showerror("Xəta", f"Fayl oxunarkən xəta baş verdi: {e}")
            return None, None
    return None, None

# Fourier əmsallarını hesabla
def calculate_fourier(x, y, nmax=10):
    """
    Computes the Fourier coefficients a0, an, bn for the signal y over time x.

    Parameters:
    - x: numpy array of time values
    - y: numpy array of signal values
    - nmax: number of harmonics to compute

    Returns:
    - a0: constant term
    - an: list of cosine coefficients
    - bn: list of sine coefficients
    """
    T = x[-1] - x[0]  # Period (assumes x covers one full period)
    omega_0 = 2 * np.pi / T

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
# Nəticələri fayla yaz
def write_results_to_file(a0, an, bn):
    try:
        output_path = "coefficients.txt"
        with open(output_path, "w") as f:
            f.write(f"a0 = {a0:.6f}\n")
            for i in range(len(an)):
                f.write(f"a{i+1} = {an[i]:.6f}, b{i+1} = {bn[i]:.6f}\n")
        return output_path
    except Exception as e:
        messagebox.showerror("Xəta", f"Fayla yazarkən xəta baş verdi: {e}")
        return None

# Qrafik göstərmə funksiyası
# Qrafik göstərmə funksiyası (GUI içində)
def plot_signal():
    global x, y, plot_frame

    if x is not None and y is not None:
        for widget in plot_frame.winfo_children():
            widget.destroy()  # Əvvəlki qrafiki təmizlə

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, y, label="Siqnal", color="blue")
        ax.set_xlabel("Zaman (t)")
        ax.set_ylabel("Siqnal (y)")
        ax.set_title("Siqnal Qrafiki")
        ax.grid(True)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    else:
        messagebox.showwarning("Diqqət", "Əvvəlcə siqnal faylını yükləyin.")

# Yükləmə düyməsi funksiyası
def on_load_button_click():
    global x, y
    x, y = load_signal_file()
    if x is not None and y is not None:
        a0, an, bn = calculate_fourier(x, y)
        result_text = f"a0 = {a0:.6f}\n"
        for i in range(len(an)):
            result_text += f"a{i+1} = {an[i]:.6f}, b{i+1} = {bn[i]:.6f}\n"
        result_label.config(text=result_text)

# Nəticələri fayla yazmaq funksiyası
def on_save_button_click():
    try:
        result_text = result_label.cget("text")
        lines = result_text.strip().split('\n')

        if not lines or len(lines) < 2:
            messagebox.showwarning("Diqqət", "Əvvəlcə nəticələri hesablamaq lazımdır.")
            return

        a0 = float(lines[0].split('=')[1].strip())

        an = []
        bn = []
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) != 2:
                continue
            a_str, b_str = parts
            an_val = float(a_str.split('=')[1].strip())
            bn_val = float(b_str.split('=')[1].strip())
            an.append(an_val)
            bn.append(bn_val)

        output_path = write_results_to_file(a0, an, bn)
        if output_path:
            messagebox.showinfo("Məlumat", f"Nəticələr fayla yazıldı: {output_path}")

    except Exception as e:
        messagebox.showerror("Xəta", f"Xəta baş verdi: {e}")

# GUI interfeysi
def create_gui():
    global result_label, plot_frame

    window = tk.Tk()
    window.title("Fourier Transform")

    load_button = ttk.Button(window, text="Siqnal Faylını Yüklə", command=on_load_button_click)
    load_button.pack(pady=10)

    result_label = tk.Label(window, text="Nəticələr burada göstəriləcək", font=("Arial", 12), justify="left")
    result_label.pack(pady=10)

    save_button = ttk.Button(window, text="Nəticələri Fayla Yaz", command=on_save_button_click)
    save_button.pack(pady=10)

    plot_button = ttk.Button(window, text="Siqnal Qrafikini Göstər", command=plot_signal)
    plot_button.pack(pady=10)

    plot_frame = tk.Frame(window)  # Qrafik üçün yer
    plot_frame.pack(pady=10)

    window.mainloop()

# GUI-ni başlat
create_gui()