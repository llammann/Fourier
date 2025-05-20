import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# signal.txt faylını sabit yerdən oxuma funksiyası
def load_signal_file(): 
    try:
        with open("../signal.txt", "r") as f:
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

# Fourier əmsallarını hesabla
def calculate_fourier(x, y, omega_0, nmax=10):
    T = x[-1] - x[0]  # Period
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

# GUI update funksiyası - omega_0 daxil ediləndə çağrılır
def on_omega_change(*args):
    omega_str = omega_var.get()
    if not omega_str:
        result_label.config(text="Omega_0 daxil edin.")
        return
    try:
        omega_0 = float(omega_str)
        if omega_0 <= 0:
            raise ValueError("Omega_0 müsbət olmalıdır.")
    except Exception as e:
        result_label.config(text=f"Xəta: {e}")
        return
    
    x, y = load_signal_file()
    if x is None or y is None:
        result_label.config(text="Siqnal faylı oxunmadı.")
        return
    
    a0, an, bn = calculate_fourier(x, y, omega_0)
    text = f"a0 = {a0:.6f}\n"
    for i in range(len(an)):
        text += f"a{i+1} = {an[i]:.6f}, b{i+1} = {bn[i]:.6f}\n"
    result_label.config(text=text)

# GUI yaratmaq
def create_gui():
    global omega_var, result_label

    window = tk.Tk()
    window.title("Fourier Analizi")

    ttk.Label(window, text="Omega_0 dəyərini daxil edin:").pack(pady=5)

    omega_var = tk.StringVar()
    omega_var.trace_add("write", on_omega_change)
    omega_entry = ttk.Entry(window, textvariable=omega_var)
    omega_entry.pack(pady=5)

    result_label = tk.Label(window, text="Nəticələr burada göstəriləcək", font=("Arial", 12), justify="left")
    result_label.pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    create_gui()
