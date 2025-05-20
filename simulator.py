import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class FourierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulator, Proqramci: Ləman Nəzirli")

        Label(root, text="Simulator", font=("Helvetica", 16)).grid(row=0, columnspan=4, pady=20, padx=10, sticky=W)

        Label(root, text="a0 dəyəri:").grid(row=1, column=0, sticky=W, padx=10)
        self.a0_entry = Entry(root)
        self.a0_entry.insert(0, "0")
        self.a0_entry.grid(row=1, column=1)

        Label(root, text="ω₀ dəyəri:").grid(row=2, column=0, sticky=W, padx=10)
        self.w0_entry = Entry(root)
        self.w0_entry.insert(0, "0")
        self.w0_entry.grid(row=2, column=1)

        Label(root, text="n_max dəyəri:").grid(row=3, column=0, sticky=W, padx=10)
        self.n_max_var = StringVar()
        self.n_max_var.trace_add("write", self.update_entries_live)
        self.n_max_entry = Entry(root, textvariable=self.n_max_var)
        self.n_max_entry.insert(0, "0")
        self.n_max_entry.grid(row=3, column=1)

        Label(root, text="N dəyəri:").grid(row=4, column=0, sticky=W, padx=10)
        self.n_var = StringVar()
        self.n_entry = Entry(root, textvariable=self.n_var, state='readonly')
        self.n_entry.insert(0, "0")
        self.n_entry.grid(row=4, column=1)

        self.a_entries = []
        self.b_entries = []
        self.labels = []

        Label(root, text="").grid(row=99, column=0, pady=10)

        self.plot_frame = Frame(root)
        self.plot_frame.grid(row=100, column=0, columnspan=4)

        self.canvas = None

        # Automatically update on input
        self.a0_entry.bind("<KeyRelease>", lambda e: self.safe_calculate())
        self.w0_entry.bind("<KeyRelease>", lambda e: self.safe_calculate())

        self.safe_calculate()

    def safe_calculate(self):
        try:
            self.calculate_fourier()
        except Exception:
            pass  # Ignore input errors during typing

    def update_entries_live(self, *args):
        value = self.n_max_var.get()
        if value.isdigit():
            self.n_var.set(str(value))
            self.add_entries(int(value))
            self.safe_calculate()

    def add_entries(self, N):
        max_current = len(self.a_entries)
        row = 6

        for i in range(N):
            if i < max_current:
                self.a_entries[i].grid(row=row, column=1)
                self.b_entries[i].grid(row=row, column=3)
                self.labels[i * 2].grid(row=row, column=0)
                self.labels[i * 2 + 1].grid(row=row, column=2)
            else:
                a_label = Label(self.root, text=f"a{i + 1}:")
                a_label.grid(row=row, column=0, padx=10)
                a_entry = Entry(self.root)
                a_entry.insert(0, "0")
                a_entry.grid(row=row, column=1)
                a_entry.bind("<KeyRelease>", lambda e: self.safe_calculate())

                b_label = Label(self.root, text=f"b{i + 1}:")
                b_label.grid(row=row, column=2, padx=10)
                b_entry = Entry(self.root)
                b_entry.insert(0, "0")
                b_entry.grid(row=row, column=3)
                b_entry.bind("<KeyRelease>", lambda e: self.safe_calculate())

                self.a_entries.append(a_entry)
                self.b_entries.append(b_entry)
                self.labels.extend([a_label, b_label])
            row += 1

        for i in range(N, max_current):
            self.a_entries[i].grid_remove()
            self.b_entries[i].grid_remove()
            self.labels[i * 2].grid_remove()
            self.labels[i * 2 + 1].grid_remove()

    def calculate_fourier(self):
        a0 = float(self.a0_entry.get())
        w0 = float(self.w0_entry.get())
        n_max = int(self.n_max_entry.get())
        N = n_max
        self.n_entry.delete(0, END)
        self.n_entry.insert(0, str(N))

        if N > len(self.a_entries):
            self.add_entries(N)

        a = [float(e.get()) for e in self.a_entries[:N]]
        b = [float(e.get()) for e in self.b_entries[:N]]

        x = np.linspace(0, 2 * np.pi, 1000)
        y = np.full_like(x, a0 / 2)

        for n in range(1, N + 1):
            y += a[n - 1] * np.cos(n * w0 * x) + b[n - 1] * np.sin(n * w0 * x)

        if np.max(np.abs(y)) > 10:
            messagebox.showerror("Error", "The signal seems too large or incorrect.")
            return

        os.makedirs("dist", exist_ok=True)
        with open("dist/signal.txt", "w") as f:
            f.write("t, X(t)\n")
            for xi, yi in zip(x, y):
                f.write(f"{xi:.4f}, {yi:.4f}\n")

        self.plot_fourier(x, y)

    def plot_fourier(self, x, y):
        fig = plt.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(x, y, label="Harmonik Siqnal")
        ax.set_title("Harmonik Siqnal")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True)
        ax.legend()

        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

if __name__ == '__main__':
    root = Tk()
    app = FourierApp(root)
    root.mainloop()
