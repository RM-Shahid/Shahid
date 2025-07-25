#!/usr/bin/env python
# coding: utf-8

# In[10]:


pip install pandas numpy scikit-learn matplotlib


# In[29]:


import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class StylishDataCleaningGUI:
    def __init__(self, master):
        bg_color = "#e3f2fd"  

        self.master = master
        self.master.title("Data Cleaning Dashboard")
        self.master.geometry("1200x800")
        self.master.configure(bg=bg_color)

        self.train_df = None
        self.test_df = None
        self.cleaned_train_df = None
        self.cleaned_test_df = None
        self.exclude_vars = []
        self.scaler_option = tk.StringVar(value="Min-Max")
        self.plot_var = tk.StringVar()

        header_font = ("Segoe UI", 16, "bold")
        label_font = ("Segoe UI", 11)

        tk.Label(master, text="Data Cleaning GUI", font=header_font, bg=bg_color).pack(pady=10)

        self.frame_top = tk.Frame(master, bg=bg_color)
        self.frame_top.pack(pady=5)

        self.frame_bottom = tk.Frame(master, bg=bg_color)
        self.frame_bottom.pack(fill="both", expand=True)

        self.frame_controls = tk.Frame(self.frame_bottom, bg=bg_color)
        self.frame_controls.pack(side="left", fill="y", padx=10)

        self.frame_output = tk.Frame(self.frame_bottom, bg="white", relief="sunken", bd=2)
        self.frame_output.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ttk.Button(self.frame_top, text="Upload Train Data", command=self.upload_train).pack(side="left", padx=10)
        ttk.Button(self.frame_top, text="Upload Test Data", command=self.upload_test).pack(side="left", padx=10)

        tk.Label(self.frame_controls, text="Scaling Method:", font=label_font, bg=bg_color).pack(pady=(15, 5))
        ttk.Combobox(self.frame_controls, textvariable=self.scaler_option, values=["Min-Max", "Standard"]).pack()

        ttk.Button(self.frame_controls, text="Clean & Describe", command=self.clean_data).pack(pady=10)

        tk.Label(self.frame_controls, text="Exclude Variables:", font=label_font, bg=bg_color).pack(pady=(10, 5))
        self.exclude_listbox = tk.Listbox(self.frame_controls, selectmode="multiple", height=6, exportselection=False)
        self.exclude_listbox.pack()

        tk.Label(self.frame_controls, text="Select Variable:", font=label_font, bg=bg_color).pack(pady=(15, 5))
        self.plot_combo = ttk.Combobox(self.frame_controls, textvariable=self.plot_var)
        self.plot_combo.pack()

        ttk.Button(self.frame_controls, text="Time Series Plot", command=self.plot_variable).pack(pady=(10, 5))
        ttk.Button(self.frame_controls, text="Box Plots", command=self.plot_boxplot).pack(pady=(5, 5))

        ttk.Button(self.frame_controls, text="Save Cleaned Train", command=self.save_cleaned_train).pack(pady=(10, 5))
        ttk.Button(self.frame_controls, text="Save Cleaned Test", command=self.save_cleaned_test).pack(pady=(0, 15))

        self.log_text = tk.Text(self.frame_controls, height=12, width=40)
        self.log_text.pack(pady=5)
        self.log("Welcome! Upload CSV or Excel files.")

        self.output_text = tk.Text(self.frame_output, wrap="none")
        self.output_text.pack(fill="both", expand=True)

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def output(self, title, df: pd.DataFrame):
        desc = df.describe().T.drop(columns=["count"], errors='ignore').round(3)
        self.output_text.insert(tk.END, f"\n--- {title} ---\n")
        self.output_text.insert(tk.END, desc.to_string())
        self.output_text.insert(tk.END, "\n\n")
        self.output_text.see(tk.END)

    def update_variable_lists(self):
        if self.train_df is not None:
            num_vars = list(self.train_df.select_dtypes(include=[np.number]).columns)
            self.exclude_listbox.delete(0, tk.END)
            for var in num_vars:
                self.exclude_listbox.insert(tk.END, var)
            self.plot_combo["values"] = num_vars

    def upload_train(self):
        path = filedialog.askopenfilename(filetypes=[("CSV and Excel", "*.csv *.xlsx *.xls")])
        if path:
            try:
                self.train_df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
                self.log(f"Train loaded: {path} | Shape: {self.train_df.shape}")
                self.update_variable_lists()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load train data.\n{str(e)}")

    def upload_test(self):
        path = filedialog.askopenfilename(filetypes=[("CSV and Excel", "*.csv *.xlsx *.xls")])
        if path:
            try:
                self.test_df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
                self.log(f"Test loaded: {path} | Shape: {self.test_df.shape}")
                self.update_variable_lists()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load test data.\n{str(e)}")

    def get_excluded_variables(self):
        return [self.exclude_listbox.get(i) for i in self.exclude_listbox.curselection()]

    def clean_data(self):
        self.output_text.delete("1.0", tk.END)
        self.exclude_vars = self.get_excluded_variables()

        if self.train_df is not None:
            self.log("Cleaning train data...")
            _, after = self.process(self.train_df)
            self.cleaned_train_df = after
            self.output("Train Data (Cleaned)", after)

        if self.test_df is not None:
            self.log("Cleaning test data...")
            _, after = self.process(self.test_df)
            self.cleaned_test_df = after
            self.output("Test Data (Cleaned)", after)

    def process(self, df):
        df_numeric = df.select_dtypes(include=[np.number]).copy()
        df_to_clean = df_numeric.drop(columns=self.exclude_vars, errors="ignore")
        df_excluded = df_numeric[self.exclude_vars] if self.exclude_vars else pd.DataFrame(index=df_numeric.index)

        imputer = KNNImputer(n_neighbors=3)
        df_imputed = pd.DataFrame(imputer.fit_transform(df_to_clean), columns=df_to_clean.columns)

        Q1 = df_imputed.quantile(0.25)
        Q3 = df_imputed.quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((df_imputed < (Q1 - 1.5 * IQR)) | (df_imputed > (Q3 + 1.5 * IQR))).any(axis=1)
        df_no_outliers = df_imputed[mask]

        scaler = MinMaxScaler() if self.scaler_option.get() == "Min-Max" else StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_no_outliers), columns=df_to_clean.columns, index=df_no_outliers.index)

        df_excluded_clean = df_excluded.reindex(df_scaled.index)
        df_final = pd.concat([df_scaled, df_excluded_clean], axis=1)

        return df_numeric, df_final

    def plot_variable(self):
        var = self.plot_var.get()
        if not var:
            messagebox.showinfo("Select Variable", "Please select a variable to plot.")
            return

        if self.cleaned_train_df is None and self.cleaned_test_df is None:
            messagebox.showinfo("Clean First", "Please clean the data first.")
            return

        win = tk.Toplevel(self.master)
        win.title(f"📈 Time Series Plot - {var}")
        win.geometry("2000x400")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        labels = ["Train - After Cleaning", "Test - After Cleaning"]
        datasets = [
            self.cleaned_train_df,
            self.cleaned_test_df
        ]

        for ax, label, df in zip(axes, labels, datasets):
            if df is not None and var in df.columns:
                ax.plot(df[var].dropna().values)
                ax.set_title(label)
                ax.set_xlabel("Index")
                ax.set_ylabel(var)
                ax.grid(True)
            else:
                ax.set_title(label + " (No data)")
                ax.axis("off")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def plot_boxplot(self):
        var = self.plot_var.get()
        if not var:
            messagebox.showinfo("Select Variable", "Please select a variable.")
            return

        if self.cleaned_train_df is None and self.cleaned_test_df is None:
            messagebox.showinfo("Clean First", "Please clean the data first.")
            return

        win = tk.Toplevel(self.master)
        win.title(f"Box Plots - {var}")
        win.geometry("1200x400")

        fig, axes = plt.subplots(1, 4, figsize=(12, 4))

        labels = ["Train Before", "Train After", "Test Before", "Test After"]
        datasets = [
            self.train_df[var].dropna() if self.train_df is not None and var in self.train_df else None,
            self.cleaned_train_df[var].dropna() if self.cleaned_train_df is not None and var in self.cleaned_train_df else None,
            self.test_df[var].dropna() if self.test_df is not None and var in self.test_df else None,
            self.cleaned_test_df[var].dropna() if self.cleaned_test_df is not None and var in self.cleaned_test_df else None
        ]

        for i, (data, label) in enumerate(zip(datasets, labels)):
            ax = axes[i]
            if data is not None and len(data) > 0:
                ax.boxplot(data, patch_artist=True)
                ax.set_title(label)
                ax.set_ylabel("Value")
                ax.grid(True)
            else:
                ax.axis('off')

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def save_cleaned_train(self):
        if self.cleaned_train_df is None:
            messagebox.showinfo("No Data", "Please clean the train data first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            self.cleaned_train_df.to_csv(path, index=False)
            self.log(f"Saved cleaned train data to: {path}")

    def save_cleaned_test(self):
        if self.cleaned_test_df is None:
            messagebox.showinfo("No Data", "Please clean the test data first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            self.cleaned_test_df.to_csv(path, index=False)
            self.log(f"Saved cleaned test data to: {path}")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = StylishDataCleaningGUI(root)
    root.mainloop()


# In[ ]:




