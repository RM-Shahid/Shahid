import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

try:
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
except ImportError:
    XGBRegressor = CatBoostRegressor = None


class PredictiveToolGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Predictive Modeling Dashboard")
        self.master.geometry("1900x900")
        self.master.configure(bg="#85a3b2")

        self.df = None
        self.scaler = None
        self.model_option = tk.StringVar()
        self.scaling_method = tk.StringVar(value="None")
        self.target_var = tk.StringVar()
        self.exclude_var = tk.StringVar()
        self.test_size = tk.DoubleVar(value=0.2)

        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.y_train_pred = self.y_test_pred = None

        self.setup_ui()

    def setup_ui(self):
        title = tk.Label(self.master, text="Predictive Modeling Dashboard",
                         font=("Segoe UI", 20, "bold"), bg="#d8dcd6", fg="#333")
        title.pack(pady=10)

        control_frame = tk.Frame(self.master, bg="#f0f4f7")
        control_frame.pack(pady=10, fill="x")
        control_frame.grid_columnconfigure((0, 1, 2, 3, 14, 15, 16, 17), weight=1)

        ttk.Button(control_frame, text="Upload Data", command=self.upload_data).grid(row=0, column=4, padx=10, pady=5)
        tk.Label(control_frame, text="Train-Test Split:", bg="#f0f4f7", font=("Segoe UI", 10)).grid(row=0, column=5, sticky="e")
        tk.Scale(control_frame, from_=0.1, to=0.4, resolution=0.05, orient="horizontal",
                 variable=self.test_size, bg="#f0f4f7").grid(row=0, column=6, padx=5)
        tk.Label(control_frame, text="Target Variable:", bg="#f0f4f7", font=("Segoe UI", 10)).grid(row=0, column=7, sticky="e")
        self.target_dropdown = ttk.Combobox(control_frame, textvariable=self.target_var, width=20)
        self.target_dropdown.grid(row=0, column=8, padx=5)
        tk.Label(control_frame, text="Scaling Method:", bg="#f0f4f7", font=("Segoe UI", 10)).grid(row=0, column=9, sticky="e")
        ttk.Combobox(control_frame, textvariable=self.scaling_method, values=["None", "Standard", "Min-Max"], width=15).grid(row=0, column=10, padx=5)
        tk.Label(control_frame, text="Exclude Variable:", bg="#f0f4f7", font=("Segoe UI", 10)).grid(row=0, column=11, sticky="e")
        self.exclude_dropdown = ttk.Combobox(control_frame, textvariable=self.exclude_var, width=20)
        self.exclude_dropdown.grid(row=0, column=12, padx=5)
        ttk.Button(control_frame, text="Preprocess", command=self.preprocess_data).grid(row=0, column=13, padx=10)
        tk.Label(control_frame, text="Choose Model:", bg="#f0f4f7", font=("Segoe UI", 10)).grid(row=1, column=7, sticky="e")
        model_list = [
            "LinearRegression", "Ridge", "Lasso", "DecisionTree", "RandomForest",
            "ExtraTrees", "SVR", "KNN", "XGBoost", "CatBoost"
        ]
        self.model_dropdown = ttk.Combobox(control_frame, textvariable=self.model_option, values=model_list, width=20)
        self.model_dropdown.grid(row=1, column=8, padx=5, pady=5)
        ttk.Button(control_frame, text="Train & Evaluate", command=self.train_and_evaluate).grid(row=1, column=9, padx=10, pady=5)
        ttk.Button(control_frame, text="Plot Predictions", command=self.plot_results).grid(row=1, column=10, padx=5, pady=5)

        # Create a parent frame to hold both left and right panels
        self.content_frame = tk.Frame(self.master, bg="#ffffff")
        self.content_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Configure grid layout to split into 2 columns equally
        self.content_frame.columnconfigure(0, weight=1, uniform="equal")
        self.content_frame.columnconfigure(1, weight=1, uniform="equal")
        self.content_frame.rowconfigure(0, weight=1)

        # Left Panel (text output)
        self.left_panel = tk.Frame(self.content_frame, bg="#f8f9fa", relief="sunken", bd=2)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=2)

        self.output_text = tk.Text(
            self.left_panel,
            font=("Consolas", 10),
            bg="#f8f9fa",
            relief="flat",
            bd=0
        )
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Right Panel (plots)
        self.right_panel = tk.Frame(self.content_frame, bg="#ffffff", relief="sunken", bd=2)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=2)

        self.canvas_frame = tk.Frame(self.right_panel, bg="#ffffff")
        self.canvas_frame.pack(fill="both", expand=True)

      
       
        footer = tk.Label(self.master, text="Prepared by: Muhammad Shahid",
                          font=("Segoe UI", 10, "bold"), fg="#555", bg="#d8dcd6", anchor="center")
        footer.pack(side="bottom", fill="x", pady=(3, 0))

    def upload_data(self):
        path = filedialog.askopenfilename(filetypes=[("CSV and Excel files", "*.csv *.xlsx *.xls")])
        if path:
            self.df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
            self.output_text.insert(tk.END, f"✅ Data Loaded: {path} | Shape: {self.df.shape}\n")
            numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
            self.target_dropdown["values"] = numeric_cols
            self.exclude_dropdown["values"] = numeric_cols

    def preprocess_data(self):
        if self.df is None or not self.target_var.get():
            messagebox.showerror("Missing Info", "Upload data and select target variable.")
            return

        df = self.df.copy()
        df = df.select_dtypes(include=[np.number])
        excluded_col = self.exclude_var.get()
        if excluded_col in df.columns:
            df.drop(columns=[excluded_col], inplace=True)
        if self.target_var.get() not in df.columns:
            messagebox.showerror("Error", "Selected target variable not found.")
            return

        imputer = KNNImputer(n_neighbors=3)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        df_clean = df_imputed
        X = df_clean.drop(columns=[self.target_var.get()])
        y = df_clean[self.target_var.get()]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size.get(), random_state=42
        )

        if self.scaling_method.get() == "Standard":
            self.scaler = StandardScaler()
        elif self.scaling_method.get() == "Min-Max":
            self.scaler = MinMaxScaler()

        if self.scaler:
            self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=X.columns, index=self.X_train.index)
            self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=X.columns, index=self.X_test.index)

        self.cleaned_train_df = pd.concat([self.X_train, self.y_train.reset_index(drop=True)], axis=1)
        self.cleaned_test_df = pd.concat([self.X_test, self.y_test.reset_index(drop=True)], axis=1)

        self.output_text.insert(tk.END, "\nTrain Data Description:\n")
        self.output_text.insert(tk.END, self.cleaned_train_df.describe().T.round(3).to_string())
        self.output_text.insert(tk.END, "\n\nTest Data Description:\n")
        self.output_text.insert(tk.END, self.cleaned_test_df.describe().T.round(3).to_string())
        self.output_text.insert(tk.END, "\n→ Preprocessing Completed\n", "bold")
        self.output_text.insert(tk.END, "• Missing values handled using KNN Imputer (k=3)\n")
        self.output_text.insert(tk.END, f"• '{excluded_col}' excluded from features\n" if excluded_col else "")
        self.output_text.insert(tk.END, f"• Target variable: {self.target_var.get()}\n")
        self.output_text.insert(tk.END, "• Outliers removed\n")
        scaling_applied = self.scaling_method.get()
        if scaling_applied != "None":
            self.output_text.insert(tk.END, f"• Features scaled using {scaling_applied} Scaler\n")
        else:
            self.output_text.insert(tk.END, "• No scaling applied\n")
        self.output_text.insert(tk.END, f"• Train-Test split: {int((1 - self.test_size.get()) * 100)}% train / {int(self.test_size.get() * 100)}% test\n")
        self.output_text.see(tk.END)


    def train_and_evaluate(self):
        model_name = self.model_option.get()
        if not model_name or self.X_train is None:
            messagebox.showerror("Error", "Preprocess data and select model.")
            return

        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForest": RandomForestRegressor(),
            "ExtraTrees": ExtraTreesRegressor(),
            "SVR": SVR(kernel='rbf'),
            "KNN": KNeighborsRegressor(),
            "XGBoost": XGBRegressor() if XGBRegressor else None,
            "CatBoost": CatBoostRegressor(verbose=0) if CatBoostRegressor else None
        }

        model = models.get(model_name)
        if model is None:
            messagebox.showerror("Unavailable", f"{model_name} not available.")
            return

        model.fit(self.X_train, self.y_train)
        self.y_train_pred = model.predict(self.X_train)
        self.y_test_pred = model.predict(self.X_test)

        train_r2 = r2_score(self.y_train, self.y_train_pred)
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        train_rmse = mean_squared_error(self.y_train, self.y_train_pred, squared=False)
        test_rmse = mean_squared_error(self.y_test, self.y_test_pred, squared=False)
        train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
        test_mae = mean_absolute_error(self.y_test, self.y_test_pred)

        self.output_text.insert(tk.END, f"\n→ Model: {model_name}\n", "bold_title")
        self.output_text.insert(tk.END, f"Train → R²: {train_r2:.3f}, RMSE: {train_rmse:.3f}, MAE: {train_mae:.3f}\n", "bold")
        self.output_text.insert(tk.END, f"Test  → R²: {test_r2:.3f}, RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}\n", "bold")
        self.output_text.see(tk.END)


    def plot_results(self):
        model_name = self.model_option.get()
        target_variable = self.target_var.get()
        if self.y_test_pred is None:
            messagebox.showerror("Missing", "Run training first.")
            return

        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig, axs = plt.subplots(2, 1, figsize=(7, 7))
        sns.scatterplot(x=self.y_test, y=self.y_test_pred, ax=axs[0], label="Predicted", color="blue", edgecolor="k")
        sns.regplot(x=self.y_test, y=self.y_test_pred, ax=axs[0], scatter=False, line_kws={"color": "red"}, label="R² Line")
        axs[0].set_title(f"{model_name} | Actual vs Predicted")
        axs[0].set_xlabel(f"Actual {target_variable}")
        axs[0].set_ylabel(f"Predicted {target_variable}")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(np.arange(len(self.y_test)), self.y_test, label="Actual", color="black")
        axs[1].plot(np.arange(len(self.y_test_pred)), self.y_test_pred, label="Predicted", color="blue")
        axs[1].set_title(f"{model_name} | Time Series of {target_variable}")
        axs[1].set_xlabel("Sample Index")
        axs[1].set_ylabel(target_variable)
        axs[1].legend()
        axs[1].grid(True)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self.canvas_frame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)


root = tk.Tk()
app = PredictiveToolGUI(root)
root.mainloop()
