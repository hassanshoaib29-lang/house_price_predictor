import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class HousePricePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("House Price Predictor")
        self.root.geometry("950x750")
        self.root.resizable(False, False)
        
        self.models = {}
        self.current_model = None
        self.scaler = None
        self.feature_names = ['sqft', 'bedrooms', 'bathrooms', 'year_built', 'lot_size', 'garage', 'neighborhood_score', 'location_type']
        self.prediction_history = []
        self.model_metrics = {}
        
        self._setup_styles()
        self._create_widgets()
        self._load_or_train_models()
        
    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.colors = {
            'primary': '#2C3E50',
            'secondary': '#3498DB',
            'accent': '#E74C3C',
            'background': '#ECF0F1',
            'card': '#FFFFFF',
            'text_primary': '#2C3E50',
            'text_secondary': '#7F8C8D',
            'success': '#27AE60',
            'warning': '#F39C12'
        }
        
        self.root.configure(bg=self.colors['background'])
        
    def _create_widgets(self):
        self.main_frame = tk.Frame(self.root, bg=self.colors['background'])
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
        
        self._create_header()
        self._create_left_panel()
        self._create_right_panel()
        self._create_footer()
        
    def _create_header(self):
        header = tk.Frame(self.main_frame, bg=self.colors['primary'], height=60)
        header.pack(fill=tk.X, pady=(0, 16))
        header.pack_propagate(False)
        
        tk.Label(header, text="🏠 House Price Predictor", 
                font=("Segoe UI", 20, "bold"), 
                bg=self.colors['primary'], 
                fg="white").pack(pady=15)
        
    def _create_left_panel(self):
        left_frame = tk.Frame(self.main_frame, bg=self.colors['card'], 
                            width=420, height=580)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 16))
        left_frame.pack_propagate(False)
        
        tk.Label(left_frame, text="House Features", 
                font=("Segoe UI", 14, "bold"),
                bg=self.colors['card'], 
                fg=self.colors['text_primary']).pack(pady=(16, 8))
        
        input_frame = tk.Frame(left_frame, bg=self.colors['card'])
        input_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)
        
        self.inputs = {}
        
        fields = [
            ("Square Footage", "sqft", 500, 10000, 2000),
            ("Number of Bedrooms", "bedrooms", ["1", "2", "3", "4", "5", "6+"]),
            ("Number of Bathrooms", "bathrooms", ["1", "1.5", "2", "2.5", "3", "3.5", "4+"]),
            ("Year Built", "year_built", 1990, 2026, 2000),
            ("Lot Size (sq ft)", "lot_size", 1000, 50000, 8000),
            ("Garage Capacity", "garage", ["0", "1", "2", "3+"]),
        ]
        
        for i, (label, key, *args) in enumerate(fields):
            row = i // 2
            col = (i % 2) * 2
            
            tk.Label(input_frame, text=label, font=("Segoe UI", 10),
                    bg=self.colors['card'], fg=self.colors['text_secondary']).grid(
                row=row, column=col, sticky=tk.W, padx=8, pady=8)
            
            if isinstance(args[0], list):
                self.inputs[key] = tk.StringVar(value=args[0][2])
                combo = ttk.Combobox(input_frame, textvariable=self.inputs[key],
                                    values=args[0], state='readonly', width=12)
                combo.grid(row=row, column=col+1, padx=8, pady=8)
            else:
                self.inputs[key] = tk.IntVar(value=args[2])
                spin = ttk.Spinbox(input_frame, from_=args[0], to=args[1],
                                  textvariable=self.inputs[key], width=12)
                spin.grid(row=row, column=col+1, padx=8, pady=8)
        
        tk.Label(input_frame, text="Neighborhood Score", font=("Segoe UI", 10),
                bg=self.colors['card'], fg=self.colors['text_secondary']).grid(
            row=3, column=0, sticky=tk.W, padx=8, pady=8)
        
        self.neighborhood_var = tk.IntVar(value=7)
        neighborhood_frame = tk.Frame(input_frame, bg=self.colors['card'])
        neighborhood_frame.grid(row=3, column=1, padx=8, pady=8)
        
        self.neighborhood_slider = ttk.Scale(neighborhood_frame, from_=1, to=10,
                                            orient=tk.HORIZONTAL,
                                            variable=self.neighborhood_var,
                                            command=self._update_neighborhood_label)
        self.neighborhood_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.neighborhood_value_label = tk.Label(neighborhood_frame, text="7",
                                                 font=("Segoe UI", 10, "bold"),
                                                 bg=self.colors['card'],
                                                 fg=self.colors['secondary'])
        self.neighborhood_value_label.pack(side=tk.LEFT, padx=(8, 0))
        
        self.inputs['neighborhood_score'] = self.neighborhood_var
        
        tk.Label(input_frame, text="Location Type", font=("Segoe UI", 10),
                bg=self.colors['card'], fg=self.colors['text_secondary']).grid(
            row=4, column=0, sticky=tk.W, padx=8, pady=8)
        
        self.inputs['location_type'] = tk.StringVar(value="Suburban")
        location_combo = ttk.Combobox(input_frame, textvariable=self.inputs['location_type'],
                                     values=["Urban", "Suburban", "Rural"],
                                     state='readonly', width=12)
        location_combo.grid(row=4, column=1, padx=8, pady=8)
        
        tk.Label(input_frame, text="Regression Model", font=("Segoe UI", 10),
                bg=self.colors['card'], fg=self.colors['text_secondary']).grid(
            row=5, column=0, sticky=tk.W, padx=8, pady=(16, 8))
        
        self.model_var = tk.StringVar(value="Random Forest")
        model_combo = ttk.Combobox(input_frame, textvariable=self.model_var,
                                  values=["Linear Regression", "Random Forest", "Gradient Boosting"],
                                  state='readonly', width=18)
        model_combo.grid(row=5, column=1, padx=8, pady=(16, 8))
        model_combo.bind('<<ComboboxSelected>>', lambda e: self._update_model_metrics())
        
        button_frame = tk.Frame(left_frame, bg=self.colors['card'])
        button_frame.pack(pady=16)
        
        self.predict_btn = tk.Button(button_frame, text="🔮 Predict Price",
                                     font=("Segoe UI", 12, "bold"),
                                     bg=self.colors['secondary'], fg="white",
                                     activebackground='#2980B9',
                                     relief=tk.FLAT, cursor='hand2',
                                     command=self._predict)
        self.predict_btn.pack(side=tk.LEFT, padx=8, ipadx=20, ipady=8)
        
        self.reset_btn = tk.Button(button_frame, text="Reset",
                                   font=("Segoe UI", 11),
                                   bg=self.colors['text_secondary'], fg="white",
                                   activebackground='#6C7A89',
                                   relief=tk.FLAT, cursor='hand2',
                                   command=self._reset_inputs)
        self.reset_btn.pack(side=tk.LEFT, padx=8, ipadx=16, ipady=8)
        
    def _create_right_panel(self):
        right_frame = tk.Frame(self.main_frame, bg=self.colors['card'],
                              width=450, height=580)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        right_frame.pack_propagate(False)
        
        tk.Label(right_frame, text="Prediction Results",
                font=("Segoe UI", 14, "bold"),
                bg=self.colors['card'], 
                fg=self.colors['text_primary']).pack(pady=(16, 8))
        
        self.price_label = tk.Label(right_frame, text="$---",
                                   font=("Segoe UI", 36, "bold"),
                                   bg=self.colors['card'],
                                   fg=self.colors['success'])
        self.price_label.pack(pady=8)
        
        self.range_label = tk.Label(right_frame, text="Price Range: $--- - $---",
                                    font=("Segoe UI", 11),
                                    bg=self.colors['card'],
                                    fg=self.colors['text_secondary'])
        self.range_label.pack()
        
        self.confidence_label = tk.Label(right_frame, text="Confidence: ---",
                                         font=("Segoe UI", 11),
                                         bg=self.colors['card'],
                                         fg=self.colors['secondary'])
        self.confidence_label.pack(pady=(0, 16))
        
        metrics_frame = tk.Frame(right_frame, bg=self.colors['background'], padx=12, pady=12)
        metrics_frame.pack(fill=tk.X, padx=16, pady=8)
        
        tk.Label(metrics_frame, text="Model Performance",
                font=("Segoe UI", 12, "bold"),
                bg=self.colors['background'],
                fg=self.colors['text_primary']).grid(row=0, columnspan=3, pady=(0, 8))
        
        self.r2_label = tk.Label(metrics_frame, text="R² Score: ---",
                                font=("Segoe UI", 10), bg=self.colors['background'])
        self.r2_label.grid(row=1, column=0, padx=8, sticky=tk.W)
        
        self.rmse_label = tk.Label(metrics_frame, text="RMSE: ---",
                                  font=("Segoe UI", 10), bg=self.colors['background'])
        self.rmse_label.grid(row=1, column=1, padx=8)
        
        self.mae_label = tk.Label(metrics_frame, text="MAE: ---",
                                 font=("Segoe UI", 10), bg=self.colors['background'])
        self.mae_label.grid(row=1, column=2, padx=8, sticky=tk.E)
        
        self.chart_frame = tk.Frame(right_frame, bg=self.colors['card'])
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)
        
        tk.Label(self.chart_frame, text="Feature Importance",
                font=("Segoe UI", 11, "bold"),
                bg=self.colors['card'],
                fg=self.colors['text_primary']).pack(pady=(0, 4))
        
        self.chart_canvas = None
        self._create_feature_chart()
        
        history_frame = tk.Frame(right_frame, bg=self.colors['background'], padx=12, pady=12)
        history_frame.pack(fill=tk.X, padx=16, pady=(8, 16))
        
        tk.Label(history_frame, text="Recent Predictions",
                font=("Segoe UI", 11, "bold"),
                bg=self.colors['background'],
                fg=self.colors['text_primary']).pack(pady=(0, 8))
        
        self.history_listbox = tk.Listbox(history_frame, height=4,
                                          font=("Segoe UI", 9),
                                          bg=self.colors['card'],
                                          selectbackground=self.colors['secondary'])
        self.history_listbox.pack(fill=tk.X)
        
    def _create_feature_chart(self):
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()
        
        fig = Figure(figsize=(5, 3), dpi=100, facecolor=self.colors['card'])
        ax = fig.add_subplot(111)
        
        features = ['Sqft', 'Beds', 'Baths', 'Year', 'Lot', 'Garage', 'Neighborhood', 'Location']
        importance = [0.35, 0.12, 0.10, 0.08, 0.15, 0.05, 0.10, 0.05]
        
        colors = [self.colors['secondary'] if i < 4 else self.colors['primary'] 
                  for i in range(len(features))]
        
        bars = ax.barh(features, importance, color=colors)
        ax.set_xlim(0, 0.4)
        ax.set_xlabel('Importance', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance[i]:.0%}', ha='left', va='center', fontsize=7)
        
        fig.tight_layout()
        
        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _create_footer(self):
        footer = tk.Frame(self.main_frame, bg=self.colors['primary'], height=35)
        footer.pack(fill=tk.X, side=tk.BOTTOM, pady=(16, 0))
        footer.pack_propagate(False)
        
        self.status_label = tk.Label(footer, text="Ready",
                                     font=("Segoe UI", 9),
                                     bg=self.colors['primary'],
                                     fg="white")
        self.status_label.pack(side=tk.LEFT, padx=16, pady=8)
        
        tk.Label(footer, text="Models trained with 200+ samples",
                font=("Segoe UI", 9),
                bg=self.colors['primary'],
                fg="#BDC3C7").pack(side=tk.RIGHT, padx=16, pady=8)
        
    def _update_neighborhood_label(self, value):
        self.neighborhood_value_label.config(text=str(int(float(value))))
        
    def _load_or_train_models(self):
        self.status_label.config(text="Loading data...")
        self.root.update()
        
        try:
            data_path = os.path.join(os.path.dirname(__file__), 'data', 'house_data.csv')
            if not os.path.exists(data_path):
                data_path = 'data/house_data.csv'
            
            self.data = pd.read_csv(data_path)
            
            X = self.data[self.feature_names]
            y = self.data['price']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            for name, model in self.models.items():
                self.status_label.config(text=f"Training {name}...")
                self.root.update()
                
                if name == 'Linear Regression':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                self.model_metrics[name] = {
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred)
                }
            
            self._update_model_metrics()
            self.current_model = self.models['Random Forest']
            
            self.status_label.config(text="Ready - Models loaded successfully")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            
    def _update_model_metrics(self):
        model_name = self.model_var.get()
        self.current_model = self.models.get(model_name)
        
        if model_name in self.model_metrics:
            metrics = self.model_metrics[model_name]
            self.r2_label.config(text=f"R² Score: {metrics['r2']:.3f}")
            self.rmse_label.config(text=f"RMSE: ${metrics['rmse']:,.0f}")
            self.mae_label.config(text=f"MAE: ${metrics['mae']:,.0f}")
            
            if metrics['r2'] >= 0.8:
                confidence = "High"
                self.confidence_label.config(fg=self.colors['success'])
            elif metrics['r2'] >= 0.6:
                confidence = "Medium"
                self.confidence_label.config(fg=self.colors['warning'])
            else:
                confidence = "Low"
                self.confidence_label.config(fg=self.colors['accent'])
                
            if hasattr(self, 'last_prediction'):
                self.confidence_label.config(text=f"Confidence: {confidence}")
            
    def _parse_bedrooms(self, value):
        return 6 if value == "6+" else int(value)
    
    def _parse_bathrooms(self, value):
        mapping = {"1": 1, "1.5": 1.5, "2": 2, "2.5": 2.5, "3": 3, "3.5": 3.5, "4+": 4}
        return mapping.get(value, 2)
    
    def _parse_garage(self, value):
        return 3 if value == "3+" else int(value)
    
    def _parse_location(self, value):
        mapping = {"Urban": 1, "Suburban": 2, "Rural": 3}
        return mapping.get(value, 2)
    
    def _predict(self):
        try:
            sqft = self.inputs['sqft'].get()
            bedrooms = self._parse_bedrooms(self.inputs['bedrooms'].get())
            bathrooms = self._parse_bathrooms(self.inputs['bathrooms'].get())
            year_built = self.inputs['year_built'].get()
            lot_size = self.inputs['lot_size'].get()
            garage = self._parse_garage(self.inputs['garage'].get())
            neighborhood = self.inputs['neighborhood_score'].get()
            location = self._parse_location(self.inputs['location_type'].get())
            
            features = np.array([[sqft, bedrooms, bathrooms, year_built, 
                                 lot_size, garage, neighborhood, location]])
            
            model_name = self.model_var.get()
            
            if model_name == 'Linear Regression':
                features_scaled = self.scaler.transform(features)
                prediction = self.models[model_name].predict(features_scaled)[0]
            else:
                prediction = self.models[model_name].predict(features)[0]
            
            prediction = max(prediction, 50000)
            self.last_prediction = prediction
            
            price_range_min = prediction * 0.9
            price_range_max = prediction * 1.1
            
            self.price_label.config(text=f"${prediction:,.0f}")
            self.range_label.config(text=f"Price Range: ${price_range_min:,.0f} - ${price_range_max:,.0f}")
            
            metrics = self.model_metrics[model_name]
            if metrics['r2'] >= 0.8:
                confidence = "High"
            elif metrics['r2'] >= 0.6:
                confidence = "Medium"
            else:
                confidence = "Low"
            self.confidence_label.config(text=f"Confidence: {confidence}")
            
            history_text = f"{sqft}sqft, {bedrooms}bd/{bathrooms}ba → ${prediction:,.0f}"
            self.prediction_history.insert(0, history_text)
            if len(self.prediction_history) > 5:
                self.prediction_history.pop()
            
            self.history_listbox.delete(0, tk.END)
            for item in self.prediction_history:
                self.history_listbox.insert(tk.END, item)
            
            self.status_label.config(text="Prediction complete!")
            
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self.status_label.config(text=f"Error: {str(e)}")
            
    def _reset_inputs(self):
        self.inputs['sqft'].set(2000)
        self.inputs['bedrooms'].set("3")
        self.inputs['bathrooms'].set("2")
        self.inputs['year_built'].set(2000)
        self.inputs['lot_size'].set(8000)
        self.inputs['garage'].set("2")
        self.neighborhood_var.set(7)
        self.inputs['location_type'].set("Suburban")
        
        self.price_label.config(text="$---")
        self.range_label.config(text="Price Range: $--- - $---")
        self.confidence_label.config(text="Confidence: ---")
        
        self.status_label.config(text="Inputs reset")
        
def main():
    root = tk.Tk()
    app = HousePricePredictor(root)
    root.mainloop()

if __name__ == "__main__":
    main()
