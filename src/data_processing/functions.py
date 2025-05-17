import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tkinter as tk
from tkinter import ttk


# Step 2: Load the Parquet file
def load_parquet_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_parquet(file_path)
    return df

# Step 3: Preview the data
def preview_data(df):
    print("\nDataFrame Head:\n", df.head())
    print("\nDataFrame Info:\n")
    print(df.info())
    print("\nSummary Statistics:\n", df.describe())

# Step 4: Basic visualizations
def visualize_data(df):

    numeric_columns = df.select_dtypes(include='number').columns

    # Histogram for numerical columns
    df[numeric_columns].hist(figsize=(10, 8), bins=20)
    plt.suptitle("Histograms of Numeric Features")
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    if len(numeric_columns) > 1:
        corr = df[numeric_columns].corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()
    else:
        print("Not enough numeric columns for a correlation heatmap.")

def display_table(df):
    root = tk.Tk()
    root.title("Parquet Viewer")

    # Create a Treeview widget
    tree = ttk.Treeview(root)
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"

    # Set columns
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor="center")

    # Add rows
    for index, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    # Add scrollbar
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

    tree.pack(expand=True, fill="both")
    root.mainloop()