import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class KMeansApp:
    def __init__(self, master):
        self.master = master
        self.master.title("K-Means Clustering")

        # Upper-Left Section
        self.column1_label = tk.Label(master, text="Select the first column:")
        self.column1_label.grid(row=0, column=0, padx=10, pady=2, sticky=tk.W)

        self.column1_var = tk.StringVar()
        self.column1_combobox = ttk.Combobox(master, textvariable=self.column1_var)
        self.column1_combobox.grid(row=0, column=1, padx=10, pady=2)
        self.column1_combobox['values'] = ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium',
                                           'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols',
                                           'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline',
                                           'Customer_Segment']
        self.column1_combobox.current(0)  # Set the default selection

        self.column2_label = tk.Label(master, text="Select the second column:")
        self.column2_label.grid(row=1, column=0, padx=10, pady=2, sticky=tk.W)

        self.column2_var = tk.StringVar()
        self.column2_combobox = ttk.Combobox(master, textvariable=self.column2_var)
        self.column2_combobox.grid(row=1, column=1, padx=10, pady=2)
        self.column2_combobox['values'] = ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium',
                                           'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols',
                                           'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline',
                                           'Customer_Segment']
        self.column2_combobox.current(1)  # Set the default selection

        self.label3 = tk.Label(master, text="Enter n clusters:")
        self.label3.grid(row=2, column=0, pady=2)
        self.entry3 = tk.Entry(master)
        self.entry3.grid(row=2, column=1, pady=2)

        self.run_button = tk.Button(master, text="Run", command=self.run_clustering)
        self.run_button.grid(row=3, column=0, pady=2, sticky=tk.W + tk.E)

        self.reset_button = tk.Button(master, text="Reset", command=self.reset_display)
        self.reset_button.grid(row=3, column=1, pady=2, sticky=tk.W + tk.E)

        # Lower-Left Section
        self.label_clusters_centroids = tk.Label(master, text="Clusters and Centroids", font=("Arial", 12, "bold"))
        self.label_clusters_centroids.grid(row=4, column=0, columnspan=2)

        self.listbox = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=40, height=20, font=("Arial", 10))
        self.listbox.grid(row=5, column=0, rowspan=6, columnspan=2)

        # Right Section (for scatterplot)
        self.right_frame = tk.Frame(master)
        self.right_frame.grid(row=0, column=2, rowspan=8)

        self.scatter_canvas = FigureCanvasTkAgg(plt.Figure(figsize=(8, 6)), master=self.right_frame)
        self.scatter_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.scatter_canvas.draw()

        # Configure columns to have equal weight
        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)

    def plot_scatter(self, data):
        plt.figure(figsize=(8, 6))

        for i in range(data['cluster'].nunique()):
            cluster_data = data[data['cluster'] == i]
            plt.scatter(cluster_data[self.column1_var.get()],
                        cluster_data[self.column2_var.get()],
                        label=f'Cluster {i}')

        plt.scatter(data['centroid_x'], data['centroid_y'], marker='X', s=100, c='red', label='Centroids')

        plt.xlabel(self.column1_var.get())
        plt.ylabel(self.column2_var.get())
        plt.title('K-Means Clustering')
        plt.legend()

        canvas = FigureCanvasTkAgg(plt.gcf(), master=self.master)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=2, rowspan=8)

    def calculate_distance(self, point, centroid):
        return np.sqrt(np.sum((point - centroid) ** 2))

    def update_centroids(self, clusters, data):
        new_centroids = []
        for cluster in clusters:
            if len(cluster) > 0:
                new_centroid = np.mean(data.iloc[cluster], axis=0)
                new_centroids.append(new_centroid)
        return np.array(new_centroids)

    def kmeans(self, data, num_clusters, max_iterations=100):
        centroids = data.sample(num_clusters, random_state=42).to_numpy()

        for _ in range(max_iterations):
            clusters = [[] for _ in range(num_clusters)]
            for i, point in data.iterrows():
                distances = [self.calculate_distance(point.to_numpy(), centroid) for centroid in centroids]
                nearest_cluster = np.argmin(distances)
                clusters[nearest_cluster].append(i)

            new_centroids = self.update_centroids(clusters, data)

            if np.array_equal(centroids, new_centroids):
                break

            centroids = new_centroids

        return centroids, clusters

    def reset_display(self):
        self.listbox.delete('1.0', tk.END)  # Clear the listbox

        self.scatter_canvas.figure.clf()
        self.scatter_canvas.draw_idle()

    def run_clustering(self):
        try:
            column1 = self.column1_var.get()
            column2 = self.column2_var.get()
            num_clusters = int(self.entry3.get())

            file_path = "wine.csv"
            data = pd.read_csv(file_path)
            selected_data = data[[column1, column2]]

            centroids, clusters = self.kmeans(selected_data, num_clusters)

            output_file_path = "output.csv"
            data['cluster'] = -1
            for i, cluster in enumerate(clusters):
                data.loc[cluster, 'cluster'] = i
            data.to_csv(output_file_path, index=False)

            # Scatterplot
            self.scatter_canvas.figure.clf()  # Clear the previous plot
            ax = self.scatter_canvas.figure.add_subplot(111)
            for i, cluster in enumerate(clusters):
                ax.scatter(selected_data.iloc[cluster][column1], selected_data.iloc[cluster][column2],
                           label=f'Cluster {i}')

            ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='red', label='Centroids')
            ax.set_xlabel(column1)
            ax.set_ylabel(column2)
            ax.set_title('K-Means Clustering')
            ax.legend()
            ax.grid(True)

            self.scatter_canvas.draw_idle()

            self.listbox.delete('1.0', tk.END)  # Clear previous content
            for i, centroid in enumerate(centroids):
                self.listbox.insert(tk.END, f"Centroid: {i} {tuple(centroid)}\n")
                self.listbox.insert(tk.END, f"{selected_data.iloc[clusters[i]].to_string(index=False)}\n\n")

            messagebox.showinfo("Clustering Complete", f"Results saved to {output_file_path}")

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = KMeansApp(root)
    root.mainloop()
