import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter
import tkinter.messagebox
from tkinter import filedialog
import customtkinter as CTk
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
import time


CTk.set_appearance_mode("system")
CTk.set_default_color_theme("green")

class dprosaGUI(CTk.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Deep Rosa Visualizer")
        self.geometry("1200x800")
        self.k = 50

        # configure grid layout (4x4)
        #self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((1, 2, 3), weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # sidebar frame
        self.sidebar_frame = CTk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.logo_label = CTk.CTkLabel(self.sidebar_frame, text="Deep Rosa", font=CTk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=20)
        self.sidebar_upload_btn = CTk.CTkButton(self.sidebar_frame, text="Upload", command=self.upload_event)
        self.sidebar_upload_btn.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_reset_btn = CTk.CTkButton(self.sidebar_frame, text="Reset", fg_color="indianred1", command=self.reset_event)
        self.sidebar_reset_btn.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_cluster_var = tkinter.StringVar(value="No. of Clusters: 60")
        self.sidebar_cluster_label = CTk.CTkLabel(self.sidebar_frame, text=self.sidebar_cluster_var.get(), anchor='w')
        self.sidebar_cluster_label.grid(row=3, column=0, padx=20, pady=(10, 0))
        self.sidebar_cluster_slider = CTk.CTkSlider(self.sidebar_frame, from_=20, to=100, number_of_steps=80, command=self.change_cluster_slider_event)
        self.sidebar_cluster_slider.grid(row=4, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.sidebar_cluster_btn = CTk.CTkButton(self.sidebar_frame, text="Cluster", command=self.cluster_event)
        self.sidebar_cluster_btn.grid(row=5, column=0, padx=20, pady=(10,400))

        self.ui_settings_label = CTk.CTkLabel(self.sidebar_frame, text="UI Settings:", anchor="w")
        self.ui_settings_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_menu = CTk.CTkOptionMenu(self.sidebar_frame, values=["System", "Light", "Dark"], command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=10, column=0, padx=20, pady=(10, 20))

        # data info frame
        self.data_info_tab = CTk.CTkTabview(self, height=800, width=450)
        self.data_info_tab.grid(row=0, rowspan=2, column=1, padx=(10,5), pady=10, sticky="nsew")
        self.data_info_tab.add("General")
        self.data_info_tab.add("All Items")
        self.data_info_tab.add("Timegaps")
        self.data_info_tab.tab("General").grid_columnconfigure(0, weight=1)
        self.data_info_tab.tab("All Items").grid_columnconfigure(0, weight=1)
        self.data_info_tab.tab("Timegaps").grid_columnconfigure(0, weight=1)

        self.general_text = CTk.CTkTextbox(self.data_info_tab.tab("General"), height=800, width=470)
        self.general_text.grid(row=0, column=0, sticky="nsew")

        self.items_text = CTk.CTkTextbox(self.data_info_tab.tab("All Items"), height=800, width=470)
        self.items_text.grid(row=0, column=0, sticky="nsew")

        self.timegaps_text = CTk.CTkTextbox(self.data_info_tab.tab("Timegaps"), height=800, width=470)
        self.timegaps_text.grid(row=0, column=0, sticky="nsew")

        # clustered items frame
        self.cluster_info_frame = CTk.CTkFrame(self, width=240)
        self.cluster_info_frame.grid(row=0, column=2, padx=5, pady=(10,5), sticky="nsew")

        self.k = 60
        temp = []
        for i in range(0, self.k):
            temp.append(f"Cluster {i+1}")

        self.cluster_list_menu = CTk.CTkOptionMenu(master=self.cluster_info_frame, values=temp, width=280, command=self.print_cluster)
        self.cluster_list_menu.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.cluster_info_text = CTk.CTkTextbox(master=self.cluster_info_frame, height=190, width=280)
        self.cluster_info_text.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        

        # plot selector frame
        self.plot_selector_frame = CTk.CTkFrame(self, width=160)
        self.plot_selector_frame.grid(row=0, column=3, padx=(5,10), pady=(10,5), sticky="nsew")
        self.plots_var = tkinter.IntVar(value=0)
        self.plots_radio_label = CTk.CTkLabel(master=self.plot_selector_frame, text="Graph Selector")
        self.plots_radio_label.grid(row=0, column=2, padx=10, pady=10, stick="")
        self.items_graph_radio = CTk.CTkRadioButton(master=self.plot_selector_frame, text="Items Graph", variable=self.plots_var, value=1, command=self.plot_preview_event)
        self.items_graph_radio.grid(row=1, column=2, pady=10, padx=20, sticky="w")
        self.cluster_graph_radio = CTk.CTkRadioButton(master=self.plot_selector_frame, text="Cluster Graph", variable=self.plots_var, value=2, command=self.plot_preview_event)
        self.cluster_graph_radio.grid(row=2, column=2, pady=10, padx=20, sticky="w")
        self.distance_matrix_radio = CTk.CTkRadioButton(master=self.plot_selector_frame, text="Distance Matrix", variable=self.plots_var, value=3, command=self.plot_preview_event)
        self.distance_matrix_radio.grid(row=3, column=2, pady=10, padx=20, sticky="w")

        # plot display frame
        self.plot_preview_frame = CTk.CTkFrame(self, height=520, width=470)
        self.plot_preview_frame.grid(row=1, rowspan=1, column=2, columnspan=2, padx=(5,10), pady=(5,10), sticky="nsew")

        # set default values
        self.appearance_mode_menu.set("System")
        self.reset_event()

    # UI EVENTS

    def change_appearance_mode_event(self, appearance_mode):
        if appearance_mode == 'System':
            CTk.set_appearance_mode("system")
        elif appearance_mode == 'Light':
            CTk.set_appearance_mode("light")
        elif appearance_mode == 'Dark':
            CTk.set_appearance_mode("dark")

    def change_cluster_slider_event(self, kcluster): 
        self.sidebar_cluster_var.set(f"No. of Clusters: {int(kcluster)}")
        self.sidebar_cluster_label = CTk.CTkLabel(self.sidebar_frame, text=self.sidebar_cluster_var.get(), anchor='w')
        self.sidebar_cluster_label.grid(row=3, column=0, padx=20, pady=(10, 0))
        self.k = int(kcluster)
        print(self.k)

    def print_data(self):
        self.general_text.configure(state='normal')
        self.general_text.delete(1.0, 'end')
        self.general_text.insert('end', f"Total Items: {len(self.item_list)}\n")
        self.general_text.insert('end', f"Total Pairs: {len(self.timegap_dict)}\n")
        self.general_text.insert('end', f"Total Shoppers: {self.total_shoppers}\n\n")
        self.general_text.insert('end', f"Running time: {self.running_time:.2f}s\n\n")
        self.general_text.configure(state='disabled')

        self.items_text.configure(state='normal')
        self.items_text.delete(1.0, 'end')
        for i, item in enumerate(self.item_list):
            self.items_text.insert('end', f"{i + 1}. {item}\n")
        self.items_text.configure(state='disabled')

        self.timegaps_text.configure(state='normal')
        self.timegaps_text.delete(1.0, 'end')
        for i, (key, value) in enumerate(self.timegap_dict.items()):
            item_x, item_y = key
            self.timegaps_text.insert('end', f"{i + 1}. {item_x}, {item_y} - {value:.2f}\n")
        self.timegaps_text.configure(state='disabled')

    def print_cluster(self, cluster_string):
        temp = []
        for i in range(0, self.k):
            temp.append(f"Cluster {i+1}")
        self.cluster_list_menu.configure(values=temp)  

        cluster_int = int(cluster_string.split()[-1])
        self.cluster_info_text.configure(state='normal')
        self.cluster_info_text.delete(1.0, 'end')
        cluster_items = []
        for key, value in self.cluster_dict.items():
            if value == cluster_int:
                cluster_items.append(f"{key}")

        cluster_items.sort()
        for i, item in enumerate(cluster_items):
            self.cluster_info_text.insert('end', f"{i+1}. {item}\n")
        
        self.cluster_info_text.configure(state='disabled')

    # BUTTON EVENTS
    
    def upload_event(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        df = pd.read_csv(file_path, header=None)
        self.item_list = sorted(df[(df.iloc[:, 2].apply(lambda x: isinstance(x, (int, float))) | \
                                    df.iloc[:, 2].apply(lambda x: str(x).isdigit() or x == 'Good'))].iloc[:, 0].unique())
        
        self.timegap_dict, self.total_shoppers, self.running_time = self.calculate_timegap(df)

        self.print_data()
        self.sidebar_upload_btn.configure(state='disabled')
        self.sidebar_cluster_slider.configure(state='normal')
        self.sidebar_cluster_btn.configure(state='normal')
        self.sidebar_reset_btn.configure(state='normal')
        self.items_graph_radio.configure(state='normal')

    def cluster_event(self):
        self.cluster_dict = self.cluster_items()
        self.cluster_graph_radio.configure(state='normal')
        self.sidebar_cluster_slider.configure(state='disabled')
        self.sidebar_cluster_btn.configure(state='disabled')
        self.cluster_list_menu.configure(state='normal')
        self.print_cluster("Cluster 1")

    def reset_event(self):
        
        self.sidebar_upload_btn.configure(state='normal')

        self.general_text.configure(state='normal')
        self.items_text.configure(state='normal')
        self.timegaps_text.configure(state='normal')
        self.general_text.delete(1.0, 'end')
        self.items_text.delete(1.0, 'end')
        self.timegaps_text.delete(1.0, 'end')
        self.general_text.configure(state='disabled')
        self.items_text.configure(state='disabled')
        self.timegaps_text.configure(state='disabled') 

        self.sidebar_cluster_var.set("No. of Clusters: 60")

        self.sidebar_cluster_slider.configure(state='disabled')
        self.sidebar_cluster_btn.configure(state='disabled')
        self.sidebar_reset_btn.configure(state='disabled')

        self.cluster_list_menu.set("Cluster 1")
        self.cluster_list_menu.configure(state='disabled')
        self.cluster_info_text.configure(state='normal')
        self.cluster_info_text.delete(1.0, 'end')
        self.cluster_info_text.configure(state='disabled')

        self.plots_var.set(0)
        self.items_graph_radio.configure(state='disabled')
        self.cluster_graph_radio.configure(state='disabled')
        self.distance_matrix_radio.configure(state='disabled')

        for widget in self.plot_preview_frame.winfo_children():
            widget.destroy()

    def plot_preview_event(self):
        plots_var = self.plots_var.get()
        for widget in self.plot_preview_frame.winfo_children():
            widget.destroy()
            
        if plots_var == 1:
            self.items_graph()
        elif plots_var == 2:
            self.cluster_graph()
        elif plots_var == 3:
            self.distance_matrix()

    # FUNCTIONS

    def calculate_timegap(self, df):
        start_time = time.time()
        timegap_dict = {}
        total_shoppers = 0
        temp_dict = {}

        for index1 in range(len(df) - 1):
            item_x = df.iloc[index1, 0]
            value_x = df.iloc[index1, 1]
            value_x_next = df.iloc[index1 + 1, 1]
            status_x = df.iloc[index1, 2]

            if (isinstance(status_x, (int, float)) or str(status_x).isdigit() or status_x == 'Good'):
                if value_x == 0:
                    total_shoppers += 1
                    temp_dict.clear()
                    temp_dict[item_x] = value_x
                else:
                    temp_dict[item_x] = value_x

            if value_x_next == 0 or value_x_next is None:
                sorted_keys = sorted(temp_dict.keys())
                num_items = len(sorted_keys)
                weights = [1 / (i + 1) for i in range(num_items - 1)]

                for i in range(num_items - 1):
                    for j in range(i + 1, num_items):
                        pair = tuple(sorted([sorted_keys[i], sorted_keys[j]]))
                        diff = abs(temp_dict[sorted_keys[i]] - temp_dict[sorted_keys[j]])

                        weighted_diff = diff * weights[i]

                        if pair in timegap_dict:
                            timegap_dict[pair].append(weighted_diff)
                        else:
                            timegap_dict[pair] = [weighted_diff]

                temp_dict.clear()

        for key in timegap_dict:
            timegap_dict[key] = sum(timegap_dict[key]) / len(timegap_dict[key])

        sorted_timegap_dict = dict(sorted(timegap_dict.items(), key=lambda x: x[0]))
        end_time = time.time()
        return sorted_timegap_dict, total_shoppers, end_time - start_time
    
    def cluster_items(self):
        max_timegap = 50
        cluster_dict = {}
        cluster_index = 0
        sorted_pairs = sorted(self.timegap_dict.items(), key=lambda x: x[1])

        for key, timegap in sorted_pairs:
            item_x, item_y = key

            if timegap <= max_timegap:
                if item_x not in cluster_dict and item_y not in cluster_dict:
                    cluster_dict[item_x] = cluster_index
                    cluster_dict[item_y] = cluster_index
                    cluster_index += 1
                elif item_x in cluster_dict and item_y not in cluster_dict:
                    cluster_dict[item_y] = cluster_dict[item_x]
                elif item_y in cluster_dict and item_x not in cluster_dict:
                    cluster_dict[item_x] = cluster_dict[item_y]

        # If the specified number of clusters is less than the actual number, combine clusters
        if self.k < cluster_index:
            cluster_mapping = {}
            new_cluster_index = 0

            for item, cluster in cluster_dict.items():
                if cluster not in cluster_mapping:
                    cluster_mapping[cluster] = new_cluster_index
                    new_cluster_index += 1
                cluster_dict[item] = cluster_mapping[cluster]

        return cluster_dict

    def items_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.item_list)

        edge_labels = {}
        for key, value in self.timegap_dict.items():
            item_x, item_x_plus_1 = key
            
            if value != 0 or G.has_edge(item_x, item_x_plus_1):
                G.add_edge(item_x, item_x_plus_1, weight=value)
                edge_labels[(item_x, item_x_plus_1)] = f'{value:.2f}'

        isolated_nodes = [node for node in G.nodes() if G.degree[node] == 0]
        G.remove_nodes_from(isolated_nodes)

        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots()

        nx.draw(G, pos, with_labels=False, node_color='green', node_size=6, font_size=1, width=0.01, alpha=1, ax=ax)
        #edge_labels = nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_preview_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)  

    def cluster_graph(self):
        cluster_dict = self.cluster_dict
        G = nx.Graph()

        for datapoint, cluster in cluster_dict.items():
            G.add_node(datapoint, cluster=cluster)

        for datapoint, cluster in cluster_dict.items():
            for other_datapoint, other_cluster in cluster_dict.items():
                if cluster == other_cluster and datapoint != other_datapoint:
                    G.add_edge(datapoint, other_datapoint)

        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots()

        clusters = set(cluster_dict.values())
        cmap = plt.cm.get_cmap('viridis', len(clusters))

        # Drawing nodes and edges
        for cluster in clusters:
            nodes = [node for node, data in G.nodes(data=True) if data['cluster'] == cluster]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[cmap(cluster % len(clusters)) for _ in range(len(nodes))], alpha=0.8, ax=ax)
            #nx.draw_networkx_labels(G, pos, labels={node: node for node in nodes}, font_color='white')
            #nx.draw_networkx_edges(G, pos, edgelist=G.edges())

        canvas = FigureCanvasTkAgg(fig, master=self.plot_preview_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)  

    def distance_matrix(self):
        item_pairs = list(self.timegap_dict.keys())
        time_gaps = [self.timegap_dict[key] for key in item_pairs]

        unique_items = list(set(item for pair in item_pairs for item in pair))

        num_items = len(unique_items)
        distance_matrix = np.zeros((num_items, num_items))

        for i, item_x in enumerate(unique_items):
            for j, item_y in enumerate(unique_items):
                if i != j:
                    key = tuple(sorted([item_x, item_y]))
                    if key in item_pairs:
                        index = item_pairs.index(key)
                        distance_matrix[i, j] = time_gaps[index]

        G = nx.Graph()
        G.add_nodes_from(unique_items)
        for i in range(num_items):
            for j in range(i + 1, num_items):
                if distance_matrix[i, j] > 0:
                    G.add_edge(unique_items[i], unique_items[j], weight=distance_matrix[i, j])

        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots()

        nx.draw(G, pos, with_labels=False, node_size=6, font_size=-1, font_color='black', node_color='green', edge_color='gray', width=0.01, ax=ax)
        #labels = nx.get_edge_attributes(G, 'weight')
        #x.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_preview_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)  


dprosa = dprosaGUI()
dprosa.mainloop()