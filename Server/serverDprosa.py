import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
#import math
import csv
import time
import os

#from Models._dprosa import \
#    initialize_timegap, add_timegap, check_timegap, normalize_timegaps,\
#    dict_to_matrix, agglomerative_clustering, kmeans_clustering, sort_shopping_list


class serverDprosa():

    def compilereadCSV(self,directory):
        self.start_time = time.time()
        '''
        ------------------------------------------------------------ COMPILE CSV ------------------------------------------------------------
        '''
        print("Compiling CSV")

        # Create a new directory for the compiled CSV files
        compiled_directory = os.path.join(directory, 'CSVRecordings', 'compiled_csv')
        os.makedirs(compiled_directory, exist_ok=True)

        output_file = os.path.join(compiled_directory, 'compiledData.csv')

        csvdirectory = os.path.join(directory, 'CSVRecordings')
        print(f"Accessing files in folder: {csvdirectory}")

        csv_files = [f for f in os.listdir(csvdirectory) if f.endswith('.csv')]
        csv_files.sort()

        # Delete the existing CSV file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)

        with open(output_file, 'a', newline='') as output_csv:
            writer = csv.writer(output_csv)

            for csv_file in csv_files:
                file_path = os.path.join(csvdirectory, csv_file)
                with open(file_path, 'r', errors='replace') as input_csv:
                    csv_text = input_csv.read().replace('\x00', '')
                    csv_lines = csv_text.splitlines()

                    reader = csv.reader(csv_lines)

                    for row in reader:
                        writer.writerow(row)
        print("Done compiling CSV")
        '''
        ------------------------------------------------------------ TIMEGAP ------------------------------------------------------------
        '''
        df = pd.read_csv(output_file, header=None)
        print("--- %s seconds ---    || AFTER CSV READ" % (time.time() - self.start_time))

        self.item_list = sorted(df[(df.iloc[:, 2].apply(lambda x: isinstance(x, (int, float))) | \
                                    df.iloc[:, 2].apply(lambda x: str(x).isdigit() or x == 'Good'))].iloc[:, 0].unique())

        print("--- %s seconds ---    || AFTER SORTING" % (time.time() - self.start_time))

            
        self.timegap_dict, self.total_shoppers = self.init_timegap()
        print("--- %s seconds ---    || AFTER INIT TIMEGAP" % (time.time() - self.start_time))
        self.timegap_dict, self.total_shoppers = self.add_timegap(df) 
        print("--- %s seconds ---    || AFTER ADD TIMEGAP" % (time.time() - self.start_time))
        self.timegap_dict = dict(sorted(self.timegap_dict.items(), key=lambda x: x[1]))     #sort from lowest timegap


        print("--- %s seconds ---    || AFTER TIMEGAP DICT" % (time.time() - self.start_time))
            

    def init_timegap(self):
        timegap_dict = {}
        threshold_dict = {}
        total_shoppers = 0
        for key1 in self.item_list:
            for key2 in self.item_list:
                if key1 != key2:
                    pair = tuple(sorted((key1, key2)))
                    if pair not in timegap_dict:
                        timegap_dict[pair] = [100000]

        sorted_timegap_dict = dict(sorted(timegap_dict.items(), key=lambda x: x[0]))

        for pair in sorted_timegap_dict:
            threshold_dict[pair] = 0
        
        self.threshold_dict = threshold_dict
        return sorted_timegap_dict, total_shoppers


        def print_data(self):
            self.running_time = 0    
            print(f"Total Items: {len(self.item_list)}\n")
            print(f"Total Pairs: {len(self.timegap_dict)}\n")
            print(f"Total Shoppers: {self.total_shoppers}\n\n")
            #print(f"Running time: {self.running_time:.2f}s\n\n")

    def cluster_event(self,directory):
        self.dict_to_matrix()

        self.agglomerative_clustering()

        #self.cluster_dendrogram()
        self.export_as_csv(directory)
        self.centroid_dict = self.calculate_centroid()
        #print(self.centroid_dict)
        #self.adjust_clusters()



    def agglomerative_clustering(self):
        k = 120
        agglomerative = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=50)
        cluster_labels = agglomerative.fit_predict(self.timegap_matrix)

        clustered_items = {}
        for label in set(cluster_labels):
            clustered_items[label] = []

        for item, label in zip(self.item_list, cluster_labels):
            clustered_items[label].append(item)

        self.cluster_dict = clustered_items
        # print(self.cluster_dict)
        self.k = 120


    def calculate_centroid(self):
        centroid_dict = {}

        for cluster_index, cluster_items in self.cluster_dict.items():
            for i in range(0, len(cluster_items)-1):
                item_x = cluster_items[i]
                for j in range(0, len(cluster_items)-1):
                    if i != j:
                        item_y = cluster_items[j]
                        pair = (item_x, item_y)
                        sorted_pair = tuple(sorted(list(pair)))
                        if sorted_pair in self.timegap_dict:
                            if cluster_index not in centroid_dict:
                                centroid_dict[cluster_index] = []
                            centroid_dict[cluster_index].append(self.timegap_dict[sorted_pair])

        for cluster_index, cluster_values in centroid_dict.items():
            if len(cluster_values) > 0:
                centroid_dict[cluster_index] = sum(cluster_values) / len(cluster_values)

        return centroid_dict  

    def sort_shopping_list(self):
        self.sorted_list = self.shopping_list
        self.cluster_anchor = 0
        self.sorted_list = sorted(
            self.sorted_list,
            key=lambda x: (
                -1 if any(x in value for value in self.cluster_dict.get(self.cluster_anchor-1, [])) else 0,
                next((key for key, value in self.cluster_dict.items() if x in value), 0)
            )
        )   
        for i in range(len(self.sorted_list) - 1):
            item_x = self.sorted_list[i]
            item_y = self.sorted_list[i + 1]
            cluster_x = next((key for key, value in self.cluster_dict.items() if item_x in value), None)
            cluster_y = next((key for key, value in self.cluster_dict.items() if item_y in value), None)

            if cluster_x is not None and cluster_x == cluster_y:
                continue

            min_timegap = float('inf')
            min_timegap_item = None

            for j in range(i + 1, len(self.sorted_list)):
                next_item = self.sorted_list[j]

                if (item_x, next_item) in self.timegap_dict:
                    timegap = self.timegap_dict[(item_x, next_item)]

                    if timegap < min_timegap:
                        min_timegap = timegap
                        min_timegap_item = next_item

            if min_timegap_item is not None:
                index_min_timegap_item = self.sorted_list.index(min_timegap_item)
                self.sorted_list[i + 1], self.sorted_list[index_min_timegap_item] = self.sorted_list[index_min_timegap_item], self.sorted_list[i + 1]

        self.shopping_list = self.sorted_list
        self.print_shopping_list()

    def print_shopping_list(self):
        for i, item in enumerate(self.shopping_list):
            self.print('end', f"{i+1}. {item}\n")

    def add_timegap(self, df):
        timegap_dict = self.timegap_dict
        total_shoppers = self.total_shoppers
        temp_dict = {}
        print_max = len(df) // 5

        for index in range(len(df)):
            item_x, value_x, status_x = df.iloc[index]
            value_x_next = 0 if index == len(df) - 1 else df.iloc[index + 1, 1]

            if value_x == 0:
                total_shoppers += 1
                temp_dict.clear()
                temp_dict[item_x] = value_x
            else:
                temp_dict[item_x] = value_x

            if value_x_next == 0 and len(temp_dict) != 1:
                sorted_keys = sorted(temp_dict.keys())
                num_items = len(sorted_keys)

                for i in range(num_items - 1):
                    key1, key2 = sorted_keys[i], sorted_keys[i + 1]
                    pair = tuple(sorted((key1, key2)))
                    diff = abs(temp_dict[key1] - temp_dict[key2])

                    if pair in timegap_dict:
                        timegap_values = timegap_dict[pair]
                        if diff < timegap_values[-1] - 10 or diff > timegap_values[-1] + 10:
                            if pair not in self.threshold_dict:
                                self.threshold_dict[pair] = 1
                            else:
                                self.threshold_dict[pair] += 1

                            if self.threshold_dict[pair] >= 3:
                                timegap_values.append(sum(timegap_values[-3:]) / 3)
                                self.threshold_dict[pair] = 0
                        timegap_values.append(diff)
                    else:
                        timegap_dict[pair] = [diff]

                temp_dict.clear()

        for values in timegap_dict.values():
            if len(values) > 1:
                values.pop(0)

        for key in timegap_dict:
            timegap_dict[key] = sum(timegap_dict[key]) / len(timegap_dict[key])

        sorted_timegap_dict = dict(sorted(timegap_dict.items(), key=lambda x: x[0]))
        return sorted_timegap_dict, total_shoppers



    def dict_to_matrix(self):
        items = self.item_list
        matrix = np.zeros((len(items), len(items)))

        for i in range(len(items)):
            for j in range(len(items)):
                if i != j:
                    key = (items[i], items[j])
                    if key in self.timegap_dict:
                        matrix[i][j] = self.timegap_dict[key]
                        matrix[j][i] = self.timegap_dict[key]

        self.timegap_matrix = matrix

    def export_as_csv(self,directory):

        # Create a new directory for the compiled CSV files
        compiled_directory = os.path.join(directory, 'clusteredCSV')
        os.makedirs(compiled_directory, exist_ok=True)

        clustercsv = os.path.join(compiled_directory, 'clusters.csv')

        print(f"Accessing file in folder: {compiled_directory}")

        # Prepare CSV data
        csv_data = []
        for cluster, items in self.cluster_dict.items():
            for item in items:
                csv_data.append([item, cluster])

        # Create a CSV string
        csv_string = ""
        with open(clustercsv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Item', 'Cluster'])
            writer.writerows(csv_data)
        
        print("--- %s seconds ---    || DONE..." % (time.time() - self.start_time))
        
        return csv_string