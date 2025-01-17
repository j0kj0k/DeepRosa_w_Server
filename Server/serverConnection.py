import socket
import threading
from concurrent.futures import ThreadPoolExecutor

from Server.serverDprosa import serverDprosa

MAX_THREADS = 10  # Maximum number of threads in the thread pool
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)

timegap_dict = {}
cluster_dict = {}
global_var_lock = threading.Lock()

def perform_cluster(client_socket,directory):
    # Perform action 1 based on the received data
    print("Performing cluster with directory:", directory)
    #client_socket.send(directory.encode('utf-8'))
    global timegap_dict
    global cluster_dict
    sD = serverDprosa()

    sD.compilereadCSV(directory)
    sD.cluster_event(directory)
    timegap_dict,cluster_dict = sD.timegap_cluster()

    sD.print_data()

    print("Clustering Done..")
    print("*****************************************")
    print("*****************************************")
    client_socket.send("DONE.".encode('utf-8'))
    clientID = client_socket

    

def perform_sort(client_socket,data):
    global timegap_dict
    global cluster_dict
    # Perform action 2 based on the received data
    print("Performing sorting with list:", data)

    stage, data = data.split('^')
    data = data.strip()

    if stage == "start":
        X = None
    elif stage == "mid":
        X, data = data.split('/')
        data = data.strip()
        data = X + ',' + data

    elif stage == "end":
        X = None
    else:
        print("Unknown stage description:", stage)

    sD = serverDprosa()
    itemList = sD.convertData(data)

    #print(stage+" "+X+" "+data)

    if stage == "end":
        sortedItem = ' '.join(itemList)
        print(sortedItem)

    else:
        sortedList, timegap_dict, cluster_dict = sD.sort_shoppingList(X, itemList, timegap_dict, cluster_dict)
        sortedItem = ', '.join(sortedList)
        print(sortedItem)

    print("*****************************************")
    print("*****************************************")
    client_socket.send(sortedItem.encode('utf-8'))



# Define a mapping of descriptions to functions
ACTION_FUNCTIONS = {
    "cluster": perform_cluster,
    "sort": perform_sort
}

def handle_client(client_socket):

    thread_number = threading.get_ident()
    print(f"Thread {thread_number}: Handling client connection.")

    while True:
        # Receive data and description from the client
        received_data = client_socket.recv(1024).decode('utf-8')
        if not received_data:
            break

        # Split the received data into description and data
        description, data = received_data.split('|')
        description = description.strip()

        # Check if the description corresponds to a known action
        if description in ACTION_FUNCTIONS:
            # Call the appropriate function based on the description
            action_function = ACTION_FUNCTIONS[description]
            action_function(client_socket,data)
            
            break
            thread_pool.submit(action_function, client_socket, data)

            client_socket.close()
        else:
            print("Unknown action description:", description)
                


    # Close the client socket when the connection is closed
    client_socket.close()

def server():
    SERVER_ADDRESS = ('localhost', 8080)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    

    try:
        server_socket.bind(SERVER_ADDRESS)
        server_socket.listen(5)
        print("Server is listening on", SERVER_ADDRESS)
    except socket.error as e:
        print("Unable to start server:", str(e))
        return
    

    with thread_pool:
        while True:
            client_socket, client_address = server_socket.accept()
            print("Accepted connection from", client_address)
            thread_pool.submit(handle_client, client_socket)
'''
    while True:
        client_socket, client_address = server_socket.accept()
        print("Accepted connection from", client_address)

        client_handler_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler_thread.start()
'''

    
def start_server():

    server_thread = threading.Thread(target=server)
    server_thread.start()
