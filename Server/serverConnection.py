import socket
import threading

from Server.serverDprosa import serverDprosa


def perform_cluster(client_socket,directory):
    # Perform action 1 based on the received data
    print("Performing cluster with directory:", directory)
    #client_socket.send(directory.encode('utf-8'))

    sD = serverDprosa()
    sD.compilereadCSV(directory)
    sD.cluster_event(directory)

    print("Clustering Done..")

    client_socket.send("Compiling Done".encode('utf-8'))
    clientID = client_socket

    

def perform_action2(data):
    # Perform action 2 based on the received data
    print("Performing action 2 with data:", data)


# Define a mapping of descriptions to functions
ACTION_FUNCTIONS = {
    "cluster": perform_cluster,
    "action2": perform_action2
}

def handle_client(client_socket):
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
    


    while True:
        client_socket, client_address = server_socket.accept()
        print("Accepted connection from", client_address)

        client_handler_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler_thread.start()
    
def start_server():
    server_thread = threading.Thread(target=server)
    server_thread.start()