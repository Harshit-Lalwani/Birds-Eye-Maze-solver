import bluetooth
import time

# ESP32 Bluetooth MAC address or use 'ESP32_BT' for the name
server_address = 'D8:13:2A:43:22:9E'  # Replace with your ESP32 Bluetooth MAC address
port = 1  # Default RFCOMM port for Bluetooth SPP

# Create a Bluetooth socket
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

try:
    print(f"Connecting to {server_address}...")
    sock.connect((server_address, port))
    print("Connected to ESP32")

    # Send data to ESP32 in a loop based on user input
    while True:
        # Get two numbers from the user
        print("Enter direction (1 for forward\n2 for backward\n3 for righ\n 4 for left) or 'exit' to quit: ")
        user_input_1 = input("Enter the first decimal number (or type 'exit' to quit): ")
        user_input_2 = input("Enter the second decimal number (or type 'exit' to quit): ")

        # Check if the user wants to exit
        if user_input_1.lower() == 'exit' or user_input_2.lower() == 'exit':
            print("Exiting...")
            break

        # Validate and convert inputs to floats
        try:
            num1 = int(user_input_1)
            num2 = float(user_input_2)
            data = f"{num1},{num2}"  # Format the numbers as "num1,num2"
            print(f"Sending data: {data}")
            sock.send(data)  # Send the numbers to ESP32

            # Receive response from ESP32
            response = sock.recv(1024)
            print(f"Received from ESP32: {response.decode()}")
        except ValueError:
            print("Invalid input! Please enter valid decimal numbers.")
            
except bluetooth.BluetoothError as e:
    print(f"Failed to connect or send data: {e}")
finally:
    sock.close()
    print("Bluetooth connection closed")
