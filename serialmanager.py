import serial
import threading

class SerialManager:
    def __init__(self, port="COM3", baudrate=115200, timeout=1):
        #Initializes the serial connection
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.lock = threading.Lock()

    def write(self, message: str):
        #Writes a string over the serial connection
        with self.lock:
            if self.ser.is_open:
                self.ser.write((message + "\n").encode("utf-8"))

    def read_line(self) -> str:
        #Reads a line from the serial connection and returns a string
        with self.lock:
            if self.ser.is_open and self.ser.in_waiting > 0:
                return self.ser.readline().decode("utf-8", errors="ignore").strip()
        return ""

    def close(self):
        #Close the serial connection at the end of the program's execution
        if self.ser.is_open:
            self.ser.close()
