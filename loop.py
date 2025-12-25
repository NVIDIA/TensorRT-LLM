import socket
import time

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("localhost", 10010))
    s.listen(1024)
    while True:
        time.sleep(1)
