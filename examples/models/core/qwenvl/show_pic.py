import argparse

import cv2
import numpy as np
import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='8006')
    return parser.parse_args()


args = parse_arguments()
ip_addr = "tcp://" + args.ip + ":" + args.port
socket.connect(ip_addr)

while True:
    socket.send(b"a")
    message = socket.recv()
    if len(message) == 1 and message == b'x':
        break
    image = np.frombuffer(message, dtype=np.uint8)
    image = cv2.imdecode(image, 1)
    image = cv2.resize(image, dsize=(512, 384))
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
