import cv2
import threading
import numpy as np
import base64
import time
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

# Image Processing Functions
def gray_out(pixel_list, width, height, lock, events):
    for i in range(height):
        for j in range(width):
            r, g, b = pixel_list[i][j]
            gray_value = int(0.21 * r + 0.72 * g + 0.07 * b)
            with lock:
                pixel_list[i][j] = (gray_value, gray_value, gray_value)
        print(f"Row {i} converted to grayscale")
        events[i].set()  

def flip_image(pixel_list, width, height, lock, events, flip_events):
    """Flips each row horizontally after grayscale conversion."""
    for i in range(height):
        events[i].wait()  # Wait until grayscale is done
        with lock:
            pixel_list[i] = pixel_list[i][::-1]  # Flip row in place
        print(f"Row {i} flipped")
        flip_events[i].set()  # Signal row `i` is flipped

def blur_image(pixel_list, width, height, lock, flip_events):
    """Applies blur to each row after flipping."""
    kernel_size = 7  # Blur intensity
    for i in range(height):
        flip_events[i].wait()  # Wait until flipping is done
        with lock:
            row = np.array(pixel_list[i], dtype=np.uint8)
            blurred_row = cv2.GaussianBlur(row, (kernel_size, kernel_size), 0)
            pixel_list[i] = blurred_row.tolist()
        print(f"Row {i} blurred")

        # Send live update to web page
        send_image_update(pixel_list, width, height)

def send_image_update(pixel_list, width, height):
    """Sends live image updates to the web page via WebSockets."""
    image_array = np.array(pixel_list, dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', image_array)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    socketio.emit('image_update', {'image': image_base64})

@app.route('/')
def index():
    return render_template('index.html')

def main():
    image_path = "pepene.png"
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not load image.")
        return

    height, width, _ = image.shape
    pixel_list = [[tuple(image[i, j]) for j in range(width)] for i in range(height)]  # Convert image to list

    lock = threading.Lock()  # Prevents race conditions
    events = [threading.Event() for _ in range(height)]  # Sync grayscale -> flip
    flip_events = [threading.Event() for _ in range(height)]  # Sync flip -> blur
    # Start 3 processing threads
    thread1 = threading.Thread(target=gray_out, args=(pixel_list, width, height, lock, events))
    thread2 = threading.Thread(target=flip_image, args=(pixel_list, width, height, lock, events, flip_events))
    thread3 = threading.Thread(target=blur_image, args=(pixel_list, width, height, lock, flip_events))

    time.sleep(5)
    thread1.start()
    thread2.start()
    thread3.start()


    thread1.join()
    thread2.join()
    thread3.join()


    # Save the final processed image
    processed_image = np.array(pixel_list, dtype=np.uint8)
    cv2.imwrite("processed_image.jpg", processed_image)

if __name__ == "__main__":
    threading.Thread(target=main).start()
    socketio.run(app, debug=True)
