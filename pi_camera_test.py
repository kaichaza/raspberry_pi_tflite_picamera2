from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()

#camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
#picam2.configure(camera_config)

preview_config = picam2.create_preview_configuration(main={"size": (800, 600)})
picam2.configure(preview_config)

picam2.start_preview(Preview.QTGL)

picam2.start()

time.sleep(2)
filename = "photos/myphoto.jpg"
picam2.capture_file(filename)