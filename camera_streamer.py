import cv2
import threading
import time

class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

# 2. Set exposure manually (-5 to -7 gives high frame rates in normal indoor light)
# Lower values (e.g. -6) mean faster shutter speed = higher FPS
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)

# 3. Disable Low Light Compensation (forces driver to drop resolution/brightness rather than FPS)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1) # Keeps white balance auto

        # 2. Force MJPEG mode BEFORE setting resolution/FPS
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # 3. Request higher FPS & Resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.cap.set(cv2.CAP_PROP_FPS, 30) # Or 30

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.grabbed, self.frame = self.cap.read()
        self.new_frame_available = False
        self.running = True
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap.grab():
                _, self.frame = self.cap.retrieve()
                self.new_frame_available = True  # Mark frame as fresh

    def read(self):
        """Returns (has_new_frame, frame)"""
        if self.new_frame_available:
            self.new_frame_available = False  # Consume the fresh frame
            return True, self.frame
        return False, self.frame  # Frame is a duplicate

    def release(self):
        self.running = False
        self.cap.release()
