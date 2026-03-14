# import libraries
import time
import serial

class ThrottleSerial:

    """
    Serial interface for the throttle subsystem.

    Expects numeric commands over serial:
        0   -> idle throttle
        100 -> full throttle

    The Arduino firmware handles stepper motor control and safety logic.
    """

    def __init__(self, port: str, baud_rate: int):
        # Open serial connection to the throttle controller.
        self.serial_port = serial.Serial(port,
                                         baudrate=baud_rate,
                                         timeout=0.05)

        # Allow time for Arduino reset after serial connection opens.
        time.sleep(2.0)

        # Switch controller into autonomous mode.
        self.serial_port.write(b"AUTO\n")
        self.serial_port.flush()

    def set_percent(self, throttle_percent: int) -> None:
        # Clamp throttle command to valid range [0, 100].
        clamped_percent = max(0, min(100, int(throttle_percent)))

        # Send throttle command as a newline-terminated ASCII number.
        self.serial_port.write(f"{clamped_percent}\n".encode("utf-8"))
        self.serial_port.flush()

    def stop(self) -> None:
        # Immediately halt throttle motion (supported by stepper controller).
        self.serial_port.write(b"STOP\n")
        self.serial_port.flush()

    def close(self) -> None:
        # Safely close the serial port if it is open.
        try:
            self.serial_port.close()
        except Exception:
            pass