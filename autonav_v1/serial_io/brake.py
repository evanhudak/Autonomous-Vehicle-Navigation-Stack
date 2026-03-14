# import libraries
import time
import serial

class BrakeSerial:

    """
    Serial interface for the brake subsystem.

    Expects simple numeric commands over serial:
        0   -> no braking
        100 -> full braking

    The Arduino side is responsible for mapping percentage commands
    to the actual brake actuator hardware.
    """

    def __init__(self, port: str, baud_rate: int):
        # Open serial connection to the brake controller.
        # A short delay is required to allow the Arduino to reset.
        self.serial_port = serial.Serial(port,
                                         baudrate=baud_rate,
                                         timeout=0.05)
        time.sleep(2.0)

    def set_percent(self, brake_percent: int) -> None:
        # Clamp brake command to valid range [0, 100].
        clamped_percent = max(0, min(100, int(brake_percent)))

        # Send brake command as a newline-terminated ASCII number.
        self.serial_port.write(f"{clamped_percent}\n".encode("utf-8"))
        self.serial_port.flush()

    def close(self) -> None:
        # Safely close the serial port if it is open.
        try:
            self.serial_port.close()
        except Exception:
            pass