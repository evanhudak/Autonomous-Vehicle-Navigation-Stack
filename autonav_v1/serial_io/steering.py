# import library
import serial

class SteeringSerial:

    """
    Serial interface for the steering subsystem.

    Expects prefixed commands over serial:
        S:0   -> full left
        S:50  -> centered steering
        S:100 -> full right

    The Arduino firmware is responsible for mapping these percentages
    to actual wheel angles (e.g., -45° to +45°).
    """

    def __init__(self, port: str, baud_rate: int):
        # Open serial connection to the steering controller.
        self.serial_port = serial.Serial(port,
                                         baudrate=baud_rate,
                                         timeout=0.05)

    def set_percent(self, steering_percent: int) -> None:
        # Clamp steering command to valid range [0, 100].
        clamped_percent = max(0, min(100, int(steering_percent)))

        # Send steering command using prefixed protocol.
        self.serial_port.write(f"S:{clamped_percent}\n".encode("utf-8"))
        self.serial_port.flush()

    def close(self) -> None:
        # Safely close the serial port if it is open.
        try:
            self.serial_port.close()
        except Exception:
            pass