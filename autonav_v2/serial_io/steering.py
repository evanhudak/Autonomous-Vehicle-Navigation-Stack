# Import libraries
import serial


class SteeringSerial:

    # Serial interface for the steering subsystem
    # S:0   = full left
    # S:50  = centered
    # S:100 = full right
    # The Arduino maps this percentage into a physical wheel angle

    def __init__(self, port: str, baud_rate: int) -> None:
        # Open serial connection to the steering controller
        # No delay required here unless firmware requires reset time
        self.serial_port = serial.Serial(
            port,
            baudrate=baud_rate,
            timeout=0.05,
        )

    def set_percent(self, steering_percent: int) -> None:
        # Clamp steering command so we never send invalid values
        clamped_steering_percent = max(0, min(100, int(steering_percent)))

        # Build protocol string expected by the Arduino firmware
        command_string = f"S:{clamped_steering_percent}\n"

        # Send steering command over serial
        self.serial_port.write(command_string.encode("utf-8"))

        # Flush to ensure the command is transmitted immediately
        self.serial_port.flush()

    def close(self) -> None:
        # Close serial safely so the port is not left in use
        try:
            self.serial_port.close()
        except Exception:
            # Ignore close errors to avoid masking shutdown issues
            pass