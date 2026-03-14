# Import libraries
import time
import serial


class ThrottleSerial:

    # Serial interface for the throttle subsystem
    # 0   = idle throttle
    # 100 = full throttle
    # The Arduino firmware handles stepper motion and safety logic

    def __init__(self, port: str, baud_rate: int) -> None:
        # Open serial connection to the throttle controller
        # Opening the port resets the Arduino, so we wait before sending commands
        self.serial_port = serial.Serial(
            port,
            baudrate=baud_rate,
            timeout=0.05,
        )

        # Allow the microcontroller to finish resetting
        time.sleep(2.0)

        # Put controller into autonomous mode so it listens to serial commands
        self.serial_port.write(b"AUTO\n")
        self.serial_port.flush()

    def set_percent(self, throttle_percent: int) -> None:
        # Clamp throttle so we never send out-of-range commands
        clamped_throttle_percent = max(0, min(100, int(throttle_percent)))

        # Convert numeric value into newline-terminated ASCII string
        command_string = f"{clamped_throttle_percent}\n"

        # Send throttle command over serial
        self.serial_port.write(command_string.encode("utf-8"))

        # Flush immediately to avoid buffered delays
        self.serial_port.flush()

    def stop(self) -> None:
        # Send emergency stop command supported by the stepper controller
        self.serial_port.write(b"STOP\n")
        self.serial_port.flush()

    def close(self) -> None:
        # Close serial safely so the port is released properly
        try:
            self.serial_port.close()
        except Exception:
            # Ignore close errors to keep shutdown robust
            pass