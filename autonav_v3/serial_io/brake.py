# Import libraries
import time
import serial


class BrakeSerial:

    # Simple serial interface for the brake subsystem
    # 0   = no braking
    # 100 = full braking
    # The Arduino maps this percentage to the actual actuator hardware

    def __init__(self, port: str, baud_rate: int) -> None:
        # Open serial connection to the brake controller
        # The Arduino resets when serial opens, so we wait for it to boot
        self.serial_port = serial.Serial(
            port,
            baudrate=baud_rate,
            timeout=0.05,
        )

        # Give the microcontroller time to finish resetting
        time.sleep(2.0)

    def set_percent(self, brake_percent: int) -> None:
        # Clamp command so we never send invalid brake values
        clamped_brake_percent = max(0, min(100, int(brake_percent)))

        # Convert number to newline-terminated ASCII for the Arduino
        command_string = f"{clamped_brake_percent}\n"

        # Send brake command over serial
        self.serial_port.write(command_string.encode("utf-8"))

        # Flush to ensure the command leaves immediately
        self.serial_port.flush()

    def close(self) -> None:
        # Close serial safely so we do not leave the port locked
        try:
            self.serial_port.close()
        except Exception:
            # Ignore close errors to avoid masking shutdown issues
            pass