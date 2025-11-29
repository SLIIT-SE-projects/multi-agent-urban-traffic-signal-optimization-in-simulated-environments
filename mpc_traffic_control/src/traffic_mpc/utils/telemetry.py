"""
Telemetry Utility.
Buffered CSV writer for high-frequency simulation data.
"""
import csv
import os
from typing import List, Any

class TelemetryRecorder:
    def __init__(self, output_dir: str, filename: str, headers: List[str], buffer_size: int = 100):
        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(output_dir, filename)
        self.buffer = []
        self.buffer_size = buffer_size
        self.headers = headers
        
        # Initialize file with headers
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def record(self, row: List[Any]):
        """Add a row of data to the buffer."""
        self.buffer.append(row)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Write buffered data to disk."""
        if not self.buffer:
            return
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.buffer)
        self.buffer.clear()

    def close(self):
        """Flush remaining data."""
        self.flush()