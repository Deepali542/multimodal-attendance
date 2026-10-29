"""
Analytics Subsystem - Session Logging and Data Persistence.
Stores timestamped classroom attendance and engagement records locally.
"""

from typing import List, Dict, Any, Optional
import json
import csv
import os
import time
from datetime import datetime


class SessionDataLogger:
    """
    Manages local storage of session logs, attendance records, and multimodal timeline metrics.
    Ensures ethical, privacy-preserving local storage without external leakage.
    """

    def __init__(self, output_dir: str = "session_records"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.current_session_id = datetime.now().strftime("Session_%Y%m%d_%H%M%S")

    def export_attendance_csv(self, attendance_records: List[Dict], filename: Optional[str] = None) -> str:
        """
        Exports verified attendance to a CSV spreadsheet.
        """
        if filename is None:
            filename = os.path.join(self.output_dir, f"{self.current_session_id}_attendance.csv")

        fields = [
            "participant_id",
            "status",
            "first_seen",
            "last_seen",
            "total_present_minutes",
            "presence_percentage",
            "sustained_verified",
            "attendance_timestamp"
        ]

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in attendance_records:
                filtered_row = {k: row.get(k, "") for k in fields}
                writer.writerow(filtered_row)

        return filename

    def export_session_json(self, session_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Exports full multimodal session analytics into structured JSON.
        """
        if filename is None:
            filename = os.path.join(self.output_dir, f"{self.current_session_id}_analytics.json")

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)

        return filename
