import pandas as pd
import random
from datetime import datetime, timedelta

# --- Configuration ---
NUM_ROWS = 200  # Target number of rows (between 100 and 150)
NUM_MACHINES = 6
NUM_SHIFTS_PER_DAY = 2
SHIFT_DURATION_HOURS = 8
SHIFT_DURATION_MINUTES = SHIFT_DURATION_HOURS * 60
# Assume an ideal cycle time for performance calculation in the data generation
ASSUMED_IDEAL_CYCLE_TIME_SECONDS = 30
NUM_MONTHS = 36

# --- Data Generation ---
data = []
start_date = datetime(2023, 1, 1)  # Start date for timestamps

# Explicitly define the column order to match the requirement
column_names = [
    "id",
    "machine_id",
    "shift",
    "timestamp",
    "off_time",
    "idle_time",
    "production_time",
    "total_parts",
    "good_parts",
    "bad_parts",
    "availability",
    "performance",
    "quality",
    "availability_loss",
    "performance_loss",
    "quality_loss",
    "oee",
    "update_date",  # Changed updatedate to update_date
]


for i in range(NUM_ROWS):
    # Basic Identifiers
    row_id = i + 1
    machine_id = random.randint(1, NUM_MACHINES)
    # Assign shifts based on 8-hour blocks within a day for some realism
    # Simple approach: shift 1 starts around 6-8 AM, shift 2 starts around 2-4 PM
    # More complex but realistic: link timestamp to shift
    # Let's keep it simple for synthetic data: just assign shift 1 or 2 randomly for the timestamp
    shift = random.randint(1, NUM_SHIFTS_PER_DAY)

    # Timestamps
    random_days = random.randint(0, NUM_MONTHS * 30)  # Data over  num_months
    # Generate a timestamp within an 8-hour window on a random day
    random_seconds_in_day = random.randint(0, 24 * 3600)
    timestamp = start_date + timedelta(days=random_days, seconds=random_seconds_in_day)

    # Calculate times relative to an 8-hour planned shift duration
    # Planned Production Time for this entry is 8 hours
    planned_production_minutes = SHIFT_DURATION_MINUTES

    # Generate idle and off time within the planned time
    # Ensure sum of off + idle is less than planned
    max_downtime_minutes = planned_production_minutes * random.uniform(
        0.05, 0.25
    )  # 5% to 25% downtime
    off_minutes = random.uniform(
        0, max_downtime_minutes * random.uniform(0.2, 0.8)
    )  # Allocate part of downtime to off
    idle_minutes = max_downtime_minutes - off_minutes  # Remaining downtime is idle

    production_minutes = planned_production_minutes - (off_minutes + idle_minutes)
    production_minutes = max(
        0, production_minutes
    )  # Should not be negative with this logic

    # Convert minutes to HH:MM:SS string format
    def format_minutes_to_hhmmss(total_minutes):
        if total_minutes < 0:
            total_minutes = 0  # Handle potential floating point negatives
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        seconds = int((total_minutes * 60) % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    off_time_str = format_minutes_to_hhmmss(off_minutes)
    idle_time_str = format_minutes_to_hhmmss(idle_minutes)
    production_time_str = format_minutes_to_hhmmss(production_minutes)

    # Parts Count
    # Base parts on production time and assumed ideal cycle time
    # Ideal parts = production_minutes * 60 / ASSUMED_IDEAL_CYCLE_TIME_SECONDS
    # Actual parts might be less than ideal based on 'performance'
    # Let's generate total parts and bad parts, then calculate good parts
    # Total parts can be loosely related to production time but add variation
    max_possible_parts = int(
        production_minutes
        * 60
        / ASSUMED_IDEAL_CYCLE_TIME_SECONDS
        * random.uniform(0.8, 1.1)
    )  # Factor in some performance effect + variation
    total_parts = (
        random.randint(int(max_possible_parts * 0.7), max_possible_parts)
        if max_possible_parts > 0
        else 0
    )
    total_parts = max(10, total_parts)  # Ensure minimum parts if time is > 0
    bad_parts = random.randint(
        0, int(total_parts * random.uniform(0.01, 0.08))
    )  # 1% to 8% bad parts
    good_parts = total_parts - bad_parts
    good_parts = max(0, good_parts)  # Ensure good parts is not negative

    # OEE Calculation Components (Availability, Performance, Quality)
    # Calculate these here so they are stored in the CSV
    # Availability = Running Time / Planned Production Time
    # Running Time = Production Time (from synthetic data)
    availability = (
        production_minutes / planned_production_minutes
        if planned_production_minutes > 0
        else 0.0
    )
    availability = min(1.0, availability)  # Cap at 100%

    # Performance = (Ideal Cycle Time * Total Count) / Running Time
    # Ideal Run Time = Total Parts * ASSUMED_IDEAL_CYCLE_TIME_SECONDS / 60 (minutes)
    ideal_run_time_minutes = (total_parts * ASSUMED_IDEAL_CYCLE_TIME_SECONDS) / 60.0
    performance = (
        ideal_run_time_minutes / production_minutes if production_minutes > 0 else 0.0
    )
    performance = min(1.0, performance)  # Cap at 100%

    # Quality = Good Count / Total Count
    quality = good_parts / total_parts if total_parts > 0 else 0.0
    quality = min(1.0, quality)  # Cap at 100%

    # OEE
    oee = availability * performance * quality
    oee = min(1.0, oee)  # Cap at 100%

    # Loss Components
    availability_loss = 1.0 - availability
    performance_loss = 1.0 - performance
    quality_loss = 1.0 - quality

    # Update Date (slightly after timestamp)
    update_date = timestamp + timedelta(seconds=random.randint(60, 3600))

    # Append row data in the specified order
    data.append(
        [
            row_id,
            machine_id,
            shift,
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),  # Format timestamp as string
            off_time_str,
            idle_time_str,
            production_time_str,
            total_parts,
            good_parts,
            bad_parts,
            round(availability, 4),  # Store calculated A, P, Q, OEE
            round(performance, 4),
            round(quality, 4),
            round(availability_loss, 4),
            round(performance_loss, 4),
            round(quality_loss, 4),
            round(oee, 4),
            update_date.strftime("%Y-%m-%d %H:%M:%S"),  # Format updatedate as string
        ]
    )

# --- Create DataFrame ---
# Use the explicitly defined column_names list
df = pd.DataFrame(data, columns=column_names)

# --- Store as CSV ---
csv_filename = "synthetic_machine_data.csv"
df.to_csv(csv_filename, index=False)

print(f"Successfully generated {len(df)} rows of data and saved to '{csv_filename}'")