import re
from datetime import datetime
import numpy as np

def parse_timestamp(line):
    ts = line.split(" - ")[0]
    return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S,%f')

def parse_log_file(file_path):
    """Parse log file for tstep and data transport events."""
    events = []
    # Handle both "dt time: " and "dt time:" formats
    pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?tstep time: ([\d\.]+), dt time:\s*([\d\.]+)')
    with open(file_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                timestamp = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f')
                tstep_time = float(m.group(2))
                actual_dt_time = float(m.group(3))
                events.append({
                    'timestamp': timestamp,
                    'tstep_time': tstep_time,
                    'actual_dt_time': actual_dt_time
                })
    return events

def parse_log_file_for_io(file_path):
    pattern1 = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?Write the data')
    pattern2 = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?Read data')
    pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*?tstep time: ([\d\.]+), dt time:\s*([\d\.]+)')
    events = []
    with open(file_path, 'r') as f:
        num_dt = 0
        for line in f:
            if pattern1.match(line):
                num_dt += 1
            elif  pattern2.match(line):
                num_dt += 2
            m = pattern.search(line)
            if m:
                timestamp = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f')
                tstep_time = float(m.group(2))
                actual_dt_time = float(m.group(3))
                if actual_dt_time > 10.0:
                    continue
                events.append({
                    'timestamp': timestamp,
                    'tstep_time': tstep_time,
                    'actual_dt_time': actual_dt_time,
                    "num_dt": num_dt
                })
                num_dt = 0
    return events

def compute_io_throughput(file_path, num_elements, dtype):
    """
    Compute IO throughput based on log file events.
    
    Args:
        file_path: Path to the log file
        num_elements: Number of array elements transferred
        dtype: Data type of the array elements (e.g., np.float32, np.float64, 'float32', 'float64')
    
    Returns:
        Dictionary containing throughput metrics
    """
    events = parse_log_file_for_io(file_path)
    
    if not events:
        return {'throughput_MB_per_sec': 0, 'total_data_MB': 0, 'total_time_sec': 0}
    
    # Convert dtype to numpy dtype if it's a string
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)
    elif hasattr(dtype, 'dtype'):
        dtype = dtype.dtype
    
    # Calculate data size in bytes
    element_size_per_write = (dtype.itemsize * num_elements)/(1024 * 1024)  # Convert to MB

    count = 0
    mean_throughput_MB_per_sec = 0
    total_data_MB = 0
    for event in events:
        if event["num_dt"] > 0:
            # Calculate total data transferred in MB
            total_data_MB += element_size_per_write * event['num_dt']
            mean_throughput_MB_per_sec += (element_size_per_write * event['num_dt'] / event['actual_dt_time'] if event['actual_dt_time'] > 0 else 0)
            count += 1
    if count > 0:
        mean_throughput_MB_per_sec /= count

    # Calculate total time for all data transport operations
    total_dt_time = sum(event['actual_dt_time'] for event in events)
    
    return {
        'mean_throughput_MB_per_sec': mean_throughput_MB_per_sec,
        'total_data_MB': total_data_MB,
        'total_time_sec': total_dt_time,
        'total_num_dt': sum([e["num_dt"] for e in events]),
        'avg_time_per_dt': total_dt_time / sum([e["num_dt"] for e in events])
    }

def get_runtime(file_path):
    events = parse_log_file(file_path)
    return (events[-1]['timestamp'] - events[0]['timestamp']).total_seconds() if events else 0