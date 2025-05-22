import re
import statistics

def calculate_mean_dt_time(log_file_path):
    # Regular expression to extract dt time values
    dt_pattern = re.compile(r'dt time: (\d+\.\d+)')
    dt_times = []
    
    # Read the log file and extract dt times
    with open(log_file_path, 'r') as f:
        for line in f:
            match = dt_pattern.search(line)
            if match:
                dt_time = float(match.group(1))
                # Only include non-zero dt times
                if dt_time > 0:
                    dt_times.append(dt_time)
    
    # Calculate statistics
    if dt_times:
        mean_dt = statistics.mean(dt_times)
        count_dt = len(dt_times)
        total_entries = sum(1 for _ in open(log_file_path))
        
        # print(f"Total log entries: {total_entries}")
        # print(f"Non-zero dt times: {count_dt}")
        # print(f"Mean dt time: {mean_dt:.6f} seconds")
        
        # Optional: print all non-zero dt times
        # print(f"All non-zero dt times: {dt_times}")
        
        return mean_dt
    else:
        print("No valid dt times found in the log file.")
        return None


def compute_throughput(log_file_path:str,nranks:int,data_size_per_rank:int):
    mean_dt = calculate_mean_dt_time(log_file_path)
    return (nranks*data_size_per_rank*4/mean_dt/1024/1024/1024) ##GB/s

if __name__ == "__main__":
    logfiles = [
                "logs_all_colocated_2node_per_component/sim_0.log",
                "logs_db_and_ai_colocated_1node_per_component/sim_0.log",
                "logs_pfs_1_node_per_component/sim_0.log",
                "logs_all_colocated_2node_per_component/AI.log",
                "logs_db_and_ai_colocated_1node_per_component/AI.log",
                "logs_pfs_1_node_per_component/AI.log"
                ]
    nranks = [24,12,12,24,12,12]
    for i in range(len(logfiles)):
        print(f"logfile {logfiles[i]}")
        t=compute_throughput(logfiles[i],nranks[i],4000000)
        print(f"Throughput {t}GB/S")