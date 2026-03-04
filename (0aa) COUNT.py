


import time
from datetime import datetime, timedelta

def format_duration(seconds):
    """Converts seconds to HH:MM:SS format."""
    seconds = int(seconds) # Ensure integer for divmod
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def get_next_1am():
    """Get the next 1 AM datetime."""
    now = datetime.now()
    next_1am = now.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # If it's already past 1 AM today, get 1 AM tomorrow
    if now >= next_1am:
        next_1am += timedelta(days=1)
    
    return next_1am

# --- Configuration ---
INTERVAL_SECONDS = 10
# -------------------

# Calculate target time
target_time = get_next_1am()
current_time = datetime.now()
total_seconds_remaining = (target_time - current_time).total_seconds()

print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Target time (next 1 AM): {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total time remaining: {format_duration(total_seconds_remaining)}")
print(f"Counting down in {INTERVAL_SECONDS} second increments.")
print("Press Ctrl+C to stop early.")
print("-" * 50)

start_time = time.time()

try:
    increment = 1
    while True:
        current_time = datetime.now()
        
        # Check if we've reached 1 AM
        if current_time >= target_time:
            print(f"\n🎉 It's 1 AM! Target reached at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            break
        
        # Calculate remaining time
        remaining_seconds = (target_time - current_time).total_seconds()
        elapsed_seconds = time.time() - start_time
        
        print(f"Increment: {increment:>5} | "
              f"Time Remaining: {format_duration(remaining_seconds)} | "
              f"Elapsed: {format_duration(elapsed_seconds)} | "
              f"Current: {current_time.strftime('%H:%M:%S')}")
        
        increment += 1
        time.sleep(INTERVAL_SECONDS)
    
    print("\n" + "-" * 50)
    print("--- Countdown to 1 AM completed! ---")

except KeyboardInterrupt:
    print("\n" + "-" * 50)
    print("Timer stopped manually by user.")
finally:
    end_time = time.time()
    actual_time_run = end_time - start_time
    final_time = datetime.now()
    print(f"Final time: {final_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Actual script run time: {format_duration(actual_time_run)}")