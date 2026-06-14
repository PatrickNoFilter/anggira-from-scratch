"""Simulate cron: 3 bursts with 60s gaps (like cron schedule)."""
import sys, os, time, subprocess
script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'burst_train.py')
for i in range(3):
    t0 = time.time()
    r = subprocess.run(['python3', '-u', script], capture_output=True, text=True, timeout=70, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    elapsed = time.time() - t0
    last_line = r.stdout.strip().split('\n')[-1] if r.stdout else ''
    for line in r.stdout.split('\n'):
        if 'Burst done' in line:
            print(f'Run {i+1}: {line.strip()} | wall={elapsed:.1f}s')
    if i < 2:
        print(f'  → cooling 60s...')
        time.sleep(60)
