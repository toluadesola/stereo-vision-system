import subprocess
import time

last = {}
COOLDOWN = 3

def notify(label, dist):

    now = time.time()

    if label in last and now - last[label] < COOLDOWN:
        return

    last[label] = now

    msg = f"{label} {dist:.1f} meters"

    print("ALERT:", msg)

    subprocess.Popen(["espeak", msg],
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
