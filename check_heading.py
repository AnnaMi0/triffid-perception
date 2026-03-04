#!/usr/bin/env python3
"""Refined heading analysis - focused checks."""

import sqlite3, math, os
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, Imu


def quat_to_yaw(q):
    siny = 2.0*(q.w*q.z + q.x*q.y)
    cosy = 1.0 - 2.0*(q.y**2 + q.z**2)
    return math.atan2(siny, cosy)


def stamp_sec(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9


def load_all(db_path, topic, msg_type, limit=None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id FROM topics WHERE name = ?", (topic,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return []
    tid = row[0]
    q = "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp"
    if limit:
        q += f" LIMIT {limit}"
    cur.execute(q, (tid,))
    msgs = []
    for (blob,) in cur:
        try:
            msgs.append(deserialize_message(blob, msg_type))
        except:
            pass
    conn.close()
    return msgs


DB1 = '/home/triffid/hua_ws/rosbag2_active_20260220_164455/rosbag2_active_20260220_164455_0.db3'
DB2 = '/home/triffid/hua_ws/rosbag2_active_20260220_170131/rosbag2_active_20260220_170131_0.db3'


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 1: High-resolution stationary yaw drift (Bag 1, first 15s)
# ═══════════════════════════════════════════════════════════════════════════
print("="*72)
print("  CHECK 1: Stationary yaw drift (Bag 1, first 15s, ALL messages)")
print("="*72)

odom1 = load_all(DB1, '/dog_odom', Odometry, limit=7500)  # ~15s at 500Hz
t0 = stamp_sec(odom1[0])

# Find when robot starts moving (position changes > 1cm/s)
stationary_end = 0
for i in range(1, len(odom1)):
    dt = stamp_sec(odom1[i]) - stamp_sec(odom1[i-1])
    if dt > 0:
        p1 = odom1[i-1].pose.pose.position
        p2 = odom1[i].pose.pose.position
        speed = math.sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2) / dt
        if speed > 0.05:  # 5cm/s
            stationary_end = i
            break

elapsed_stat = stamp_sec(odom1[stationary_end]) - t0
print(f"  Robot starts moving at: {elapsed_stat:.2f}s (sample {stationary_end})")

# Measure yaw during stationary period
yaw_start = math.degrees(quat_to_yaw(odom1[0].pose.pose.orientation))
yaw_at_end = math.degrees(quat_to_yaw(odom1[stationary_end-1].pose.pose.orientation))
delta = yaw_at_end - yaw_start
while delta > 180: delta -= 360
while delta < -180: delta += 360

drift_per_sec = delta / elapsed_stat if elapsed_stat > 0 else 0
drift_per_hour = drift_per_sec * 3600

print(f"  Stationary duration:  {elapsed_stat:.2f}s")
print(f"  Yaw at t=0:           {yaw_start:.6f}°")
print(f"  Yaw at t={elapsed_stat:.1f}:       {yaw_at_end:.6f}°")
print(f"  Δyaw:                 {delta:+.6f}°")
print(f"  Drift rate:           {drift_per_sec:+.6f} °/s")
print(f"  Drift per hour:       {drift_per_hour:+.2f} °/h")
print()
print(f"  Interpretation:")
print(f"    < 0.5°/h → very stable, likely magnetometer-fused")
print(f"    1-10°/h  → typical MEMS gyro drift (NO magnetometer)")
print(f"    > 10°/h  → poor gyro or uncalibrated")

# Sample every 1s during stationary period
print(f"\n  Yaw samples during stationary (every ~1s):")
print(f"  {'Time':>6}  {'Yaw°':>12}  {'Δ from start':>14}")
next_t = 0.0
for i in range(stationary_end):
    elapsed = stamp_sec(odom1[i]) - t0
    if elapsed >= next_t:
        yd = math.degrees(quat_to_yaw(odom1[i].pose.pose.orientation))
        print(f"  {elapsed:6.2f}  {yd:12.6f}  {yd - yaw_start:+14.6f}")
        next_t += 1.0


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 2: IMU orientation covariance
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  CHECK 2: IMU orientation covariance")
print("="*72)

imu = load_all(DB2, '/dog_imu_raw', Imu, limit=1)
if imu:
    m = imu[0]
    cov = m.orientation_covariance
    print(f"  orientation_covariance[0] = {cov[0]}")
    if cov[0] == -1:
        print("  → -1 means orientation is INVALID/not available (raw gyro readings)")
        print("    This strongly suggests NO magnetometer fusion.")
    elif cov[0] == 0:
        print("  → 0 covariance: either not set or perfectly known")
    else:
        print(f"  → Orientation covariance available, suggesting fused estimate")
    print(f"  Full covariance: {list(cov)}")
else:
    print("  No /dog_imu_raw messages found")


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 3: Longer-baseline GPS heading (Bag 2)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  CHECK 3: GPS heading (long baselines, > 10s apart)")
print("="*72)

gps = load_all(DB2, '/fix', NavSatFix)
odom2_all = load_all(DB2, '/dog_odom', Odometry, limit=500)  # ~1s worth for initial check
# Load sparse odom for lookup
odom2_sparse = []
conn = sqlite3.connect(DB2)
cur = conn.cursor()
cur.execute("SELECT id FROM topics WHERE name = '/dog_odom'")
tid = cur.fetchone()[0]
cur.execute("SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp", (tid,))
idx = 0
for (blob,) in cur:
    if idx % 250 == 0:  # ~2Hz from 500Hz
        try:
            odom2_sparse.append(deserialize_message(blob, Odometry))
        except:
            pass
    idx += 1
conn.close()

if gps:
    print(f"  GPS fixes: {len(gps)}")
    gps_t0 = stamp_sec(gps[0])
    
    # Use baselines of at least 10s
    print(f"\n  {'t_start':>8}  {'t_end':>8}  {'GPS_hdg°':>10}  {'Odom_yaw°':>10}  {'Diff°':>8}  {'Dist_m':>8}")
    
    diffs_clean = []
    for i in range(len(gps)):
        for j in range(i+1, len(gps)):
            dt = stamp_sec(gps[j]) - stamp_sec(gps[i])
            if dt < 10:
                continue
            if dt > 30:
                break
            
            dlat = gps[j].latitude - gps[i].latitude
            dlon = gps[j].longitude - gps[i].longitude
            dlat_m = dlat * 111320
            cos_lat = math.cos(math.radians(gps[j].latitude))
            dlon_m = dlon * 111320 * cos_lat
            dist = math.sqrt(dlat_m**2 + dlon_m**2)
            
            if dist < 2.0:  # too small for reliable heading
                continue
            
            # Speed sanity check: walking robot < 3 m/s
            speed = dist / dt
            if speed > 3.0:
                continue
            
            # GPS heading ENU: atan2(north, east)
            gps_hdg = math.degrees(math.atan2(dlat_m, dlon_m))
            
            # Midpoint time for odom lookup
            mid_t = (stamp_sec(gps[i]) + stamp_sec(gps[j])) / 2
            best = min(odom2_sparse, key=lambda m: abs(stamp_sec(m) - mid_t))
            odom_yaw = math.degrees(quat_to_yaw(best.pose.pose.orientation))
            
            diff = gps_hdg - odom_yaw
            while diff > 180: diff -= 360
            while diff < -180: diff += 360
            diffs_clean.append(diff)
            
            t1 = stamp_sec(gps[i]) - gps_t0
            t2 = stamp_sec(gps[j]) - gps_t0
            print(f"  {t1:8.1f}  {t2:8.1f}  {gps_hdg:10.1f}  {odom_yaw:10.1f}  {diff:+8.1f}  {dist:8.1f}")
    
    if diffs_clean:
        mean = sum(diffs_clean) / len(diffs_clean)
        std = (sum((d-mean)**2 for d in diffs_clean) / len(diffs_clean))**0.5
        print(f"\n  Mean diff: {mean:+.1f}°  StdDev: {std:.1f}°")
        print(f"  If StdDev < 20°: GPS and odom are correlated → magnetometer likely")
        print(f"  If StdDev > 60°: No correlation → gyro-only likely")
    else:
        print("  No valid long-baseline GPS heading pairs found")
        print("  (GPS too noisy or robot speed > 3 m/s)")

    # Overall trajectory heading
    if len(gps) > 1:
        dlat = gps[-1].latitude - gps[0].latitude
        dlon = gps[-1].longitude - gps[0].longitude
        dlat_m = dlat * 111320
        dlon_m = dlon * 111320 * math.cos(math.radians(gps[0].latitude))
        dist = math.sqrt(dlat_m**2 + dlon_m**2)
        hdg = math.degrees(math.atan2(dlat_m, dlon_m))
        print(f"\n  Overall GPS trajectory (first→last fix):")
        print(f"    Heading: {hdg:.1f}° (ENU), Distance: {dist:.1f}m")


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 4: Search for magnetometer-related topics
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  CHECK 4: Magnetometer/heading-related topics")
print("="*72)

conn = sqlite3.connect(DB2)
cur = conn.cursor()
cur.execute("SELECT name, type FROM topics")
for name, typ in cur:
    low = name.lower()
    if any(k in low for k in ['mag', 'heading', 'compass', 'north', 'orient']):
        cur2 = conn.cursor()
        cur2.execute(
            "SELECT COUNT(*) FROM messages WHERE topic_id = "
            "(SELECT id FROM topics WHERE name = ?)", (name,)
        )
        cnt = cur2.fetchone()[0]
        print(f"  {name:<50s}  type={typ:<45s}  msgs={cnt}")
conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 5: Bag 2 stationary period drift
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  CHECK 5: Stationary yaw drift (Bag 2)")
print("="*72)

odom2_start = load_all(DB2, '/dog_odom', Odometry, limit=15000)  # ~30s
t0_2 = stamp_sec(odom2_start[0])

# Find stationary period
stat_end = 0
for i in range(1, len(odom2_start)):
    dt = stamp_sec(odom2_start[i]) - stamp_sec(odom2_start[i-1])
    if dt > 0:
        p1 = odom2_start[i-1].pose.pose.position
        p2 = odom2_start[i].pose.pose.position
        speed = math.sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2) / dt
        if speed > 0.05:
            stat_end = i
            break

if stat_end > 100:
    elapsed = stamp_sec(odom2_start[stat_end-1]) - t0_2
    y0 = math.degrees(quat_to_yaw(odom2_start[0].pose.pose.orientation))
    yf = math.degrees(quat_to_yaw(odom2_start[stat_end-1].pose.pose.orientation))
    dy = yf - y0
    while dy > 180: dy -= 360
    while dy < -180: dy += 360
    rate = dy / elapsed if elapsed > 0 else 0
    print(f"  Stationary until: {elapsed:.2f}s")
    print(f"  Yaw start: {y0:.6f}°  Yaw end: {yf:.6f}°")
    print(f"  Δyaw: {dy:+.6f}°")
    print(f"  Drift rate: {rate*3600:+.2f} °/h")
else:
    print(f"  Robot starts moving immediately (stat_end={stat_end})")
    print(f"  Cannot measure stationary drift in bag 2")
    # Still show first few yaw values
    print(f"  First 5 yaw values:")
    for i in range(min(5, len(odom2_start))):
        y = math.degrees(quat_to_yaw(odom2_start[i].pose.pose.orientation))
        print(f"    t={stamp_sec(odom2_start[i]) - t0_2:.4f}s  yaw={y:.4f}°")


# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  FINAL ASSESSMENT")
print("="*72)
