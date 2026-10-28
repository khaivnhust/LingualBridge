#!/system/bin/sh
# Benchmark mode: disable all non-essential subsystems for maximum performance.
# Run as root: su -c "sh /sdcard/Download/bench_mode.sh"
# WARNING: Disables display output. Use ADB over USB to interact.

LOG=/data/local/tmp/bench_mode.log
echo "=== Benchmark mode started $(date) ===" > $LOG

# ── Helper: run as root ─────────────────────────────────────────────────────
R() { eval "$*" >> $LOG 2>&1; }

# ── 0. Apply settings (before SurfaceFlinger goes down) ─────────────────────
echo "[0] Applying settings..." >> $LOG
R "settings put system screen_brightness 0"
R "settings put system screen_off_timeout 1000"
R "settings put global auto_sync 0"
R "content insert --uri content://settings/system --bind name:s:accelerometer_rotation --bind value:i:0 2>/dev/null; true"

# ── 1. Stop display compositor (SurfaceFlinger) ─────────────────────────────
echo "[1] Stopping SurfaceFlinger..." >> $LOG
R "setprop ctl.stop surfaceflinger"
sleep 1
# Kill any remaining surfaceflinger process
R "pkill -9 surfaceflinger 2>/dev/null; true"

# ── 2. Stop SystemUI ────────────────────────────────────────────────────────
echo "[2] Stopping SystemUI..." >> $LOG
R "am force-stop com.android.systemui"
R "pm disable --user 0 com.android.systemui 2>/dev/null; true"

# ── 3. Disable thermal throttling ───────────────────────────────────────────
echo "[3] Disabling thermal throttling..." >> $LOG
# Stop thermal daemon
R "setprop ctl.stop thermal-engine 2>/dev/null; true"
R "setprop ctl.stop thermal_manager 2>/dev/null; true"
R "setprop ctl.stop thermald 2>/dev/null; true"
R "pkill -9 thermal-engine 2>/dev/null; true"
R "pkill -9 thermal_manager 2>/dev/null; true"
R "pkill -9 thermald 2>/dev/null; true"

# Stop thermal daemon (userspace throttling) but KEEP kernel thermal zones enabled.
# The kernel still protects against overheating (~95C hardware shutdown).
# Only the aggressive OxygenOS thermal policy is removed.
R "setprop ctl.stop thermal-engine 2>/dev/null; true"
R "setprop ctl.stop thermal_manager 2>/dev/null; true"
R "setprop ctl.stop thermald 2>/dev/null; true"
R "pkill -9 thermal-engine 2>/dev/null; true"
R "pkill -9 thermal_manager 2>/dev/null; true"
R "pkill -9 thermald 2>/dev/null; true"
echo "Thermal daemon stopped. Kernel thermal protection remains active." >> $LOG
echo "Monitor temperature during benchmarks. Keep runs under 5 min." >> $LOG

# ── 4. Disable Bluetooth ────────────────────────────────────────────────────
echo "[4] Disabling Bluetooth..." >> $LOG
R "svc bluetooth disable"
R "setprop ctl.stop bluetooth 2>/dev/null; true"

# ── 5. Disable NFC ──────────────────────────────────────────────────────────
echo "[5] Disabling NFC..." >> $LOG
R "svc nfc disable"
R "setprop ctl.stop nfc 2>/dev/null; true"

# ── 6. Disable camera service ───────────────────────────────────────────────
echo "[6] Disabling camera..." >> $LOG
R "setprop ctl.stop cameraserver 2>/dev/null; true"
R "pkill -9 cameraserver 2>/dev/null; true"

# ── 7. Disable sensors ──────────────────────────────────────────────────────
echo "[7] Disabling sensors..." >> $LOG
R "setprop ctl.stop sensorservice 2>/dev/null; true"
R "pkill -9 sensorservice 2>/dev/null; true"

# ── 8. Disable location/GPS ────────────────────────────────────────────────
echo "[8] Disabling location..." >> $LOG
R "settings put secure location_mode 0"
R "setprop ctl.stop gpsd 2>/dev/null; true"

# ── 9. Disable telephony (if not needed) ────────────────────────────────────
echo "[9] Disabling telephony..." >> $LOG
R "setprop ctl.stop rild 2>/dev/null; true"
R "setprop ctl.stop qcrild 2>/dev/null; true"

# ── 10. Disable Wi-Fi (keep USB ADB) ───────────────────────────────────────
echo "[10] Disabling Wi-Fi..." >> $LOG
R "svc wifi disable"

# ── 11. Kill heavy background apps ──────────────────────────────────────────
echo "[11] Killing heavy background apps..." >> $LOG
# Kill Google services (keep only what's needed)
for pkg in com.google.android.gms com.google.android.gsf com.google.android.apps.photos \
           com.google.android.apps.messaging com.google.android.gm com.google.android.youtube \
           com.google.android.apps.maps com.google.android.calendar com.google.android.vending \
           com.google.android.googlequicksearchbox com.google.android.inputmethod.latin \
           com.google.android.providers.media.module com.google.android.deskclock; do
    R "am force-stop $pkg 2>/dev/null; true"
    R "pm disable-user --user 0 $pkg 2>/dev/null; true"
done
# Kill social/media apps
for pkg in com.facebook.katana com.facebook.orca com.facebook.services \
           com.instagram.android com.twitter.android com.tiktok.android \
           com.snapchat.android com.spotify.music com.netflix.mediaclient; do
    R "am force-stop $pkg 2>/dev/null; true"
    R "pm disable-user --user 0 $pkg 2>/dev/null; true"
done
# Kill OnePlus/OPPO bloat
for pkg in com.oneplus.deskclock com.oplus.battery com.oplus.pantanal.ums \
           com.oplus.statistics.rom com.oplus.deepthinker com.heytap.accessory \
           com.heytap.pictorial com.coloros.assistantscreen com.oplus.appdetail \
           com.oneplus.security com.oplus.safecenter; do
    R "am force-stop $pkg 2>/dev/null; true"
    R "pm disable-user --user 0 $pkg 2>/dev/null; true"
done
# Kill any remaining third-party apps
R "pm list packages -3 | sed 's/package://' | while read pkg; do am force-stop \$pkg 2>/dev/null; done"

# ── 12. Kill sync/analytics services ────────────────────────────────────────
echo "[12] Killing sync/analytics..." >> $LOG
R "pkill -9 -f 'com.google.android.gms' 2>/dev/null; true"
R "pkill -9 -f 'com.oneplus' 2>/dev/null; true"
R "pkill -9 -f 'com.qualcomm' 2>/dev/null; true"

# ── 12. Set CPU governor to performance ─────────────────────────────────────
echo "[12] Setting CPU governor to performance..." >> $LOG
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    gov="$cpu/cpufreq/scaling_governor"
    if [ -w "$gov" ]; then
        echo performance > "$gov"
    fi
    # Set min freq to max
    minfreq="$cpu/cpufreq/scaling_min_freq"
    maxfreq="$cpu/cpufreq/scaling_max_freq"
    if [ -r "$maxfreq" ]; then
        maxval=$(cat "$maxfreq" 2>/dev/null)
        if [ -n "$maxval" ] && [ -w "$minfreq" ]; then
            R "echo $maxval > $minfreq"
        fi
    fi
done

# ── 13. Set GPU to performance ──────────────────────────────────────────────
echo "[13] Setting GPU to performance..." >> $LOG
if [ -w /sys/class/kgsl/kgsl-3d0/devfreq/governor ]; then
    R "echo performance > /sys/class/kgsl/kgsl-3d0/devfreq/governor"
fi
if [ -w /sys/class/kgsl/kgsl-3d0/max_gpuclk ]; then
    R "cat /sys/class/kgsl/kgsl-3d0/max_gpuclk > /sys/class/kgsl/kgsl-3d0/gpuclk 2>/dev/null; true"
fi

# ── 14. Disable I/O throttling ──────────────────────────────────────────────
echo "[14] Disabling I/O throttling..." >> $LOG
for q in /sys/block/*/queue; do
    if [ -w "$q/scheduler" ]; then
        R "echo none > $q/scheduler 2>/dev/null; true"
    fi
    if [ -w "$q/nr_requests" ]; then
        R "echo 256 > $q/nr_requests 2>/dev/null; true"
    fi
done

# ── 15. Reduce screen brightness to 0 (if still on) ─────────────────────────
echo "[15] Dimming screen..." >> $LOG

# ── 16. Disable auto-rotate, sync, etc. ─────────────────────────────────────
echo "[16] Disabling auto-sync..." >> $LOG

echo "=== Benchmark mode complete ===" >> $LOG
echo "Device is now in benchmark mode. Display may be off."
echo "WARNING: Thermal zones disabled. Keep runs under 5 min to avoid reboot."
echo "Use 'sh /sdcard/Download/restore.sh' to restore normal operation."
echo "Log: $LOG"
