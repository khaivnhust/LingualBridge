#!/system/bin/sh
# Restore normal operation after benchmark mode.
# Run as root: su -c "sh /sdcard/Download/restore.sh"

LOG=/data/local/tmp/restore.log
echo "=== Restore started $(date) ===" > $LOG

R() { eval "$*" >> $LOG 2>&1; }

# ── 0. Restore settings (before SurfaceFlinger comes back) ──────────────────
echo "[0] Restoring settings..." >> $LOG
R "settings put system screen_brightness 128"
R "settings put system screen_off_timeout 30000"
R "settings put global auto_sync 1"

# ── 1. Re-enable SurfaceFlinger ─────────────────────────────────────────────
echo "[1] Re-enabling SurfaceFlinger..." >> $LOG
R "pkill -9 bootanim 2>/dev/null; true"
sleep 1
R "setprop ctl.start surfaceflinger"

# ── 2. Re-enable SystemUI ───────────────────────────────────────────────────
echo "[2] Re-enabling SystemUI..." >> $LOG
R "pm enable --user 0 com.android.systemui 2>/dev/null; true"
R "pm enable com.android.systemui 2>/dev/null; true"
sleep 1
R "am start-service -n com.android.systemui/.SystemUIService 2>/dev/null; true"
R "am start -n com.android.launcher/com.android.launcher.Launcher 2>/dev/null; true"

# ── 3. Re-enable thermal management ─────────────────────────────────────────
echo "[3] Re-enabling thermal..." >> $LOG
R "setprop ctl.start thermal-engine 2>/dev/null; true"
R "setprop ctl.start thermal_manager 2>/dev/null; true"
R "setprop ctl.start thermald 2>/dev/null; true"

# ── 4. Re-enable Bluetooth ──────────────────────────────────────────────────
echo "[4] Re-enabling Bluetooth..." >> $LOG
R "svc bluetooth enable"

# ── 5. Re-enable NFC ────────────────────────────────────────────────────────
echo "[5] Re-enabling NFC..." >> $LOG
R "svc nfc enable"

# ── 6. Re-enable camera ─────────────────────────────────────────────────────
echo "[6] Re-enabling camera..." >> $LOG
R "setprop ctl.start cameraserver 2>/dev/null; true"

# ── 7. Re-enable sensors ────────────────────────────────────────────────────
echo "[7] Re-enabling sensors..." >> $LOG
R "setprop ctl.start sensorservice 2>/dev/null; true"

# ── 8. Re-enable location ───────────────────────────────────────────────────
echo "[8] Re-enabling location..." >> $LOG
R "settings put secure location_mode 3"

# ── 9. Re-enable telephony ──────────────────────────────────────────────────
echo "[9] Re-enabling telephony..." >> $LOG
R "setprop ctl.start rild 2>/dev/null; true"
R "setprop ctl.start qcrild 2>/dev/null; true"

# ── 10. Re-enable Wi-Fi ─────────────────────────────────────────────────────
echo "[10] Re-enabling Wi-Fi..." >> $LOG
R "svc wifi enable"

# ── 11. Restore CPU governor ────────────────────────────────────────────────
echo "[11] Restoring CPU governor..." >> $LOG
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    gov="$cpu/cpufreq/scaling_governor"
    if [ -w "$gov" ]; then
        echo walt > "$gov" 2>/dev/null
    fi
    minfreq="$cpu/cpufreq/scaling_min_freq"
    if [ -w "$minfreq" ]; then
        echo 0 > "$minfreq" 2>/dev/null
    fi
done

# ── 12. Restore GPU governor ────────────────────────────────────────────────
echo "[12] Restoring GPU governor..." >> $LOG
if [ -w /sys/class/kgsl/kgsl-3d0/devfreq/governor ]; then
    R "echo msm-adreno-tz > /sys/class/kgsl/kgsl-3d0/devfreq/governor 2>/dev/null; true"
fi

# ── 13. Restore screen ──────────────────────────────────────────────────────
echo "[13] Restoring screen..." >> $LOG

echo "=== Restore complete ===" >> $LOG
echo "Device restored to normal operation."
echo "Log: $LOG"
