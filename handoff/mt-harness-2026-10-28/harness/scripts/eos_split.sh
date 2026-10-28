#!/system/bin/sh
# eos_split.sh -- ASR split across devices: audio encoder (mmproj) on the Adreno
# (OpenCL), decoder on the Hexagon NPU, MT also on the NPU, TTS on CPU.
# Policy G (WORKLOG §17), 80 clips, paced like a conversation and thermally
# guarded: >= 3 s between clips, and wait until NSP < 85 C and CPU < 90 C before
# each one (never run the phone hot -- see memory no-device-overheat).
# Per-clip line in eos/split_temps.log: time, clip, NSP/CPU max before the clip.
cd /data/local/tmp/s2st-qdc
export LD_LIBRARY_PATH=$PWD/lib ADSP_LIBRARY_PATH=$PWD/lib
H=./bin/qdc_s2st_bench_v2
LOG=eos/split_temps.log; : > $LOG

zmax () {  # $1 = type prefix pattern; prints max temp in C
  m=0
  for z in /sys/class/thermal/thermal_zone*; do
    case $(cat $z/type) in $1) t=$(cat $z/temp 2>/dev/null); [ -n "$t" ] && [ $t -gt $m ] && m=$t ;; esac
  done
  echo $((m / 1000))
}
cool () {
  while :; do
    n=$(zmax 'nsp*'); c=$(zmax 'cpu*')
    [ $n -lt 85 ] && [ $c -lt 90 ] && break
    echo "$(date +%T) waiting: nsp $n cpu $c" >> $LOG
    sleep 5
  done
}

for x in $(pidof llama-server); do tr '\0' ' ' < /proc/$x/cmdline | grep -q -- "--port 8083" && kill $x; done; sleep 3
./bin/llama-server -m model/asr-r6-Q4_K_M.gguf --mmproj model/mmproj-q8_0.gguf --host 127.0.0.1 --port 8083 \
  -ngl 99 -c 2048 -np 1 -dev HTP0 -mmdev GPUOpenCL --cache-ram 0 > asr_server_split.log 2>&1 < /dev/null &
SP=$!
i=0; until curl -s -m 2 127.0.0.1:8083/health | grep -q ok; do sleep 2; i=$((i+1)); [ $i -gt 60 ] && break; done

policy () { case $1 in zh) echo "--realtime --no-pause-trigger" ;;
                       *)  echo "--realtime --pause-ms 250 --rt-min-seg 2.5 --rt-max-seg 1000" ;; esac; }
rm -f eos/Gs.jsonl.tmp
for src in en vi zh ko; do for n in $(seq 0 19); do i=$(printf %02d $n)
  cool
  echo "$(date +%T) ${src}_$i nsp $(zmax 'nsp*') cpu $(zmax 'cpu*')" >> $LOG
  tgt=en; [ $src = en ] && tgt=vi
  v=voices/$([ $tgt = vi ] && echo vi_VN-vais1000-medium || echo en_US-lessac-medium)
  $H --audio data/audio/${src}_$i.wav --src $src --tgt $tgt --tts $v.onnx --tts-map $v.onnx.json \
     --espeak-data . --warmup 0 --mt-warm --out eos/p.jsonl $(policy $src) > /dev/null 2>&1
  sed "s/^{/{\"id\":\"${src}_$i\",/" eos/p.jsonl >> eos/Gs.jsonl.tmp
  sleep 3
done; echo "$src done $(date +%T)"; done
mv eos/Gs.jsonl.tmp eos/Gs.jsonl

kill $SP; sleep 3
(nohup ./bin/llama-server -m model/qwen3-asr-1.7b-4lang-r6-q4km.gguf --mmproj model/mmproj-q8_0.gguf \
  --host 127.0.0.1 --port 8083 -ngl 99 -c 2048 -np 1 -dev GPUOpenCL --cache-ram 0 > asr_server.log 2>&1 < /dev/null &)
echo "SPLIT DONE $(date +%T)"
