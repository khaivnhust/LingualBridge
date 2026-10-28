#!/system/bin/sh
# eos_policy.sh -- Round 6 ASR on device, 80 clips:
#   B6  one-shot, language forced (accuracy reference)
#   G   per-language policy from WORKLOG §17: pause-cut segments for en/vi/ko
#       (>= 250 ms pauses, >= 2.5 s segments, no forced cuts); speculative
#       one-shot for zh, where pause cuts cost 5.5 CER.
cd /data/local/tmp/s2st-qdc
export LD_LIBRARY_PATH=$PWD/lib
H=./bin/qdc_s2st_bench_v2
policy () { case $1 in zh) echo "--realtime --no-pause-trigger" ;;
                       *)  echo "--realtime --pause-ms 250 --rt-min-seg 2.5 --rt-max-seg 1000" ;; esac; }
run () {  # cfg src i extra...
  cfg=$1; src=$2; i=$3; shift 3
  tgt=en; [ $src = en ] && tgt=vi
  v=voices/$([ $tgt = vi ] && echo vi_VN-vais1000-medium || echo en_US-lessac-medium)
  $H --audio data/audio/${src}_$i.wav --src $src --tgt $tgt --tts $v.onnx --tts-map $v.onnx.json \
     --espeak-data . --warmup 0 --mt-warm --out eos/p.jsonl "$@" > /dev/null 2>&1
  sed "s/^{/{\"id\":\"${src}_$i\",/" eos/p.jsonl >> eos/$cfg.jsonl.tmp
}
rm -f eos/B6.jsonl.tmp eos/G.jsonl.tmp
run warm en 00; run warm en 01 --realtime
for src in en vi zh ko; do for n in $(seq 0 19); do i=$(printf %02d $n)
  run B6 $src $i
  run G  $src $i $(policy $src)
done; echo "$src done $(date +%T)"; done
mv eos/B6.jsonl.tmp eos/B6.jsonl; mv eos/G.jsonl.tmp eos/G.jsonl; rm -f eos/warm.jsonl.tmp
echo "POLICY DONE $(date +%T)"
