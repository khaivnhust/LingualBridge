#!/system/bin/sh
# eos_bench.sh -- end-of-speech -> first MT token, one variable at a time.
#   A  one-shot, model emits "language X<asr_text>" itself (eval-verified prompt)
#   B  one-shot, language prefilled from the known source
#   C  B + realtime replay with pause-segmented overlapped ASR
# Same 80 clips for every config; per-clip files are concatenated at the end
# because each invocation truncates --out.
cd /data/local/tmp/s2st-qdc
export LD_LIBRARY_PATH=$PWD/lib
H=./bin/qdc_s2st_bench_v2
OUT=${OUT:-eos}; rm -rf "$OUT"; mkdir -p "$OUT/parts"
voice () { case $1 in vi) echo vi_VN-vais1000-medium ;; en) echo en_US-lessac-medium ;; zh) echo zh_CN-huayan-medium ;; esac; }
tgt_of () { case $1 in en) echo vi ;; *) echo en ;; esac; }
run () {  # cfg src i extra...
  cfg=$1; src=$2; i=$3; shift 3; tgt=$(tgt_of $src); v=voices/$(voice $tgt)
  $H --audio data/audio/${src}_$i.wav --src $src --tgt $tgt --tts $v.onnx --tts-map $v.onnx.json \
     --espeak-data . --warmup 0 --mt-warm --out "$OUT/parts/$cfg.${src}_$i.jsonl" "$@" > /dev/null 2>&1
}
# two throwaway clips: the first Adreno inference pays shader compilation
run warm en 00; run warm en 01 --realtime
for src in en vi zh ko; do for n in $(seq 0 19); do i=$(printf %02d $n)
  run A $src $i --no-force-lang
  run B $src $i
  run C $src $i --realtime
done; echo "$src done $(date +%T)"; done
for cfg in A B C; do
  for f in $OUT/parts/$cfg.*.jsonl; do
    id=$(basename $f .jsonl); id=${id#*.}
    sed "s/^{/{\"id\":\"$id\",/" $f >> $OUT/$cfg.jsonl
  done
done
rm -rf $OUT/parts
echo "ALL DONE $(date +%T)"
