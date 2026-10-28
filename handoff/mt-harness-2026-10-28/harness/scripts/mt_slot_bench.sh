#!/system/bin/sh
# mt_slot_bench.sh <label> [extra args]  -- a two-person conversation: en->vi and
# vi->en turns alternate, so with one MT slot every turn evicts the other
# target's system prompt. Run once against -np 1, once against -np 2 + --mt-slot.
cd /data/local/tmp/s2st-qdc
export LD_LIBRARY_PATH=$PWD/lib
L=$1; SLOTS=$2; mkdir -p mtslot; rm -f mtslot/$L.jsonl
turn () {  # src tgt clip slot
  v=voices/$([ $2 = vi ] && echo vi_VN-vais1000-medium || echo en_US-lessac-medium)
  ex=""; [ "$SLOTS" = 2 ] && ex="--mt-slot $4"
  ./bin/qdc_s2st_bench_v2 --audio data/audio/$1_$3.wav --src $1 --tgt $2 --tts $v.onnx \
    --tts-map $v.onnx.json --espeak-data . --warmup 0 $ex --out mtslot/t.jsonl > /dev/null 2>&1
  cat mtslot/t.jsonl >> mtslot/$L.jsonl
}
for n in $(seq 0 14); do i=$(printf %02d $n); turn en vi $i 0; turn vi en $i 1; done
echo "$L done"
