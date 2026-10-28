# LingualBridge — Bối cảnh dự án (Project Context)

> File này tóm tắt toàn bộ context của dự án để đọc lại nhanh. Tổng hợp từ 4 tài liệu
> trong thư mục (cập nhật lần cuối: 27/10/2026 — thêm phần cứng: núm chọn ngôn ngữ + tản nhiệt,
> mục 4.1–4.2). Khi có quyết định mới, cập nhật mục
> **8. Trạng thái & quyết định mới nhất** trước tiên.

## 0. Các file trong thư mục

| File | Nội dung | Ngày |
|---|---|---|
| `LingualBridge_Team.pdf` | **Technical Proposal Vòng 2** (đã nộp, giúp đội vào Top 40) | 21/08/2026 |
| `1V_P3_Prototype_Brief.pdf` | **Brief Vòng 3** của BTC (song ngữ EN/VI): yêu cầu, lịch, thang điểm | 14/10/2026 |
| `LingualBridge — Phân công công việc Vòng 3.pdf` | **Kế hoạch nội bộ Vòng 3**: phân công, deadline, rủi ro | 16/10/2026 |
| `MT_DISTILLATION.md` | **Hướng dẫn kỹ thuật train model MT** (trim vocab + fine-tune Hy-MT2-1.8B), kèm số đo thật trên thiết bị | 16/10/2026 (mới nhất) |
| `TTS_VOICE_SELECTION.md` | **Báo cáo chọn giọng TTS**: 23 giọng đã thử, số đo RTF/RAM, test code-switching, gotchas | 17/10/2026 |
| `TTS_FINETUNE_PLAN.md` | **Kế hoạch fine-tune TTS**: teacher VoxCPM2 → data tổng hợp → fine-tune Piper medium 4 ngôn ngữ (chưa bắt đầu) | 17/10/2026 |
| `TTS_REPORT.md` | **Báo cáo TTS bằng tiếng Anh cho cả nhóm**: giọng đã chọn, số đo, phát hiện, prompt MT, tóm tắt kế hoạch fine-tune, việc cần làm theo người | 18/10/2026 |
| `amd-software-adrenalin-...exe` | Driver AMD — **không liên quan dự án**, bỏ qua | — |

Chưa có source code nào trong thư mục này (chưa phải git repo).

---

## 1. Dự án là gì

**Cuộc thi:** OneVoice AI Challenge 2026 (BTC liên quan VNG + Qualcomm). Đội **LingualBridge**
đang ở **Vòng 3 — Prototype Submission (Top 40)**.

**Sản phẩm:** Thiết bị/app **dịch giọng nói → giọng nói (speech-to-speech) chạy hoàn toàn
offline** trên chip **Snapdragon**, dùng NPU Hexagon / GPU Adreno / CPU.

- **Cặp ngôn ngữ:** VI↔EN, VI↔ZH, VI↔KO (4 ngôn ngữ: vi, en, zh, ko → 6 chiều dịch).
- **Người dùng mục tiêu:** công nhân Việt Nam làm việc với quản lý/kỹ sư Hàn, Trung, Anh
  trong nhà máy FDI, công trường, kho vận, phòng khám nhà máy.
- **Vấn đề giải quyết:** Google/Microsoft Translator cần internet (nhà máy hay mất mạng),
  độ trễ cao (>5 s), gửi dữ liệu lên cloud (vi phạm NDA), phải thao tác điện thoại liên tục.
  Hiểu sai chỉ dẫn an toàn → tai nạn, dừng dây chuyền.
- **Giá trị cốt lõi:** 100% offline, không dữ liệu nào rời thiết bị, rảnh tay (hands-free,
  tự phát hiện giọng nói), hiểu thuật ngữ chuyên ngành (glossary).

## 2. Thành viên

| Người | Vai trò Vòng 3 | Trách nhiệm chính |
|---|---|---|
| **Nguyễn Thế Khải** | Team Lead / AI | Distill model MT, dữ liệu, đánh giá chất lượng dịch, kịch bản demo 30 phút |
| **Nguyễn Văn Lâm** | ML Optimization + UX | Tối ưu on-device, **lõi C++ của pipeline**, công cụ benchmark, thiết kế UX |
| **Nguyễn Trung Kiên** | Hardware + Android | Thiết bị, app Android/UI (Kotlin), audio + TTS, đóng gói APK, video demo |

(Ghi chú: proposal Vòng 2 liệt kê "Nguyen Van Lam" hai lần — một lần ML/Optimization, một
lần UX/Product Designer — nên Lâm kiêm cả hai vai.)

---

## 3. Yêu cầu Vòng 3 (từ Brief của BTC)

### Mốc thời gian

| Mốc | Ngày |
|---|---|
| Kick-off Vòng 3 | 14/10/2026 |
| Workshop + Q&A Qualcomm (online) | **23/10/2026** |
| Kiểm thử thiết bị tại VNG Campus | chưa công bố |
| **Hạn nộp prototype** | **13/11/2026 23:59** (hạn nội bộ của đội: **12/11**) |
| Công bố kết quả | 23/11/2026 |
| Field-Testing (test trực tiếp) | 28/11/2026 |
| Chung kết tại VNG Campus, TP.HCM | 05/12/2026 |

### Bắt buộc
- Chạy trên **phần cứng thật** (máy Qualcomm của BTC hoặc tự chuẩn bị).
- **Toàn bộ inference on-device**, không gọi API/cloud lúc demo.
- Có **giao diện tối thiểu** (app/CLI/nút bấm) để giám khảo tự thao tác, không chạy code tay.
- Thử ít nhất **1 cặp ngôn ngữ** đã đăng ký; vào Field-Testing/Chung kết phải thêm cặp.
- **Kịch bản hội thoại ~30 phút** trong bối cảnh Vòng 2 (nhà máy).
- Môi trường test có tạp âm nhẹ (quạt, người đi lại).

### Dataset chấm
- Mỗi cặp (VI↔EN, VI↔ZH, VI↔KO) có **tập test ẩn 1.000 câu**. Có "Round 3 - Sample Dataset" đính kèm (chưa có trong thư mục).

### Thang điểm Vòng 3

| Tiêu chí | Tỷ trọng |
|---|---|
| A. Chất lượng dịch voice-to-voice (độ chính xác ngữ nghĩa, rõ, tự nhiên) | **45%** |
| B. Hiệu năng on-device (kích thước model, thời gian suy luận, RAM/VRAM đỉnh) | **20%** |
| C. Tích hợp thiết bị & UX (cho công nhân nhà máy/công trường/kho) | **20%** |
| D. Hồ sơ demo (video, tài liệu, khả năng tái lập, roadmap) | **15%** |
| Điểm cộng phần cứng tự thiết kế | tối đa **+0,5** |

### Hồ sơ nộp (Google Form + link Drive/GitHub)
1. **Video demo end-to-end** — BẮT BUỘC quay cảnh tắt mạng/bật chế độ máy bay ngay trước khi chạy; không cắt ghép; hiện rõ log thời gian xử lý.
2. **Gói cài đặt offline** (APK hoặc source + requirements/Dockerfile + script) để BGK tự chạy.
3. **Kết quả benchmark thực đo** khi ngắt mạng, kèm log gốc/ảnh chụp.
4. (Khuyến khích) Hướng dẫn tái lập.
5. Source code trên **GitHub public** + Prototype Submission Template đã điền.
- Tên file: `LingualBridge_Prototype Submission_V3` (.docx hoặc .pdf).
- Nếu BGK cài lỗi → **video là căn cứ chấm chính**.

### Hỗ trợ từ Qualcomm
- **Qualcomm AI Hub** (aihub.qualcomm.com): convert, quantize, compile, profile model.
- **Qualcomm Device Cloud — QDC** (qdc.qualcomm.com/devices): chạy từ xa trên máy thật.
- Nền tảng khuyến nghị: RB3 Gen 2, Arduino Ventuno Q, IQ-9075, điện thoại Snapdragon 8 Gen 2/3+, Snapdragon X Elite.

---

## 4. Kiến trúc hệ thống

```
Mic → (AEC/NS, beamform) → VAD → ASR → MT (LLM) → TTS → Loa
```

| Khâu | Lựa chọn hiện tại | Ghi chú |
|---|---|---|
| Thiết bị | **OnePlus 13** (Snapdragon 8 Elite, **lưng kính**), RAM ≥ 12 GB (ưu tiên 16 GB) | Proposal V2 ghi 8 Gen 3; Vòng 3 đổi sang 8 Elite. Pin 6000 mAh, IP68/IP69, cụm camera tròn lớn góc trên trái. Phần cứng gắn thêm: mục 4.1–4.2 |
| VAD | silero-vad (ONNX, ~2 MB) | Chưa tích hợp |
| ASR | **Qwen3-ASR-1.7B** Q4_K_M, Adreno/OpenCL | **Đã chạy ổn** trên 8 Elite qua QDC. Dùng language ID của ASR để tự chọn chiều dịch |
| MT | **Hy-MT2-1.8B** (Hunyuan-MT2, Tencent) Q6_K trên Hexagon HTP | Xem mục 5 |
| TTS | Piper (ONNX), **đã chọn giọng 17/10**: VI CakeByVPBank v3 sid 2 (Quang Huy), EN en_US-ryan-high, ZH zh_CN-huayan-medium, KO YJin1983 piper_ko_female_v1 | Tất cả dùng espeak. Giọng KO cần piper-tts Python ≥1.5 (binary C++ cũ crash vì âm vị nhiều ký tự). ⚠️ License: ryan-high CC BY-NC-SA, huayan không rõ, YJin1983 không ghi. Chi tiết: `TTS_VOICE_SELECTION.md` |
| Runtime | llama.cpp (có backend Hexagon HTP / OpenCL), ONNX Runtime, QNN | |
| App | Kotlin (UI) + lõi C++ qua JNI | Chưa bắt đầu |

**Tối ưu bộ nhớ dự kiến:** lazy-load theo cặp ngôn ngữ, mmap weights, reset KV-cache sau mỗi câu.

**UX dự kiến (wireframe L4):** chọn cặp ngôn ngữ, Start/Stop, hiện câu gốc + bản dịch chữ to,
hiển thị độ trễ, import glossary. Tắt mic khi đang phát TTS (tránh tự dịch lại tiếng của chính mình).

**Tính năng nâng cao đã hứa ở V2:** zero-tap/barge-in, glossary thuật ngữ do admin nạp
(chèn vào prompt MT), language/domain pack theo module, audit log mã hoá on-device (opt-in).

### 4.1 Núm xoay chọn cặp ngôn ngữ (thiết kế 27/10, chưa làm)

- **Núm chọn cặp, không chọn chiều:** công tắc xoay 3 nấc VI↔EN / VI↔ZH / VI↔KO. Chiều dịch trong cặp
  do language ID của ASR quyết định (việc K7). Dùng **công tắc xoay nhiều nấc** (vị trí tuyệt đối, tắt máy
  bật lại vẫn đúng), không dùng encoder EC11.
- **Kết nối:** ESP32-S3 (USB native) chạy **USB CDC** → **hub USB-C có PD pass-through** ← sạc dự phòng PD.
  Hub làm cho điện thoại vừa là host data vừa nhận sạc. **Không tách dây Y tự chế được**: chân CC quyết
  định vai trò, nhận điện thì điện thoại thành device và không nói chuyện được với ESP.
- **Không root:** đọc bằng Android USB Host API + thư viện `usb-serial-for-android`; intent filter
  `USB_DEVICE_ATTACHED` + `device_filter.xml` (VID Espressif 0x303A) để app tự mở khi cắm. Root bị loại vì
  APK phải chạy trên máy BGK không root; root cũng không giúp "đọc thẳng dây data" (USB là giao thức gói tin).
- **Giao thức text:** ESP gửi `POS n\n` khi đổi nấc (chống dội 50 ms) và nhắc lại mỗi 1 s (trạng thái
  tuyệt đối). Chiều ngược lại dùng cho LED trạng thái / tốc độ quạt (mục 4.2).
- **UX:** TTS đọc tên cặp khi đổi nấc; đang xử lý dở câu thì áp dụng cặp mới từ câu sau.
- **Cần kiểm chứng:** sạc + data qua hub trên OnePlus 13 (`adb shell dumpsys battery`); rút sạc dự phòng
  có thể làm ESP reset và kết nối lại → app phải tự mở lại cổng.

### 4.2 Tản nhiệt (chốt 27/10, chưa làm)

**Nguyên tắc:** tản thụ động + quạt, **giữ nguyên lưng kính**. Tản không bao giờ lạnh hơn môi trường nên
không đọng sương. Lớp quyết định hiệu quả là **tản lá → không khí** (~12–24°C chênh ở ~6 W); kính lưng chỉ
làm kém ~2–5°C, các lớp tiếp xúc khác mỗi lớp ~0,1–2°C.

```
Lưng kính ─[pad mềm 1 mm]─ Tấm đồng 1,2 mm ─[keo / PTM7950]─ Tản nhôm bàn chông ─[khe 2–5 mm]─ Quạt 40 mm (đệm cao su)
             (khung in 3D ép cả cụm xuống ở nhiều điểm; không khoan tấm đồng vì mặt dưới phải phẳng)
```

| Món | Thông số | Ghi chú |
|---|---|---|
| Pad (kính ↔ đồng) | Mềm (Shore 00 ~30–50), **1 mm** (thử thêm 0,5 mm), ~3–6 W/mK tầm trung | Pad rẻ chỉ kém ~1–2°C; độ mềm quan trọng hơn hệ số dẫn nhiệt. Lau sạch kính trước khi dán |
| Tấm đồng | **1,2 mm**, phủ gần hết lưng, khoét lỗ tròn cho cụm camera | Trải nhiệt ≈ nhôm 2–3 mm, ~77 g. Dễ cong khi khoét → soi đèn kiểm tra, ép phẳng lại |
| Đồng ↔ tản lá | **Keo tản nhiệt không dẫn điện hoặc PTM7950**, lớp mỏng (chưa chốt loại) | Không dùng pad dày. PTM cần làm nóng cụm tản một lần ~50–55°C |
| Tản lá | **Nhôm bàn chông (pin-fin) 40×40 mm**, cột cao 10–15 mm, khoảng cách cột ≥ 2 mm, ưu tiên cột tròn | Chỉ cần to bằng quạt. Hợp quạt thổi từ trên xuống (gió thoát 4 phía) |
| Quạt | **40 mm**, 5 V, ~24 dB(A) @1 m | Sát mic ~40–50 dB(A), thấp hơn giọng nói ~20 dB. Đệm cao su chống rung |
| Khung in 3D | Ôm máy, ép cụm tản; chứa hub, ESP32-S3, núm xoay | Phần cứng tự thiết kế → điểm cộng +0,5 (N10) |

- **Vị trí:** đo điểm nóng bằng súng nhiệt hồng ngoại khi chạy pipeline, đặt tâm tản lá ngay trên đó
  (dự kiến nửa trên, cạnh/dưới cụm camera), càng xa mic cạnh trên càng tốt.
- **Luồng gió:** khung **mở hai bên hông**; **chặn phía cạnh trên** (mic lọc ồn) và **phía tay cầm**.
- **Điều khiển quạt (chưa chốt):** chạy liên tục, hoặc ESP32-S3 điều tốc PWM qua MOSFET theo
  `PowerManager.currentThermalStatus` / `getThermalHeadroom()` mà app gửi qua USB CDC. Chỉ cần điều tốc
  nếu test ASR thấy quạt gây ảnh hưởng.
- **Giảm nhiệt từ gốc:** thử **sạc bypass** trong Games của OxygenOS (chưa rõ có chạy với PD qua hub không;
  không có thì sạc công suất thấp); dồn model lên NPU (HTP); giảm độ sáng màn hình; so có/không
  High performance mode, chọn chế độ cho ms/token ổn định nhất sau 30 phút.
- **Cần kiểm chứng:** (1) tìm điểm nóng; (2) chạy 30 phút **không tản** vs **có tản**, ghi `thermalStatus`,
  nhiệt độ pin, ms/token → lưu log + biểu đồ cho hồ sơ; (3) ASR với quạt tắt vs chạy hết tốc độ
  (30–50 câu phát từ loa cách 40 cm, có tiếng ồn nền), chênh WER/CER < 1–2 điểm là đạt.
- **Đã loại:** sò lạnh/Peltier (đọng sương, ~10 W, phải điều khiển theo điểm sương); tháo lưng kính (chỉ lợi
  2–5°C, mất IP68/69, rủi ro pin/kính); tấm đồng 2 mm (nặng ~129 g, chỉ mát hơn ~0,3–1°C); miếng dán
  graphene/đồng phủ nano-carbon (quá mỏng, trải nhiệt kém tấm đồng 5–25 lần); tản lá dọc (kém bàn chông
  khi quạt thổi từ trên xuống); pad dày giữa đồng và tản lá.

---

## 5. Model MT — phần quan trọng nhất (rủi ro lớn nhất)

Kế hoạch đã tiến hoá qua các tài liệu:
- **Proposal V2:** Hunyuan-MT-7B INT4 (~4 GB).
- **Kế hoạch phân công (16/10):** distill teacher Hunyuan-MT-7B → so sánh 3 student
  (Hunyuan-MT-1.8B, Qwen 3B Instruct, NLLB-200 1.3B), chọn model thắng ngày 02/11.
- **`MT_DISTILLATION.md` (mới nhất):** **base đã chốt = Hy-MT2-1.8B** (đo trên máy thật).
  Teacher khuyến nghị = **Hy-MT2-30B-A3B** (MoE, ~3B active). Việc còn lại: **trim vocab + fine-tune phục hồi**.

### Số đo thật trên Snapdragon 8 Elite (serial, từng request)

| Khâu | Backend | Kết quả |
|---|---|---|
| ASR Qwen3-ASR-1.7B Q4_K_M | Adreno/OpenCL | RTF 0.365 (FLEURS), 0.594 (Kspon hội thoại) |
| MT Hy-MT2-7B Q6_K | Adreno/OpenCL | 209.7 ms/token |
| MT Hy-MT2-7B Q6_K | Hexagon HTP | 130.3 ms/token, median 7.72 s |
| **MT Hy-MT2-1.8B Q6_K** | **Hexagon HTP** | **34.7 ms/token, median 1.57 s** |

Chất lượng Hy-MT2-1.8B trên **267 câu FLEURS song song** (40 câu/chiều) — **baseline cần vượt:
BLEU trung bình 33.71, chrF 44.32**:

| Chiều | BLEU | chrF | Chiều | BLEU | chrF |
|---|---|---|---|---|---|
| en→ko | 39.61 | 27.56 | ko→en | 26.41 | 51.03 |
| en→vi | 36.41 | 51.71 | vi→en | 27.94 | 52.73 |
| en→zh | 42.18 | 29.48 | zh→en | 29.70 | 53.39 |

(chrF thấp ở ko/zh là bình thường do không có khoảng trắng — **so sánh giữa các ngôn ngữ bằng BLEU**.)

Ngân sách pipeline với 1.8B cho câu 10 s: ASR 4.7 s + MT 1.6 s + TTS 1.0 s ≈ **7.3 s, RTF ~0.73** (< 1.0).
**ASR giờ là khâu tốn thời gian nhất.** Latency MT trên Flores 48–55 ms/token, mục tiêu ≤ 50.

### Quy trình train MT (tóm tắt `MT_DISTILLATION.md`)

1. **Vocab trim (BẮT BUỘC):** 120,818 token → ~32k–48k, chỉ giữ token en/vi/zh/ko + special/template/punct.
   Mục đích là **đảm bảo cấu trúc** model không thể sinh token ngôn ngữ khác (chỉ tiết kiệm ~10% dung lượng).
   **Đo token usage trên corpus thật, không đoán.** Trim xong **bắt buộc fine-tune** phục hồi (embedding tied).
   Trim không chặn lẫn giữa 4 ngôn ngữ giữ lại → nếu cần thì thêm logit masking theo chiều dịch.
2. **Knowledge drop (tuỳ chọn):** bỏ các mode instruction (terminology/style/structured) và 29 ngôn ngữ thừa.
3. **Teacher:** Hy-MT2-30B-A3B Q4_K_M (18.2 GB) qua `llama-server`, **temperature 0.0** khi sinh nhãn.
   Đối thủ duy nhất: Index-Translate-35B-A3B → so trên 267 câu để quyết. Adjudicator: ~200 câu đối chiếu Google/DeepL.
4. **Prompt (không có system prompt), dùng y hệt khi train và khi deploy:**
   ```
   Translate the following text into {target_lang}. Note that you should only output the translated result without any additional explanation:

   {source_text}
   ```
   `target_lang` = `English` / `Vietnamese` / `Chinese` / `Korean`.
5. **Data:** OPUS-100 (en-vi, en-zh, en-ko; kiểm tra zh là giản thể hay lẫn) + synthetic **cả 6 chiều**
   (không chỉ pivot qua English), synthetic ≤ ~50%. **Không bao giờ train trên tập eval FLEURS**
   (`data/mt/fleurs_parallel.jsonl`). Kế hoạch K3 còn có corpus theo domain (nhà máy, an toàn lao động, y tế),
   **không dùng câu trong dataset mẫu BTC**.
6. **Train:** cloud GPU **A100 40 GB+** (GPU local 11.9 GB không đủ), **full-parameter, KHÔNG LoRA**, bf16,
   AdamW, seq len 512, loss mask chỉ trên target, cosine + 3% warmup, effective batch 256–512.
7. **Quantize:** **ship Q6_K**. HTP hỗ trợ Q4_0, Q4_1, Q8_0, Q5_K, Q6_K, MXFP4. **Q4_K KHÔNG được hỗ trợ —
   tự động rơi về CPU mà không báo.** RAM host trên HTP gần như không phụ thuộc quant (model 6.16 GB chỉ ~600 MB RSS).
8. **Eval:** BLEU/chrF **từng chiều** (không lấy trung bình che lỗi), gap với teacher, end-to-end ASR→MT trên audio thật,
   ms/token median & p95 trên 50+ câu, RTF < 1.0, **kiểm tra không sinh được token ngoài 4 ngôn ngữ**.
   Bước đo device phải chạy trên máy thật.

### Gotchas thiết bị đáng nhớ
- Log MT không có dòng `HTP` → không tìm thấy skel; `ADSP_LIBRARY_PATH` phải trỏ vào `lib/`.
- Thiết bị **không có Python** (ngoài chroot), **không có curl/wget**, chỉ `nc`.
- DNS: rootfs Debian dùng `1.1.1.1` có thể không route được → dùng `8.8.8.8` (`scripts/setup.sh` đã xử lý).
- Tải model trực tiếp trên máy ~17 MB/s, nhanh hơn push qua adb (~7 MB/s).
- Chấm điểm BLEU/chrF++ on-device bằng implementation stdlib; self-test phải ra đúng 100.00.
- Cloud instance có thể bị thu hồi bất cứ lúc nào.

---

## 6. Kế hoạch công việc Vòng 3 (tất cả đang "Chưa bắt đầu" tại 16/10)

**Mục tiêu bắt buộc:** app Android chạy end-to-end **VI↔EN** offline trên 8 Elite trước 12/11.
Nếu kịp thì thêm VI↔ZH, VI↔KO; nếu không, phải xong trước Field-Testing 28/11.
**Nguyên tắc: luôn giữ một bản VI↔EN chạy được — 1 cặp ổn định tốt hơn 3 cặp lỗi.**

### Khải (chất lượng dịch — 45% điểm)
| # | Việc | Hạn |
|---|---|---|
| K1 | Tải dataset mẫu BTC, tách dev nội bộ, script BLEU/chrF/COMET cho 6 chiều | 18/10 |
| K2 | Baseline zero-shot teacher + student, 6 chiều → gửi Lâm | 21/10 |
| K3 | Corpus nguồn theo domain, làm sạch, bỏ trùng | 23/10 |
| K4 | Sinh bản dịch bằng teacher (sequence-level KD), lọc lỗi | 26/10 |
| K5 | Distill student, có checkpoint + điểm dev | 31/10 |
| K6 | **Chốt model MT thắng** (cùng Lâm, theo điểm sau quant + số đo device) | **02/11** |
| K7 | Dùng language ID của Qwen3-ASR tự chọn chiều dịch | 30/10 |
| K8 | Glossary thuật ngữ nhà máy 4 ngôn ngữ, chèn vào prompt MT | 04/11 |
| K9 | Kịch bản hội thoại 30 phút, tập dượt | 06/11 |
| K10 | Viết phần AI/roadmap/giới hạn trong Submission Template | 11/11 |

### Lâm (hiệu năng — 20% điểm + bằng chứng benchmark)
| # | Việc | Hạn |
|---|---|---|
| L1 | Tài khoản AI Hub + QDC cho cả nhóm; lưu log test Qwen3-ASR làm bằng chứng | 19/10 |
| L2 | Quantize nhanh + profile student trên 8 Elite (ms/token, RAM đỉnh, dung lượng), chọn runtime | 23/10 |
| L3 | **Lõi C++ pipeline**: VAD → Qwen3-ASR streaming → MT → TTS, hàng đợi + luồng, JNI | **30/10** |
| L4 | Wireframe UX → giao Kiên | 23/10 |
| L5 | Quantize checkpoint distill, tối ưu bản thắng, đo độ tụt COMET | 05/11 |
| L6 | Quản lý bộ nhớ (lazy-load, mmap, reset KV-cache); RAM ổn định 30 phút, không bị Android kill | 06/11 |
| L7 | Công cụ benchmark chế độ máy bay: input file câu/audio → CSV, log thời gian từng khâu + RAM đỉnh | 06/11 |
| L8 | Chạy benchmark chính thức, lưu log gốc + screenshot (mục 7.3 brief) | 10/11 |
| L9 | Viết phần hiệu năng + hướng dẫn tái lập trong Template | 11/11 |

**Mốc quan trọng của Lâm: 30/10** — VI↔EN chạy end-to-end với student zero-shot, để sau đó chỉ cần thay file weights.

### Kiên (thiết bị, app, hồ sơ)
| # | Việc | Hạn |
|---|---|---|
| N1 | **Mua/mượn điện thoại 8 Elite**, RAM ≥ 12 GB, bật dev mode + adb | **22/10** |
| N2 | Khung app Kotlin theo wireframe | 28/10 |
| N3 | Audio: AudioRecord 16 kHz, nguồn VOICE_COMMUNICATION + AEC/NS, tắt mic khi phát TTS | 28/10 |
| N4 | TTS: Piper VI/EN/ZH + tìm TTS tiếng Hàn on-device | 30/10 |
| N5 | Ghép UI với lõi C++ qua JNI, hiện độ trễ từng câu | 03/11 |
| N6 | Test tạp âm nhẹ; đo pin, nhiệt, throttling khi chạy 30 phút | 07/11 |
| N7 | Đóng gói APK + model, hướng dẫn cài offline; repo GitHub public, model trên Drive/HF | 10/11 |
| N8 | Quay video demo (bật máy bay trước, chạy trọn kịch bản, hiện log, không cắt) | 11/11 |
| N9 | Nộp Google Form | 12/11 |
| N10 | (Tuỳ chọn) vỏ máy / mic rời lấy điểm cộng +0,5 — thiết kế đã có ở mục 4.1–4.2 (khung in 3D + núm xoay + tản nhiệt) | 27/11 |

### Điểm bàn giao
K2→L2 (21/10) · L4→N2 (23/10) · L3→N5 (30/10) · K6→L5 (02/11) · K9→N8 (06/11) · L8, K10, L9→N9 (11/11)

### Rủi ro & dự phòng
| Rủi ro | Dự phòng |
|---|---|
| Distill chưa xong / kém hơn zero-shot | Dùng zero-shot tốt nhất, ưu tiên Hunyuan-MT-1.8B |
| Student không chạy được trên NPU (dễ nhất là NLLB) | Loại sớm, dồn GPU cho model khác |
| Chưa có máy 8 Elite đến 22/10 | Dev trên QDC; đăng ký kiểm thử tại VNG |
| Quantize làm tụt chất lượng | INT8 hoặc mixed precision cho layer nhạy |
| Không có TTS tiếng Hàn | Nộp với VI↔EN, bổ sung Hàn trước 28/11 |
| Máy tự nghe lại tiếng loa | Tắt mic khi TTS phát |
| BGK cài gói lỗi | Video phải hiện rõ log độ trễ |
| Lâm quá tải | Kiên nhận L7, Khải nhận L9 |

**License:** NLLB là CC-BY-NC (cấm thương mại), Qwen 3B có license riêng → kiểm tra trước khi public repo.

### Câu hỏi cho BTC tại workshop 23/10 (Khải hỏi)
- Test ẩn 1.000 câu là audio hay text? Chấm bằng metric nào?
- BGK chạy benchmark qua app hay cần chế độ CLI chạy theo file?
- Ngày kiểm thử tại VNG? Ở đó có máy 8 Elite không?
- Field-Testing yêu cầu thêm bao nhiêu cặp ngôn ngữ?
- Model distill có phải public weights? Có được dùng model license phi thương mại (NLLB)?
- Video 30 phút phải quay một mạch hay được chia đoạn?

---

## 7. Mục tiêu đã cam kết ở Proposal V2 (BGK có thể đối chiếu)

- Độ trễ end-to-end **< 2 s** (đo từ End-of-Speech đến Start-of-Synthesis; dự kiến median ~1.05 s, p90 ~1.75 s nhờ ASR streaming).
- RTF ~0.55–0.65 median, < 0.9 p90 (gate < 1.0).
- BLEU ≥ 42 domain / ≥ 35 general (VI↔EN); COMET ≥ 0.82 domain; TTS MOS ≥ 4.0; ASR WER ≤ 12% sạch / ≤ 20% ồn 75 dBA.
- Pin > 8 h (22.2 Wh / 2.4 W trung bình ≈ 9.2 h, duty cycle 15%); không cần màn hình.
- Kinh doanh: bán thiết bị US$399–499 (BOM US$200–230), license US$40–80/thiết bị/năm, cho thuê, fine-tune theo yêu cầu; go-to-market qua pilot với 1 nhà máy.

---

## 8. Trạng thái & quyết định mới nhất (cập nhật mục này khi có thay đổi)

- **Đã chốt:** thiết bị Snapdragon 8 Elite; ASR Qwen3-ASR-1.7B (đã kiểm chứng); **MT base Hy-MT2-1.8B Q6_K trên HTP**;
  teacher đề xuất Hy-MT2-30B-A3B; ship Q6_K; train full-parameter trên cloud A100.
- **TTS (17/10):** chọn 4 giọng Piper ở mục 4 sau khi nghe so sánh (`translator/compare_tts_voices.py`,
  trang nghe `translator/piper_outputs/compare*/index.html`). `pipeline.py` và `test_piper_tts.py` đã chuyển sang
  thư viện Python piper-tts. Test code-switching: espeak đọc thuật ngữ tiếng Anh bằng âm vị tiếng Anh (phải nghe kiểm tra),
  espeak-vi đánh vần sai "jig". Hướng MT: prompt giữ nguyên thuật ngữ + viết ký hiệu thành chữ → khi đó **không làm knowledge drop**.
- **Việc tiếp theo cho MT (theo `MT_DISTILLATION.md` §11):** tải OPUS-100 → đo token usage → sinh nhãn teacher
  (temp 0.0, 6 chiều) → trim vocab → full fine-tune → (tuỳ chọn) knowledge drop → quant Q6_K, verify tensor types
  & illegal tokens → eval từng chiều → đo ms/token, RTF trên máy. Nếu RTF ≥ 1.0 → dùng model nhỏ hơn / output ngắn hơn,
  **không** hạ quant.
- **Phần cứng (27/10):** thiết bị là **OnePlus 13 lưng kính**. Núm xoay 3 nấc chọn cặp ngôn ngữ qua
  ESP32-S3 USB CDC + hub USB-C PD, không root (mục 4.1). Tản nhiệt: pad 1 mm → đồng 1,2 mm → keo/PTM →
  tản nhôm bàn chông 40×40 + quạt 40 mm, giữ lưng kính, **bỏ sò lạnh** (mục 4.2). Chưa chốt: keo hay PTM7950,
  quạt chạy liên tục hay điều tốc. Chưa mua/lắp, chưa đo.
- **OxygenOS tắt app nền rất mạnh** (liên quan L6): chạy pipeline trong foreground service, tắt tối ưu pin
  cho app, khoá app trong đa nhiệm.

### Điểm chưa nhất quán cần lưu ý
1. **Model MT:** kế hoạch phân công còn ghi "teacher Hunyuan-MT-7B, so 3 student, chốt 02/11", trong khi
   `MT_DISTILLATION.md` đã chốt Hy-MT2-1.8B + teacher 30B-A3B. Tài liệu phân công cần cập nhật theo.
2. **Độ trễ:** số đo serial hiện tại ~7.3 s cho câu 10 s (ASR 4.7 s) — RTF đạt (< 1.0), nhưng muốn đạt
   **turnaround < 2 s** đã hứa ở V2 thì ASR phải chạy **streaming theo chunk** (việc L3) để phần lớn audio
   được nhận dạng trước khi người nói dừng.
3. **Tốc độ ASR:** kế hoạch ghi "27x", còn số đo là RTF 0.365 (≈ 2.7x realtime) — có thể là lỗi đánh máy; kiểm tra lại log trước khi đưa vào hồ sơ.
4. **Tập eval:** `MT_DISTILLATION.md` dùng 267 câu FLEURS; kế hoạch K1 dùng dataset mẫu của BTC. Nên báo cáo cả hai, vì BGK chấm trên bộ của BTC.
5. **Thiết bị:** proposal V2 ghi Snapdragon 8 Gen 3; Vòng 3 dùng 8 Elite (OnePlus 13) — cần giải thích thay đổi này trong Submission Template.
6. **Pin & sạc:** proposal V2 hứa pin > 8 h, không cần màn hình; prototype hiện là điện thoại + sạc dự phòng qua hub PD
   + quạt — cần nói rõ đây là prototype, chưa phải thiết bị cuối.
