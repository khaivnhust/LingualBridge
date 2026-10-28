# Huấn luyện MT: học sinh 1.8B đã cắt từ vốn cho thiết bị

Hướng dẫn độc lập để tạo ra một mô hình dịch đa ngôn ngữ cỡ nhỏ, chạy được
trên thiết bị Android dùng chip Snapdragon (GPU Adreno / NPU Hexagon), phục vụ cho
pipeline dịch giọng nói sang giọng nói.

Tài liệu này viết để người đọc có thể làm theo mà không cần kiến thức trước về
dự án này.

**Trạng thái: mô hình nền đã được chọn và đo trực tiếp trên thiết bị đích. Phần
còn lại là cắt từ vốn và một lượt huấn luyện để khôi phục sau khi cắt.**

---

## 0. Bạn đang xây dựng gì

Lấy `Hy-MT2-1.8B` (đo trên thiết bị đích ở **34,7 ms/token** trên NPU Hexagon),
**cắt từ vốn** của nó xuống chỉ còn en/vi/zh/ko để mô hình **về mặt cấu trúc không
thể phát ra token thuộc ngôn ngữ ngoài**, rồi **tinh chỉnh để khôi phục** sau khi
cắt. Tùy chọn: loại bỏ thêm kiến thức không cần dùng.

Có hai thành phần, trong đó chỉ một là bắt buộc:

| thành phần | trạng thái | lý do |
|---|---|---|
| **cắt từ vốn** | **BẮT BUỘC** | bảo đảm mang tính cấu trúc: đầu ra không thể phát token thuộc ~29 ngôn ngữ không dùng. Không phải cải thiện chất lượng - đây là ràng buộc tuyệt đối. |
| **loại kiến thức** | nên làm | loại bỏ các chế độ chỉ dẫn (thuật ngữ, văn phong, dữ liệu có cấu trúc) và các hành vi không thuộc ngôn ngữ đích. Cải thiện độ khớp; không bắt buộc về tính đúng đắn |

Việc cắt từ vốn **không thể thực hiện mà không cần lượt huấn luyện về sau** - bỏ
token khỏi embedding đã liên kết làm thay đổi phân phối đầu ra, và mô hình phải
được thích ứng lại. Vì vậy một lượt tinh chỉnh luôn diễn ra. Điều tùy chọn chỉ là
mức độ tinh chỉnh ngoài mức tối thiểu cần thiết.

### Kết quả đo trên thiết bị đích

Tất cả đều chạy nối tiếp, từng yêu cầu một, trên thiết bị tham chiếu Snapdragon 8 Elite:

| thành phần | backend | kết quả |
|---|---|---|
| ASR Qwen3-ASR-1.7B Q4_K_M | Adreno / OpenCL | RTF **0,365** (FLEURS), **0,594** (Kspon hội thoại) |
| MT Hy-MT2-7B Q6_K | Adreno / OpenCL | 209,7 ms/token |
| MT Hy-MT2-7B Q6_K | Hexagon HTP | **130,3** ms/token, trung vị 7,72 s |
| MT Hy-MT2-1.8B Q6_K | Hexagon HTP | **34,7** ms/token, trung vị 1,57 s |

Chất lượng MT-1.8B trên 267 câu song ngữ FLEURS, mỗi chiều 40 câu:

| chiều | BLEU | chrF | | chiều | BLEU | chrF |
|---|---|---|---|---|---|---|
| en->ko | 39,61 | 27,56 | | ko->en | 26,41 | 51,03 |
| en->vi | 36,41 | 51,71 | | vi->en | 27,94 | 52,73 |
| en->zh | 42,18 | 29,48 | | zh->en | 29,70 | 53,39 |
| | | | | **trung bình** | **33,71** | **44,32** |

Ngân sách pipeline với 1.8B: ASR 4,7 s + MT 1,6 s + TTS 1,0 s = **~7,3 s cho đoạn
10 giây, RTF ~0,73**. Dưới ngưỡng và còn dư địa. Lúc này ASR đã là thành phần tốn
thời gian nhất.

Độ trễ theo chiều trên Flores là 48-55 ms/token, so với mục tiêu <= 50.

> **Lưu ý về chrF giữa các hệ chữ viết.** Tiếng Hàn và tiếng Trung không có
> ranh giới từ bằng khoảng trắng, nên chồng lấp n-gram ký tự phạt nặng sự đổi cách
> diễn đạt - hai cách diễn đạt hợp lệ có thể gần như điểm 0. **BLEU mới là chỉ số
> so sánh được giữa bốn ngôn ngữ của chúng ta; chrF chỉ có ý nghĩa bên trong cùng
> một hệ chữ viết.** Đó là lý do en->ko có chrF 27,56 trong khi en->vi là 51,71 với
> BLEU tương đương.

## 1. Yêu cầu trước khi bắt đầu

- GPU trên cloud, **A100 40 GB trở lên**. Bước sinh dữ liệu từ giáo viên là nặng
  nhất; xem mục 2.
- Ổ đĩa từ 200 GB, RAM từ 64 GB
- Python 3.10+ với PyTorch, `transformers`, `datasets`, `accelerate`, `peft`
- `huggingface_hub`
- `llama.cpp` đã được dựng và đang chạy, để phục vụ giáo viên

## 2. Vì sao không dùng GPU cục bộ

Tinh chỉnh toàn bộ mô hình 3B với Adam cần riêng trọng số và gradient đã là
**12,0 GB**. Card 11,9 GB không đủ, và đưa bộ tối ưu hóa xuống CPU càng làm tệ
hơn: nó chuyển bộ nhớ từ tài nguyên khan hiếm sang CPU, và mỗi bước lại tốn
0,3-1,0 giây truyền PCIe trong khi GPU nằm không. Với 40 GB bạn không cần offload.
**Nếu thấy mình phải dùng, phiên bản của bạn quá nhỏ** - hãy kiểm tra thời gian
mỗi bước thay vì im lặng chịu đựng.

## 3. Chọn mô hình giáo viên

Học sinh là `Hy-MT2-1.8B`. Giáo viên là một quyết định riêng, và với card 48-96 GB
thì kích thước giáo viên không còn là ràng buộc.

### Đã kiểm tra có sẵn

| mô hình | kho | kích thước | FLORES XCOMET | % của Gemini 3.1 Pro |
|---|---|---|---|---|
| **Hy-MT2-30B-A3B** | `tencent/Hy-MT2-30B-A3B-GGUF` | **18,2 GB** Q4_K_M / 32,0 GB Q8_0 | **87,47** | **98,6%** |
| Hy-MT2-7B | `unsloth/Hy-MT2-7B-GGUF` | 4,62 GB Q4_K_M / 6,16 GB Q6_K | 86,89 | 97,9% |
| Hy-MT2-1.8B | `unsloth/Hy-MT2-1.8B-GGUF` | 1,47 GB Q6_K | 79,77 | 89,9% |
| Index-Translate-35B-A3B | `IndexTeam/Index-Translate-35B-A3B-preview-GGUF` | 21,7 GB Q4_K_M / 29,2 GB Q6_K | - | - |
| Index-Translate-9B | `IndexTeam/Index-Translate-9B-GGUF` | 5,78 GB Q4_K_M | - | - |
| Index-Translate-2B | `IndexTeam/Index-Translate-2B-GGUF` | - | - | - |
| MADLAD-400 3B MT | `cstr/madlad400-3b-mt-GGUF` | - | - | - |

### Khuyến nghị: Hy-MT2-30B-A3B

Cùng họ với mô hình nền của học sinh. Điều này quan trọng hơn chênh lệch chất
lượng thuần túy:

1. **Không truyền artefact xuyên họ.** Nghiên cên về tinh chỉnh ngôn ngữ tài
   nguyên thấp ghi nhận một lỗi thái "dừng đột ngột" ở NLLB-200-3.3B được
   *"mô hình học sinh thừa hưởng trung thực"*, và khuyến nghị *"chọn giáo viên cùng
   họ decoder-only"*. Tinh chỉnh trong cùng họ tránh hẳn vấn đề này.
2. **Cùng định dạng prompt** - đã kiểm chứng trên thiết bị, không phải thiết kế lại.
3. **98,6% của Gemini 3.1 Pro**, đúng trên các ngôn ngữ chúng ta triển khai.
4. **Vừa sức thoải mái** - 18,2 GB trên card 48 GB, còn dư chỗ để batching.

**Lưu ý về MoE:** bản 30B-A3B chỉ khoảng 3B tham số *hoạt động* dù tổng là 30B.
Tốc độ suy luận của giáo viên gần với 3B, không phải 30B. Sinh toàn bộ tập dữ liệu
tổng hợp rẻ hơn nhiều so với con số tham số gợi ý.

### Phương án thay thế đáng cân nhắc

Index-Translate-35B-A3B điểm cao hơn trên một số bảng xếp hạng, và cao hơn rất
nhiều trên các cặp ngôn ngữ thiểu số:

| mô hình | FLORES minor-pair | lệch mục tiêu |
|---|---|---|
| Index-Translate-35B-A3B | 0,8168 | 2,4% |
| Index-Translate-2B | 0,7377 | 4,2% |
| Hy-MT2-7B | 0,4626 | 35,7% |
| Hy-MT2-1.8B | 0,3163 | 54,2% |

Tỉ lệ lệch mục tiêu cao của Hy-MT2 trông đáng lo ngại, nhưng **"minor pair" gần
chắc không bao gồm vi/ko** - chúng có nguồn lực tốt, và 1.8B đã được đo là sinh ra
tiếng Hàn sạch với 0 rò rỉ tiếng Anh. Đừng hành động theo bảng này khi chưa kiểm
chứng.

**Giải quyết bằng thực nghiệm:** chạy cả hai giáo viên trên 267 câu song ngữ và so
với bản tham chiếu của chúng ta. Việc này mất một giờ và loại bỏ phỏng đoán. Nếu
Index-Translate thắng áp đảo, lưu ý rằng điều đó cũng thay đổi mô hình nền của
học sinh, và prompt phải được kiểm chứng lại.

### Bảng phân xử

Không dùng để sinh dữ liệu hàng loạt - dùng để **đối chiếu khoảng 200 câu với một
dịch vụ thương mại** (Google Translate, DeepL, Microsoft). Nếu giáo viên lệch với
dịch vụ tham chiếu ở 5% câu thì bạn biết nên tin ai. Rẻ và không tăng theo cỡ tập
dữ liệu.

MADLAD-400 3B là **ý kiến thứ hai** hữu ích khi một kết quả của giáo viên trông
đáng ngờ.

### Phục vụ giáo viên và prompt

```bash
./llama-server -m Hy-MT2-30B-A3B-Q4_K_M.gguf --host 127.0.0.1 --port 8084 -ngl 99 -c 2048 -np 1
```

Chat template không mang token vai trò; `<|extra_N|>` chỉ là dấu phân cách lượt,
không phải mã ngôn ngữ. **Không có system prompt:**

```
Translate the following text into {target_lang}. Note that you should only output the translated result without any additional explanation:

{source_text}
```

`target_lang` là **tên tiếng Anh đầy đủ**: `English`, `Vietnamese`, `Chinese`,
`Korean`.

**Khi sinh nhãn huấn luyện, dùng `temperature 0.0`.** Mức 0.7 trong model card dành
cho sinh văn bản tương tác; nhãn lấy mẫu ở 0.7 sẽ đưa nhiễu lấy mẫu vào từng ví
dụ. Giữ 0.7 chỉ khi đo độ trễ của đúng cấu hình sẽ triển khai.

## 4. Cắt từ vốn (BẮT BUỘC)

### Vì sao

Yêu cầu mang tính cấu trúc. Sau khi cắt, mô hình **về mặt vật lý không thể phát ra
token thuộc bất kỳ ngôn ngữ nào ngoài bốn ngôn ngữ của chúng ta**. Đó là một bảo
đảm, không phải xu hướng thống kê - nó đúng bất kể mô hình được yêu cầu làm gì.

Hy-MT2-1.8B có vốn 120.818 token, embedding liên kết, hidden 2048:

```
thân mô hình : 32 lớp x 54,5M = 1,74B tham số  (88%)
từ vốn       : 120.818 x 2.048  = 0,247B tham số  (12%)
```

Ở Q6_K, embedding liên kết chiếm khoảng 200 MB trong file 1,47 GB, nên **cắt chỉ
tiết kiệm khoảng 10% kích thước.** Giá trị nằm ở bảo đảm tính đúng đắn, không
phải ở số byte. Cắt bớt chiều sâu mới là đòn bẩy giảm kích thước nếu sau này cần.

### Chọn kích thước - đo, đừng đoán

**Cắt trước khi huấn luyện**, và xác định kích thước từ **mức sử dụng token thực
tế trên tập huấn luyện**, không phải từ phỏng đoán. Công trình ASR đã xác lập
quy tắc này một cách tốn kém.

Cần giữ:

- mọi subword token xuất hiện trong bất kỳ ngôn ngữ nào en/vi/zh/ko **trong tập
  huấn luyện**, ở cả phía nguồn và phía đích
- mẫu chỉ dẫn và token đặc biệt
- dấu câu, chữ số, ký hiệu khoảng trắng còn lại sau chuẩn hóa

Ước tính 32k-48k là điểm khởi đầu hợp lý; hãy đo trước khi cam kết.

### Điều cắt từ vốn không bảo đảm

Sau khi cắt, mô hình vẫn giữ **cả bốn ngôn ngữ của chúng ta**. Được yêu cầu dịch
tiếng Anh sang tiếng Hàn, nó vẫn có thể phát ra tiếng Việt. Cắt bỏ token *nước
ngoài*, không cắt nhầm lẫn giữa các ngôn ngữ được giữ lại.

Để có bảo đảm tuyệt đối ở đó, thêm một trong hai:

- **che logits theo chiều** lúc suy luận - giới hạn lấy mẫu chỉ trong tập token
  của ngôn ngữ đích. Rẻ, chính xác, kết hợp tốt với việc cắt.
- hoặc một **thẻ ngôn ngữ** trong đầu ra.

Vì ASR đã đo LID đạt 100% ở cả bốn ngôn ngữ, nhầm lẫn nên là hiếm - nhưng "hiếm"
không phải "không thể", và phần che logits rất rẻ. **Hãy đo trước khi xây.**

## 5. Loại kiến thức (nên làm, tùy chọn)

Ngoài mức khôi phục, bạn có thể đẩy mô hình về đúng nhiệm vụ hẹp mà ta triển
khai: bốn ngôn ngữ, dịch câu đơn giản, không chế độ chỉ dẫn (kiểm soát thuật
ngữ, chuyển văn phong, xử lý dữ liệu có cấu trúc), không 29 ngôn ngữ khác.

Hai cách:

- **Về hành vi:** huấn luyện trên tập nhiệm vụ hẹp cho tới khi các chế độ không
  dùng biến mất. Rẻ.
- **Về cấu trúc:** sau khi đo thấy token của chế độ chỉ dẫn và của ngôn ngữ nước
  ngoài không được dùng, loại bỏ chúng trong một lượt cắt thứ hai.

Hãy đo trước lượt cắt thứ hai. Lượt cắt đầu tiên là phần quan trọng; lượt thứ
hai là nơi rất dễ làm hỏng mô hình mà không thu được lợi ích đo được nào.

## 6. Dữ liệu

### 6.1 Bạn cần văn bản song ngữ, không phải transcript của audio

Nếu đến từ bối cảnh ASR: kho giọng nói lớn **không** cho bạn dữ liệu MT. Văn bản
từ giọng nói là một ngôn ngữ. Khoảng 50 nghìn câu một ngôn ngữ chỉ bằng 1-2% khối
lượng một lần chạy MT.

### 6.2 Dữ liệu huấn luyện

```bash
hf download Helsinki-NLP/opus-100 --repo-type dataset \
  --include "en-vi/*" "en-zh/*" "en-ko/*" --local-dir ./data/opus100
```

Đã kiểm tra có sẵn: `en-vi/`, `en-zh/`, `en-ko/`, mỗi cặp có
`train/validation/test-00000-of-00001.parquet`.

**Hãy kiểm tra `en-zh` là giản thể hay hỗn hợp trước khi tin kết quả tiếng Trung.**

### 6.3 Tập đánh giá - FLEURES song ngữ

267 câu căn cứ câu, căn chỉnh trên en/vi/zh/ko, dựng từ các tập kiểm thử FLEURS
theo mã câu chung. Có trong gói kiểm thử thiết bị dưới tên
`data/mt/fleurs_parallel.jsonl`:

```json
{"id": 1660,
 "en": "Romanticism had a large element of cultural determinism...",
 "ko": "낭만주의는 괴테나 피히테, 슐레겔과 같은 작가들로부터...",
 "vi": "Chủ nghĩa lãng mạn có một yếu tố lớn về quyết định văn hóa...",
 "zh": "浪漫主义具有很强的文化决定论色彩。..."}
```

**Không bao giờ huấn luyện trên tập này.** Đường cơ sở đã đo để vượt: trung bình
**BLEU 33,71**, **chrF 44,32**.

`facebook/flores` devtest (1012 câu/ngôn ngữ) là bộ chuẩn quốc tế độc lập, nhưng
kho này **bị giới hạn truy cập** - cần tài khoản và chấp nhận điều khoản.
`Helsinki-NLP/flores-200` cũng trả về 401.

### 6.4 Sinh cặp giả cho cả sáu chiều

Vì chi phí GPU không phải ràng buộc, hãy sinh cặp giả **trực tiếp cho cả sáu
chiều**, không chỉ qua trục tiếng Anh. Dịch mỗi câu một ngôn ngữ sang ba ngôn ngữ
còn lại, tạo thành một tập dữ liệu bốn ngôn ngữ thực sự. Điều này phục vụ trực
tiếp cho các chiều không qua trục, vốn yếu nhất (`ko->en` BLEU 26,41 so với
`en->zh` 42,18).

Dùng giáo viên **decoder-only**, không phải encoder-decoder - các nghiên cứu
công bố cho thấy giáo viên decoder-only vượt trội trên nhiệm vụ này.

Giữ dữ liệu tổng hợp riêng khỏi cặp thật và giới hạn khoảng 50%; nhãn tổng hợp
thừa hưởng lỗi của giáo viên và có thể khuếch đại lỗi đó.

## 7. Huấn luyện

### 7.1 Bộ nhớ

Ở 40 GB, mô hình 2B vừa khớp: trọng số 4 GB + gradient 4 GB + Adam fp32 16 GB +
master fp32 8 GB + activation ~5 GB = ~37 GB.

### 7.2 Toàn tham số, không dùng LoRA

**LoRA không thích hợp cho tác vụ dịch.** Hai nghiên cứu lâm lý năm 2026 độc lập:

> *"không khuyến nghị dùng LoRA, vì nó chỉ cho cải thiện hiệu quả rất hạn chế"*

> *"tinh chỉnh toàn tham số và giáo viên decoder-only vượt trội hơn LoRA cũng như
> cách tiếp cận encoder-decoder"*

Các vòng huấn luyện ASR của chính dự án này cũng thống nhất - LoRA (+0,70) và
full FT (+0,59) đều thất bại ở cổng đánh giá, trong khi lượng tử hóa có nhận thức
đối với bề mặt triển khai thì đạt. Hạng LoRA không tạo ra khác biệt nào qua ba
mức trong thí nghiệm công bố.

### 7.3 Thiết lập

| thiết lập | giá trị | lý do |
|---|---|---|
| độ chính xác | bf16 | tiết kiệm ~12 GB so với fp32 |
| bộ tối ưu hóa | AdamW fp32 | dùng 8-bit Adam nếu phiên bản nhỏ hơn |
| phương pháp | **toàn tham số** | xem 7.2 |
| độ dài chuỗi | 512 | đầu ra Flores dài khoảng 65 token |
| che loss | chỉ token đích | token prompt ở mức -100 |
| lịch trình | cosine, warmup 3% | |
| batch hiệu dụng | 256-512 qua tích luỹ gradient | |

### 7.4 Hai lỗi âm thầm phá hỏng lần chạy tinh chỉnh

1. **Huấn luyện bằng prompt khác với prompt dùng khi triển khai.** Giáo viên đã
   thấy chỉ dẫn ở mục 3.3; nếu học sinh huấn luyện trên văn bản thuần, nó học
   một hợp đồng vào/ra khác. Hãy giữ mẫu prompt trong một hằng số dùng chung.
2. **Lấy trung bình kết quả đánh giá trên các ngôn ngữ.** Tiếng Anh sẽ đẹp và
   che lỗi ở tiếng Việt. Luôn báo cáo theo từng chiều.

## 8. Lượng tử hóa và triển khai

```bash
./llama-quantize ./trimmed-student-f16.gguf ./student-Q6_K.gguf Q6_K
```

**Các kiểu trọng số HTP hỗ trợ: Q4_0, Q4_1, Q8_0, Q5_K, Q6_K, MXFP4.**
**`Q4_K` KHÔNG được hỗ trợ** - không có kích thước tile cho HTP, âm thầm rơi về
CPU. Hãy kiểm tra kiểu lượng tử hóa trước khi xuất hàng.

**Hãy dùng Q6_K.** Bộ nhớ máy chủ gần như không phụ thuộc kiểu lượng tử hóa trên
HTP - mô hình 6,16 GB chỉ dùng 596-609 MB RSS vì trọng số được ánh xạ vào phiên
NPU thay vì giữ trong RAM máy chủ - nên không có áp lực bộ nhớ để hạ xuống. Q4_0 là
định dạng 4-bit **kém nhất**; hãy dùng IQ4_NL hoặc Q5_K nếu buộc phải giảm. Ở mức
1,47 GB, 1.8B đã nằm **gọn trong cửa sổ ánh xạ 3,5 GB của NPU**.

## 9. Đánh giá

1. **FLEURES song ngữ, cả sáu chiều**, BLEU và chrF **theo từng chiều**. Đường cơ
   sở để vượt: trung bình BLEU 33,71, chrF 44,32.
2. **Khoảng cách so với giáo viên** - chạy giáo viên trên cùng đầu vào.
3. **Vòng lặp thực tế** - đầu ra ASR → MT của học sinh, trên audio đã ghi thật.
4. **Thiết bị** - ms/token trên HTP, trung vị và p95 trên hơn 50 câu. Mục tiêu
   <= 50 ms/token; 1.8B chưa cắt đo được 34,7.
5. **RTF** - ASR + MT + TTS trên đoạn 10 giây. Phải dưới 1,0; 1.8B dự kiến 0,73.
6. **Kiểm tra token cấm** - xác nhận mô hình đã cắt không thể phát ra token thuộc
   ngôn ngữ không phải đích. Đây là mục đích của việc cắt; hãy kiểm tra.

Các bước 4-6 **phải** chạy trên thiết bị đích. Số đo trên máy chủ gây hiểu lầm:
khả năng lượng tử hóa khác nhau giữa các backend.

## 10. Những cạm bẫy thường gặp

| triệu chứng | nguyên nhân |
|---|---|
| học sinh tốt ở tiếng Anh, kém ở tiếng Việt | mô hình nền yếu ở vi |
| học sinh trả về văn bản thuần, bỏ qua chỉ dẫn | prompt huấn luyện lệch với triển khai |
| nhãn nhiễu, thiếu nhất quán | giáo viên chạy ở 0,7 thay vì 0,0 |
| ms/token trên thiết bị kém hơn nhiều so với máy chủ | lượng tử hóa là Q4_K, hoặc thao tác rơi về CPU |
| log MT không có dòng `HTP` | không tìm thấy skel; `ADSP_LIBRARY_PATH` phải trỏ tới `lib/` |
| lần chạy sụp giữa chừng | hết VRAM. Kiểm tra thời gian mỗi bước |
| mọi thứ ổn rồi thiết bị biến mất | phiên bản cloud đổi mà không báo |
| chrF trông hỏng với ko/zh | là kỳ vọng - trộn hệ chữ viết. So BLEU giữa các hệ |

### Lưu ý môi trường thiết bị

- Thiết bị **không có Python** ngoài chroot, và **không có curl/wget** - chỉ có `nc`.
- **DNS:** rootfs Debian đi kèm `1.1.1.1`, không phải lúc nào cũng định tuyến được.
  Triệu chứng là `Temporary failure in name resolution` dù kết nối TCP vẫn được.
  `8.8.8.8` thường hoạt động. `scripts/setup.sh` hiện đã kiểm tra và sửa.
- Tải mô hình trên thiết bị chạy ~17 MB/s, so với ~7 MB/s đẩy qua adb.
- Chấm điểm **trên thiết bị** bằng bộ BLEU/chrF++ thuần stdlib đi kèm, và chỉ kéo
  về JSON nhỏ. Hãy kiểm tra nó trước: bộ tự kiểm khẳng định trường hợp giống hệt
  tham chiếu đạt đúng 100,00 ở mọi chiều.

## 11. Thứ tự công việc

1. Tải OPUS-100; xác nhận số câu
2. **Đo mức sử dụng token** trên tập dữ liệu để chọn kích thước cắt
3. Tải giáo viên (Hy-MT2-30B-A3B Q4_K_M) và sinh nhãn ở `temperature 0.0`, cả sáu
   chiều
4. Cắt 120.818 -> N
5. Tinh chỉnh toàn tham số để khôi phục
6. Tùy chọn: lượt loại kiến thức
7. Lượng tử hóa Q6_K, kiểm tra kiểu tensor và khả năng với tới token cấm
8. Đánh giá trên tập song ngữ, theo từng chiều
9. Đo ms/token, kiểm tra token cấm, và RTF trên thiết bị
10. Nếu RTF >= 1,0, đòn bẩy là mô hình nhỏ hơn hoặc đầu ra ngắn hơn - **không**
    phải lượng tử hóa thấp hơn, vì bộ nhớ trên NPU đã không phải vật cản

---

## Ghi chú riêng về tiếng Việt

- Tiếng Việt là ngôn ngữ **sinh ra nhiều token hơn** đầu vào. Chi phí MT tăng theo
  **số token đầu ra**, không theo thời lượng âm thanh đầu vào.
- Trong bốn ngôn ngữ, **tiếng Việt là điểm yếu đã biết**: một vòng huấn luyện có
  thêm VietSuperSpeech (chỉ tiếng Việt) đã cải thiện vi nhưng không cải thiện ko.
  Bằng chứng cho thấy khoảng trống là thiếu dữ liệu **hội thoại** ở ngôn ngữ yếu.
- Đo **riêng từng chiều**, và riêng chiều vi->X. Chất lượng trung bình của cả bốn
  ngôn ngữ có thể che mất một hướng đã hỏng.
- Cặp `en-vi` của OPUS-100 là dữ liệu huấn luyện chính. Nếu dùng thêm dữ liệu giả
  cho hướng vi->X, hãy cân nhắc tỉ lệ riêng vì đây là ngôn ngữ có tài nguyên dữ
  liệu song ngữ hạn chế.
- Với giáo viên **decoder-only** như khuyến nghị, hãy kiểm tra đầu ra tiếng Việt
  thận trọng: đây là hướng mà chúng tôi đã thấy mô hình khác thất bại trước đây.