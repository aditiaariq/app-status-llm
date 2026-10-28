"""Klasifikasi status akademik mahasiswa menggunakan LLM (Groq API).
File ini dipakai saat EVALUASI (Modul 2A) dan saat DEPLOY di Streamlit (Modul 2B)
agar perilaku model yang dievaluasi sama persis dengan yang di-deploy."""
import json
import re
import time

import groq

# nama fitur yang ditampilkan ke LLM
NAMA_FITUR = {
    "ipk": "IPK",
    "kehadiran": "Kehadiran (%)",
    "jam_belajar": "Jam belajar mandiri per minggu",
    "tugas_terkumpul": "Tugas terkumpul (%)",
    "nilai_uts": "Nilai UTS",
    "aktif_organisasi": "Aktif organisasi",
}


def load_config(path="config.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_data(data):
    """dict data mahasiswa -> teks satu baris untuk LLM."""
    return "; ".join(f"{nama}: {data[kolom]}" for kolom, nama in NAMA_FITUR.items())


def build_messages(cfg, data):
    msgs = [{"role": "system", "content": cfg["system_prompt"]}]
    for ex in cfg.get("few_shot", []):                      # contoh (few-shot), boleh kosong
        msgs.append({"role": "user", "content": format_data(ex["data"])})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": format_data(data)})
    return msgs


def parse_label(output, labels):
    """Rapikan jawaban model: 'waspada.' -> 'Waspada'. Di luar daftar label -> 'invalid'."""
    if not output:
        return "invalid"
    jawab = re.sub(r"[^a-z]", "", output.lower())
    peta = {label.lower(): label for label in labels}
    if jawab in peta:
        return peta[jawab]
    # Fallback pencarian kata kunci jika model menambahkan awalan/tanda baca
    for word in re.findall(r"\b[a-zA-Z]+\b", output):
        if word.lower() in peta:
            return peta[word.lower()]
    return "invalid"


def classify(client, cfg, data, max_retry=5):
    for percobaan in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                temperature=cfg["temperature"],
                max_tokens=cfg.get("max_tokens", 10),
                messages=build_messages(cfg, data),
            )
            content = resp.choices[0].message.content or ""
            return parse_label(content, cfg["labels"])
        except groq.RateLimitError:
            wait_time = 15 * (percobaan + 1)
            print(f"\n[RateLimit] Menunggu {wait_time}s sebelum mencoba ulang...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"\n[API Error] {e}, mencoba ulang...")
            time.sleep(3 * (percobaan + 1))
    return "invalid"
