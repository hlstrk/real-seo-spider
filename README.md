# real-seo-spider

`lemaclinic.com` gibi bir domaini tarar; sayfalardan `title`, `meta description`, `H1` ve görsellerin `alt` + dosya adı bilgisini çıkarır. Uygun olmayanları CSV/JSON çıktısı olarak üretir. İstersen `gpt-4o-mini` ile ek kontrol yaptırır.

## Kurulum

```powershell
python -m pip install -r requirements.txt
```

Chromium ile render ederek tarama yapmak için Playwright Chromium’u da indir:

```powershell
python -m playwright install chromium
```

## OpenAI API Key

Tercih edilen yöntem: env var

```powershell
$env:OPENAI_API_KEY="..."
```

Alternatif: `config.example.py` → `config.py` kopyalayıp `OPENAI_API_KEY` alanına ekleyebilirsin.

## Scope (hangi sayfalar taranacak?)

Script **domain + path prefix** ile scope belirler.

- İngilizce sayfalar `/en/` altındaysa: `--scope-prefix /en/`
- İngilizce sayfalar direkt domain altındaysa: `--scope-prefix /`

İstersen `<html lang="...">` üzerinden filtre de uygulayabilirsin:

- İngilizce için: `--require-lang en`

## CLI Çalıştırma

Sadece verilen URL (varsayılan `single`):

```powershell
python .\seo_audit.py --start https://lemaclinic.com/ --scope-prefix /
```

Chromium ile (render ederek):

```powershell
python .\seo_audit.py --fetcher chromium --start https://lemaclinic.com/ --scope-prefix /
```

Tüm sayfaları gezmek için (`crawl`):

```powershell
python .\seo_audit.py --mode crawl --start https://lemaclinic.com/ --scope-prefix / --max-pages 200
```

Sitemap üzerinden seed etmek için:

```powershell
python .\seo_audit.py --mode crawl --use-sitemap --scope-prefix / --max-pages 200
```

Çıktılar:

- `out/issues.csv`
- `out/report.json`

GPT’yi kapatmak için:

```powershell
python .\seo_audit.py --no-gpt
```

## Basit GUI (Tkinter)

```powershell
python .\seo_audit_gui.py
```

## Web Uygulaması (FastAPI)

LAN’dan erişmek için (örn. `http://192.168.1.100:8000`) web uygulamasını başlat:

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
.\run_web.ps1
```

Sonra tarayıcıdan:

- `http://192.168.1.100:8000` (veya bu PC’nin LAN IP’si)

Web arayüzünde tarama başlatınca otomatik olarak **Sonuçlar** sayfasına yönlendirir ve `issues.csv` / `report.json` indirme linklerini verir.

Not: `http://192.168.1.100` (portsuz, 80) istiyorsan `uvicorn`u `--port 80` ile çalıştırman gerekir (Windows’ta genelde yönetici izni ister).

### Login

Web uygulamasında basit bir giriş ekranı var:

- Kullanıcı adı: `admin`
- Şifre: `1234`

### “Sorunu gör” (screenshot)

Checklist’teki **Sorunu gör** linki, ilgili sorunu Chromium ile render edip ekran görüntüsü alır ve mümkünse problemi kırmızı kutu ile işaretler.
