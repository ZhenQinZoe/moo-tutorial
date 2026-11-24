from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, re, subprocess, hashlib

root = Path(__file__).resolve().parent
dest = root / 'assets'
dest.mkdir(exist_ok=True)
urls = sorted({url for p in (root / 'content').glob('*.md') for url in re.findall(r'!\[[^\]]*\]\((https?://[^\s)]+)', p.read_text())})

def download(url):
    name = url.rsplit('/', 1)[-1]
    target = dest / name
    try:
        if not target.exists():
            data = subprocess.run(['/usr/bin/curl', '-L', '--fail', '--silent', '--show-error', '--max-time', '35', url], check=True, capture_output=True).stdout
            if not (data.startswith(b'\x89PNG') or b'<svg' in data[:1000]):
                raise ValueError('Response is not the expected image')
            target.write_bytes(data)
        return url, {'file': name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size}
    except Exception as error:
        return url, {'error': str(error)}

with ThreadPoolExecutor(max_workers=4) as pool:
    manifest = dict(pool.map(download, urls))
(dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
for url, value in manifest.items():
    print(value.get('file') or value['error'], value.get('bytes', ''), flush=True)
print('Downloaded:', sum('file' in value for value in manifest.values()), '/', len(urls))
