import json
import urllib.request
from pathlib import Path

BASE = 'http://127.0.0.1:8001'
IMG_PATH = Path(r'C:\Users\moham\Desktop\ONISCAN\dataset\raw\Onion-Bad.jpg')

root_bytes = urllib.request.urlopen(BASE + '/', timeout=30).read()
print('ROOT_JSON', root_bytes.decode())
app_html = urllib.request.urlopen(BASE + '/app', timeout=30).read(500).decode('utf-8', 'replace')
print('APP_HTML_PREFIX', app_html)

boundary = '----ONISCAN-Boundary-123'
content = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="{IMG_PATH.name}"\r\n'
    'Content-Type: image/jpeg\r\n\r\n'
).encode() + IMG_PATH.read_bytes() + (f'\r\n--{boundary}--\r\n').encode()

req = urllib.request.Request(
    BASE + '/analyze',
    data=content,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
    method='POST',
)
with urllib.request.urlopen(req, timeout=180) as resp:
    payload_text = resp.read().decode('utf-8', 'replace')
    print('ANALYZE_STATUS', resp.status)
    print('ANALYZE_PAYLOAD', payload_text)
    payload = json.loads(payload_text)
    detections = payload.get('detections') or []
    print('DETECTION_FIELDS', all({'class', 'confidence', 'box'} <= set(item) for item in detections))
    annotated_image = payload.get('annotated_image') or payload.get('result_image')
    if annotated_image:
        result_url = BASE + annotated_image
        result_bytes = urllib.request.urlopen(result_url, timeout=30).read(256)
        print('ANNOTATED_IMAGE_STATUS', 'OK', len(result_bytes), 'bytes')
        print('RESULT_PREFIX', result_bytes[:32])
