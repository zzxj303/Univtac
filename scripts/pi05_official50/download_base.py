"""Download generation-pinned GCS objects, verifying CRC32C (and MD5 when available)."""
import base64,hashlib,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,google_crc32c
ROOT=Path(__file__).resolve().parents[2]
STAGING=Path('/tmp/univtac_pi05_base_official')
AUDIT=ROOT/'outputs/pi05_official50/setup/base_source.json'
if AUDIT.exists():info=json.loads(AUDIT.read_text())
else:
 response=requests.get('https://storage.googleapis.com/storage/v1/b/openpi-assets/o',params={'prefix':'checkpoints/pi05_base/','maxResults':1000},timeout=60);response.raise_for_status();data=response.json();assert not data.get('nextPageToken')
 info={'source':'gs://openpi-assets/checkpoints/pi05_base','objects':data['items']};AUDIT.write_text(json.dumps(info,indent=2))
def verify(path,item):
 if not path.exists() or path.stat().st_size!=int(item['size']):return False
 crc=google_crc32c.Checksum();md5=hashlib.md5();sha=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):crc.update(b);md5.update(b);sha.update(b)
 assert base64.b64encode(crc.digest()).decode()==item['crc32c'],f'CRC32C mismatch: {path}'
 if 'md5Hash' in item:assert base64.b64encode(md5.digest()).decode()==item['md5Hash'],f'MD5 mismatch: {path}'
 return sha.hexdigest()
def download(item):
 target=STAGING/Path(item['name']).relative_to('checkpoints/pi05_base');target.parent.mkdir(parents=True,exist_ok=True)
 checksum=verify(target,item)
 if not checksum:
  temp=target.with_name(target.name+'.partial');checksum=verify(temp,item)
  if not checksum:
   offset=temp.stat().st_size if temp.exists() else 0
   response=requests.get('https://storage.googleapis.com/openpi-assets/'+item['name'],params={'generation':item['generation']},headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,120));response.raise_for_status()
   with temp.open('ab' if offset and response.status_code==206 else 'wb') as f:
    for chunk in response.iter_content(4*1024*1024):f.write(chunk)
   checksum=verify(temp,item);assert checksum,target
  temp.replace(target)
 print('VERIFIED',target,flush=True)
 return {'name':item['name'],'generation':item['generation'],'bytes':int(item['size']),'sha256':checksum,'crc32c':item['crc32c'],'md5':item.get('md5Hash')}
with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(download,info['objects']))
(AUDIT.parent/'base_download_checksums.json').write_text(json.dumps(checks,indent=2))
print('BASE_DOWNLOAD_VERIFIED',flush=True)
