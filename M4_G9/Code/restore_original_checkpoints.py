"""Restore original Google Drive CausalLOB checkpoint tree from GitHub-sized parts.

From the M4_G9 root:
  python Code/src/restore_original_checkpoints.py --output /tmp/CausalLOB_FROZEN_REPRO_EXPORT.zip
  python Code/src/restore_original_checkpoints.py --output /tmp/CausalLOB_FROZEN_REPRO_EXPORT.zip --extract-root /content/drive/MyDrive/CausalLOB

Existing files are NOT overwritten if their hashes differ from the frozen export.
Requires Python stdlib only; no network or GitHub account.
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[2]
PARTS=ROOT/'Code/checkpoints'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('/tmp/CausalLOB_FROZEN_REPRO_EXPORT.zip'))
    parser.add_argument('--extract-root',type=Path,default=None)
    args=parser.parse_args()
    info=json.loads((PARTS/'PARTS_MANIFEST.json').read_text())
    output=args.output.expanduser().resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() and sha256(output)==info['sha256']:
        print('Archive already reconstructed and verified:',output)
    else:
        if output.exists():
            raise RuntimeError(f'Output file exists but has wrong SHA-256; refusing to overwrite: {output}')
        for item in info['parts']:
            f=PARTS/item['name']
            assert f.stat().st_size==item['size_bytes'],f'part size mismatch: {f}'
            assert sha256(f)==item['sha256'],f'part digest mismatch: {f}'
        with open(output,'wb') as out:
            for item in info['parts']:
                with open(PARTS/item['name'],'rb') as inp:
                    shutil.copyfileobj(inp,out,8*1024*1024)
        assert output.stat().st_size==info['size_bytes'],'Reconstructed archive length mismatch'
        assert sha256(output)==info['sha256'],'Reconstructed archive SHA mismatch'
        print('Reconstructed archive and verified original SHA-256:',output)
    with zipfile.ZipFile(output) as z:
        source=json.loads(z.read('FROZEN_CHECKPOINT_MANIFEST.json'))
        for name,metadata in source['files'].items():
            assert name in z.namelist(),name
        print('Original checkpoint file count:',len(source['files']))
        if args.extract_root is not None:
            dest=args.extract_root.expanduser().resolve()
            dest.mkdir(parents=True,exist_ok=True)
            extracted=already=0
            for name, metadata in source['files'].items():
                pure=PurePosixPath(name)
                assert not pure.is_absolute() and '..' not in pure.parts and '\\' not in name,name
                target=(dest/name).resolve()
                assert target.is_relative_to(dest),'Unsafe archive filename'
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if sha256(target)!=metadata['sha256']:
                        raise RuntimeError('Refusing to replace a different existing file: '+str(target))
                    already+=1
                    continue
                with z.open(name,'r') as source_f,open(target,'xb') as output_f:
                    shutil.copyfileobj(source_f,output_f,8*1024*1024)
                if sha256(target)!=metadata['sha256']:
                    raise RuntimeError('Restored file has an unexpected hash: '+str(target))
                extracted+=1
            print(f'RESTORE VERIFIED: {extracted} files created; {already} preexisting identical; original frozen files not changed')

if __name__=='__main__':main()
