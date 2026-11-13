"""Use the upstream converter, preserving FP32 expert weights before mixed storage."""
import os
os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT.parent/'OpenPI-RoboLab/examples/convert_jax_model_to_pytorch.py'
OUT=ROOT/'outputs/pi05_official50/base'
checks=json.loads((ROOT/'outputs/pi05_official50/setup/base_download_checksums.json').read_text())
assert len(checks)==len(json.loads((ROOT/'outputs/pi05_official50/setup/base_source.json').read_text())['objects'])
source=SOURCE.read_text()
source=source.replace('pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)', 'pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config).float()\n    pi0_model.paligemma_with_expert.gemma_expert.lm_head = torch.nn.Identity()')
source=source.replace('pi0_model.load_state_dict(all_params, strict=False)', '''result = pi0_model.load_state_dict(all_params, strict=False)
    print('LOAD_AUDIT', result, flush=True)
    allowed = {'paligemma_with_expert.paligemma.lm_head.weight'}
    assert not result.unexpected_keys, result
    assert set(result.missing_keys) <= allowed, result''')
source=source.replace('pi0_model = pi0_model.to(torch.float32)', 'pi0_model = pi0_model.to(torch.float32)\n        pi0_model.paligemma_with_expert.paligemma.to(torch.bfloat16)')
ns={'__name__':'upstream_conversion'}
exec(compile(source,str(SOURCE),'exec'),ns)
import torch
torch.set_num_threads(8)
from openpi.models.pi0_config import Pi0Config
ns['convert_pi0_checkpoint']('/tmp/univtac_pi05_base_official','float32',str(OUT),Pi0Config(pi05=True,action_horizon=50))
config=json.loads((OUT/'config.json').read_text());config.update(precision='vlm_bfloat16_expert_float32',pi05=True,max_token_len=200,discrete_state_input=True)
(OUT/'config.json').write_text(json.dumps(config,indent=2))
p=OUT/'model.safetensors';h=hashlib.sha256()
with p.open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
(OUT/'manifest.json').write_text(json.dumps({'sha256':h.hexdigest(),'bytes':p.stat().st_size,'upstream_converter_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'storage':'VLM BF16; expert and projections FP32','source_manifest':'../setup/base_source.json'},indent=2))
print('CONVERSION_VERIFIED',flush=True)
