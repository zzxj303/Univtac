"""Prepare 50 episodes with upstream decode/resize and qpos(t+1) convention."""
import json,os
from pathlib import Path
import cv2,h5py,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/official50'
TASKS=['insert_HDMI','insert_hole','pull_out_key']
keys={'cam_high':'observation/head/rgb','cam_wrist':'observation/wrist/rgb','tac_left':'tactile/left_tactile/rgb_marker','tac_right':'tactile/right_tactile/rgb_marker'}
registry={}
for task in TASKS:
 dst=ROOT/'data/official50'/task;dst.mkdir(parents=True,exist_ok=True)
 for i in range(50):
  target=dst/f'episode_{i}.hdf5'
  source=ROOT/'data/isaac51'/task/'hdf5'/f'{i}.hdf5'
  with h5py.File(source) as src:
   q=src['embodiment/joint'][:,:8].astype('float32')
   if not target.exists():
    old=ROOT/'data/act_six'/task/f'episode_{i}.hdf5'
    if old.exists():
     with h5py.File(old) as f:
      np.testing.assert_array_equal(f['action'][:],q[1:]);np.testing.assert_array_equal(f['observations/qpos'][:],q[:-1])
     target.symlink_to(old)
    else:
     tmp=target.with_suffix('.tmp')
     with h5py.File(tmp,'w') as f:
      f.create_dataset('action',data=q[1:]);f.create_dataset('observations/qpos',data=q[:-1])
      for name,key in keys.items():
       images=np.stack([cv2.resize(cv2.imdecode(np.frombuffer(bytes(x),dtype='uint8'),cv2.IMREAD_COLOR),(256,256),interpolation=cv2.INTER_LINEAR) for x in src[key][:-1]])
       f.create_dataset('observations/images/'+name,data=images,compression='lzf',chunks=(1,256,256,3))
     tmp.rename(target)
  with h5py.File(target) as f:
   assert f['action'].shape[1]==8
   for name in keys: assert f['observations/images/'+name].shape==(len(q)-1,256,256,3)
  print(task,i,'ready',flush=True)
 split=np.random.RandomState(1).permutation(50).tolist()
 (dst/'split.json').write_text(json.dumps({'seed':1,'train_ids':split[:40],'validation_ids':split[40:],'source_ids':list(range(50))},indent=2))
 name=f'sim-{task}-official50-50'
 registry[name]={'dataset_dir':str(dst),'num_episodes':50,'episode_len':max(h5py.File(dst/f'episode_{i}.hdf5')['action'].shape[0] for i in range(50))}
 for mode in ['rgb','rgb_tactile']:
  out=OUT/'act'/task/mode;out.mkdir(parents=True,exist_ok=True)
  cfg=yaml.safe_load((ROOT/'policy/ACT'/('train_config_vision_all.yml' if mode=='rgb' else 'train_config_all.yml')).read_text())
  cfg.update(num_epochs=6000,tactile_ckpt=str(ROOT/'data/checkpoints/encoder.pth'))
  (out/'config.yml').write_text(yaml.safe_dump(cfg))
  (out/'split.json').write_text((dst/'split.json').read_text())
(OUT/'SIM_TASK_CONFIGS.json').write_text(json.dumps(registry,indent=2))
