"""Native physical demonstration replay. No model weights, no forced actor placement."""
import argparse, json, os, shutil, sys, time, traceback
from pathlib import Path
from orchestrator.common import ROOT, VENDOR, read, atomic, require_vendor_imports

def main():
    early=argparse.ArgumentParser(add_help=False);early.add_argument('--request',required=True)
    args,_=early.parse_known_args(); request=read(args.request);folder=Path(request['folder']);seed=request['seed']
    app=task=None;phase='startup';started=time.time();result={'success':False,'phase':phase,'actions':0,'seed':seed}
    def progress(): atomic(folder/'progress.json',{'phase':phase,'elapsed_s':time.time()-started,'actions':result['actions']})
    try:
        os.environ['XDG_CACHE_HOME']=str(ROOT/'runtime/baseline-cache')
        cfg=ROOT/'runtime/baseline-user.config.json'
        if cfg.exists(): shutil.copyfile(cfg,folder/'kit-user.config.json')
        from isaaclab.app import AppLauncher
        parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);AppLauncher.add_app_launcher_args(parser)
        a=parser.parse_args();a.enable_cameras=True;a.num_envs=1;a.livestream=0
        a.kit_args=(getattr(a,'kit_args','')+' --portable-root '+str(folder/'kit-runtime')+' --/app/tokens/logs='+str(folder/'kit-logs')+' --/app/tokens/cache='+os.environ['XDG_CACHE_HOME']+' --/log/file='+str(folder/'kit.log')+' --/app/userConfigPath='+str(folder/'kit-user.config.json')+' --/app/localCachePath='+os.environ['XDG_CACHE_HOME']+' --/app/cachePath='+os.environ['XDG_CACHE_HOME'])
        progress();app=AppLauncher(a).app
        import numpy as np,torch,h5py,cv2
        from envs.utils.env_parser import load_task_config,create_task_env
        config,path=load_task_config('clean51',[])
        phase='create_environment';progress()
        env=create_task_env(request['task'],config,path.stem,'eval',device=a.device,save_dir=folder/'native')
        task=env.task
        atomic(folder/'imports.json',require_vendor_imports(['envs.utils.env_parser','envs._base_task']))
        phase='reset';progress();task.reset(seed=seed,instructions=['Empty'])
        phase='validate_commands';progress()
        with h5py.File(request['commands'],'r') as f:commands=f['action'][:]
        with h5py.File(request['reference'],'r') as f:reference=f['action'][:]
        if commands.shape!=reference.shape or commands.ndim!=2 or commands.shape[1]!=8 or not np.isfinite(commands).all(): raise ValueError('Invalid command shape/values')
        # Fixed before executing either original or candidate; full demonstration length, not policy-test budget.
        task.cfg.step_lim=len(commands);task.mean_steps=len(commands)
        rm=task._robot_manager;ids=torch.cat([rm._arm_ids,rm._gripper_ids[:1]])
        bounds=rm.robot.data.soft_joint_pos_limits[0,ids].detach().cpu().numpy()
        if np.any(commands<bounds[:,0]-1e-5) or np.any(commands>bounds[:,1]+1e-5): raise ValueError('Commands exceed native joint limits')
        atomic(folder/'protocol.json',{'seed':seed,'source_seed':seed,'config':'clean51','force':False,'action_type':'qpos','action_limit':len(commands),'decimation':task.cfg.decimation,'joint_bounds':bounds.tolist(),'sampling':'initial observation plus one observation after every action','labels':'next observed qpos; issued commands stored separately'})
        phase='replay';progress();errors=[];early_stop=False
        from adapters.datasets import CHANNELS
        def array(x): return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
        with h5py.File(folder/'recording.hdf5','x') as out, (folder/'trajectory.jsonl').open('x') as trace:
            qds=out.create_dataset('embodiment/joint',(len(commands)+1,9),dtype='float32')
            images={name:out.create_dataset(key,(len(commands)+1,),dtype=h5py.vlen_dtype(np.dtype('uint8'))) for name,key in CHANNELS.items()}
            def capture(index):
                obs=task._get_observations();q=array(obs['embodiment']['joint']).reshape(-1);qds[index]=q[:9]
                for name,key in CHANNELS.items():
                    value=obs
                    for part in key.split('/'):value=value[part]
                    img=array(value).squeeze()
                    if img.dtype!=np.uint8:img=np.clip(img,0,255).astype('uint8')
                    ok,jpg=cv2.imencode('.jpg',img)
                    if not ok:raise ValueError('Cannot encode live image')
                    images[name][index]=jpg.reshape(-1)
                return q
            before=capture(0)
            for i,command in enumerate(commands):
                execution,_=task.take_action(torch.as_tensor(command,device=task.device,dtype=torch.float32),action_type='qpos',force=False,stop_on_success=False)
                after=capture(i+1);result['actions']=i+1;errors.append((after[:8]-reference[i]).tolist())
                early_stop=bool(task.check_early_stop())
                trace.write(json.dumps({'index':i,'state_before':before.tolist(),'command':command.tolist(),'state_after':after.tolist(),'execution_success':bool(execution),'success':bool(task.check_success()),'early_stop':early_stop})+'\n');trace.flush();before=after
                if early_stop or not execution:break
                if (i+1)%20==0:progress()
            result['complete_trajectory']=result['actions']==len(commands)
            result['success']=bool(task.check_success()) and not early_stop and result['complete_trajectory']
            result['tracking_rmse']=float(np.sqrt(np.mean(np.square(errors)))) if errors else None
            result['early_stop']=early_stop
            out.attrs['valid_frames']=result['actions']+1
            out.attrs['complete_trajectory']=result['complete_trajectory']
        result['recording']=str(folder/'recording.hdf5')
        result['termination']='native_success' if result['success'] else 'native_early_stop' if early_stop else 'unsuccessful_replay'
    except BaseException as e:
        result.update(error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        result.update(phase=phase,elapsed_s=time.time()-started)
        if task is not None:
            try:task.clean_cache(result='success' if result['success'] else 'failed');task.close()
            except BaseException as e:result['cleanup_error']=repr(e)
        atomic(folder/'result.json',result)
        if app is not None:app.close()
if __name__=='__main__': main()
