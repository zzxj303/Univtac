"""Trusted simulation service outside the model process's inherited GPU restrictions.
Only typed repair operations for one registered session are accepted. Native workers
still use the existing Landlock write confinement and simulation budget ledger.
"""
import argparse,json,os,socket,subprocess,time
from pathlib import Path
from orchestrator.common import ROOT,PYTHON,read,atomic,environment,terminate
from adapters.repairs import session_path


def start(session):
    session=session_path(session)
    # Short Unix-socket paths stay below the Linux 108-byte limit.
    import uuid
    address=ROOT/'runtime'/('repair-'+uuid.uuid4().hex[:12]+'.sock')
    log=(session/'broker.log').open('ab')
    proc=subprocess.Popen([str(PYTHON),'-m','adapters.repair_broker','--session',str(session),'--socket',str(address)],
                          cwd=ROOT,env=environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    for _ in range(200):
        if address.exists() and (session/'broker.json').exists() and read(session/'broker.json').get('pid') == proc.pid:return proc,address
        if proc.poll() is not None:raise RuntimeError('Simulation broker failed; inspect broker.log')
        time.sleep(.05)
    terminate(proc,grace=5);raise TimeoutError('Simulation broker startup')

def request(session,operation,args):
    session=session_path(session);info=read(session/'broker.json');address=info['socket']
    if not Path(address).is_relative_to(ROOT/'runtime'):raise ValueError('Broker address outside runtime')
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
        client.settimeout(1850);client.connect(address)
        client.sendall((json.dumps({'operation':operation,'args':args})+'\n').encode())
        stream=client.makefile('rb');line=stream.readline(2**20)
    if not line:raise RuntimeError('Broker disconnected; reconcile running trial')
    response=json.loads(line)
    if not response['ok']:raise RuntimeError(response['error'])
    return response['result']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session',required=True);parser.add_argument('--socket',required=True);args=parser.parse_args()
    session=session_path(args.session);address=Path(args.socket)
    from adapters.registry import get_adapter
    adapter=get_adapter(read(session/'session.json')['group'])
    if not address.is_relative_to(ROOT/'runtime'):raise ValueError('Invalid socket path')
    import signal
    def stop(*_):raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop)
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
            server.bind(str(address));os.chmod(address,0o600);server.listen(1);server.settimeout(18000)
            atomic(session/'broker.json',{'pid':os.getpid(),'socket':str(address),'session':str(session),'started_at':time.time()})
            while True:
                client,_=server.accept()
                with client:
                    try:
                        stream=client.makefile('rb');payload=json.loads(stream.readline(2**20));a=payload['args']
                        if payload['operation']=='try':
                            if set(a)!={'demo_id','patch'}:raise ValueError('Invalid repair arguments')
                            result=adapter.try_repair(session,a['demo_id'],a['patch'])
                        elif payload['operation']=='verify':
                            if set(a)!={'demo_id','candidate'}:raise ValueError('Invalid verification arguments')
                            result=adapter.verify_repair(session,a['demo_id'],a['candidate'])
                        else:raise ValueError('Unrecognized operation')
                        response={'ok':True,'result':result}
                    except Exception as e:response={'ok':False,'error':repr(e)}
                    client.sendall((json.dumps(response,allow_nan=False)+'\n').encode())
    finally:
        if address.exists():address.unlink()
if __name__=='__main__':main()
