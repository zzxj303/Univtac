"""Bounded HDF5 access and source-addressable numeric citations."""
import base64, json
import h5py
import numpy as np

def numeric(path, dataset, start, stop):
    if type(start) != int or type(stop) != int or not 0 <= start < stop <= start + 64:
        raise ValueError('Read 1..64 rows using an exclusive stop')
    with h5py.File(path,'r') as f:
        ds=f[dataset]
        if not isinstance(ds,h5py.Dataset) or ds.dtype.kind not in 'biuf' or ds.ndim > 2 or ds.ndim < 1:
            raise ValueError('Select a numeric vector/matrix, not an image')
        if stop>len(ds): raise ValueError('Slice past end')
        a=ds[start:stop]
        if a.size>4096 or not np.isfinite(a).all(): raise ValueError('Nonfinite/oversized numeric slice')
        return a.tolist()

def cited(path, field):
    dataset, indices=field[5:].rsplit('@',1)
    start,stop=map(int,indices.split(':'))
    return numeric(path,dataset,start,stop)

def read_hdf5(path, args):
    dataset=args.get('dataset')
    if not dataset:
        rows=[]
        with h5py.File(path,'r') as f:
            f.visititems(lambda n,o: rows.append({'dataset':n,'shape':list(o.shape),'dtype':str(o.dtype)}) if isinstance(o,h5py.Dataset) else None)
        return {'source':str(path),'datasets':rows}
    if args.get('frames') is not None:
        import cv2
        frames=args['frames']
        if not 1<=len(frames)<=3 or any(type(i)!=int or i<0 for i in frames): raise ValueError('Select 1..3 frame indices')
        content=[]
        with h5py.File(path,'r') as f:
            ds=f[dataset]
            for i in frames:
                if i>=len(ds): raise ValueError('Frame past end')
                a=ds[i]
                if np.asarray(a).ndim<3: a=cv2.imdecode(np.frombuffer(bytes(a),dtype='uint8'),cv2.IMREAD_COLOR)
                if a is None or a.ndim!=3 or a.shape[-1]!=3: raise ValueError('Not a supported RGB/marker image')
                h,w=a.shape[:2]; a=cv2.resize(a,(min(w,640),max(1,round(h*min(w,640)/w))))
                ok,jpg=cv2.imencode('.jpg',a)
                if not ok: raise ValueError('Image encoding failed')
                content += [{'type':'text','text':json.dumps({'source':str(path),'dataset':dataset,'frame':i})},
                            {'type':'image','mimeType':'image/jpeg','data':base64.b64encode(jpg).decode()}]
        return {'content':content}
    start=args.get('start',0); stop=args.get('stop',start+1)
    return {'source':str(path),'field':f'hdf5:{dataset}@{start}:{stop}','value':numeric(path,dataset,start,stop)}
