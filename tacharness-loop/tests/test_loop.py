import tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from orchestrator.common import ROOT,atomic,read
from orchestrator.experiment import group_loop
class ThreeRoundLoop(unittest.TestCase):
    def test_three_actions_tie_earliest_and_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime/tmp') as tmp:
            root=Path(tmp)
            baseline={'id':'g:baseline','checkpoint':'base','stats':'stats','config':'config','train_dir':'train'}
            candidate={**baseline,'id':'g:other','checkpoint':'other'}
            g={'id':'g','baseline':baseline,'feedback_seeds':[20001],'test_seeds':[20031],'original_steps':4000}
            atomic(root/'configs/token_budget.json',{'tokens_per_group':1000000})
            atomic(root/'runs/g/rounds.json',[{'round':0,'artifact':baseline,'feedback':'baseline-feedback','successes':0,'decision':'baseline'}])
            trained=[];evaluated=[];tested=[]
            def diagnose(group,k,rounds,training_used,budget):
                folder=root/'runs/g'/f'round-{k}'/'input'
                artifacts=[baseline,candidate]+[r['artifact'] for r in rounds[1:]]
                atomic(folder/'context.json',{'artifacts':artifacts})
                if k==1:return {'decision':'select_ckpt','init_from':baseline['id'],'parameters':{'checkpoint_ids':[candidate['id']]}},None
                return {'decision':'finetune_more' if k==2 else 'finetune_data','init_from':rounds[-1]['artifact']['id'],'parameters':{'steps':100,'lr_scale':1,'train_ids':[] if k==2 else [50]}},None
            def train(self,group,art,req,out):
                new={**art,'id':f'g:trained-{len(trained)+1}'};trained.append(new);return new
            def evaluate(self,group,art,seeds,out):
                evaluated.append(art['id']);return {'complete':True,'successes':1}
            def test(group,art,gate):
                tested.append((art['id'],gate['gate_id']));return {'reused':False,'result':{'successes':1}}
            with patch('orchestrator.experiment.ROOT',root),patch('orchestrator.experiment.diagnose',diagnose),patch('orchestrator.experiment.UniVTAC.train',train),patch('orchestrator.experiment.UniVTAC.evaluate',evaluate),patch('orchestrator.experiment.test_once',test),patch('orchestrator.experiment.audit',lambda:None):
                gate=group_loop(g)
                self.assertEqual(gate['selected_round'],1)
                self.assertEqual(len(trained),2);self.assertEqual(len(evaluated),3)
                self.assertEqual([r['round'] for r in read(root/'runs/g/rounds.json')],[0,1,2,3])
                resumed=group_loop(g)
                self.assertEqual(gate,resumed);self.assertEqual(len(trained),2);self.assertEqual(len(evaluated),3)
if __name__=='__main__':unittest.main()
