import copy,json,tempfile,unittest,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ssl_common import *
from ssl_data import *
from ssl_neural import *
from ssl_statistics import *
from synthetic_fixture import create

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=create(Path(cls.tmp.name)/'history',users_train=5,users_eval=4,length=13)
        cls.data=Prepared(cls.root/'stage4_event_ssl/prepared_inputs','A_train')
        cls.model,cls.ck=load_model(cls.root/'stage4_event_ssl/models/predictive_state/best_model.pt',cls.data)
        cls.mask,_=alignment_mask(cls.data.manifest);torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def temp(self):return Path(tempfile.mkdtemp(dir=self.tmp.name))
    def test_01_source_hashes(self):
        for n,h in load_json(ROOT/'source_contract.json')['files'].items():self.assertEqual(sha(source_path(n)),h)
    def test_02_prepared_validation(self):self.assertEqual(self.data.validate()['rows'],65)
    def test_03_mask_fourteen_columns(self):self.assertEqual(int(self.mask.sum()),14)
    def test_04_alias_fail_closed(self):
        m=copy.deepcopy(self.data.manifest);m['numeric_input_source_columns'].append('secret_alignment_alias')
        with self.assertRaises(AuditError):alignment_mask(m)
    def test_05_forbidden_coordinate(self):
        m=copy.deepcopy(self.data.manifest);m['numeric_input_source_columns'][0]='M_response_prebalanced_pre'
        with self.assertRaises(AuditError):alignment_mask(m)
    def test_06_feature_order(self):
        m=copy.deepcopy(self.data.manifest);m['numeric_feature_names_after_expansion']=list(reversed(m['numeric_feature_names_after_expansion']))
        with self.assertRaises(AuditError):alignment_mask(m)
    def test_07_chunk_batch_equivalence(self):
        a=predict_native(self.model,self.data,self.temp(),chunk=4,batch_sequences=3)
        b=predict_native(self.model,self.data,self.temp(),chunk=512,batch_sequences=1)
        self.assertTrue(np.allclose(a,b,atol=3e-6))
    def test_08_original_prediction_identity(self):
        p=predict_native(self.model,self.data,self.temp(),chunk=5,batch_sequences=2)
        s,n,_,_=module('evaluate_event_ssl_structure.py').predict_split(self.model,self.data.as_original_arrays(),torch.device('cpu'),5)
        np.testing.assert_allclose(p[:,:2],s,atol=2e-6);np.testing.assert_allclose(p[:,4:6],n,atol=2e-6)
    def test_09_current_interval_causality(self):
        x=torch.tensor(self.data.values('num')[:6].copy())[None];c=torch.tensor(self.data.values('cat')[:6].copy())[None]
        self.model.eval();a=self.model(x,c);xx=x.clone();xx[:,2,:]+=.4;b=self.model(xx,c)
        torch.testing.assert_close(a['state'][:,2],b['state'][:,2],atol=0,rtol=0)
        self.assertGreater(float((a['next_state'][:,2]-b['next_state'][:,2]).abs().max()),1e-7)
    def test_10_same_readout_identity(self):
        p=predict_native(self.model,self.data,self.temp(),chunk=3,batch_sequences=2)
        for s,e in zip(self.data.offsets[:-1],self.data.offsets[1:]):np.testing.assert_allclose(p[s:e-1,2:4],p[s+1:e,:2],atol=3e-6)
    def test_11_proposal_decomposition(self):
        dest=self.temp();p=predict_native(self.model,self.data,dest);decomposition(self.data,p,dest/'d')
        self.assertLess(load_json(dest/'d/decomposition_contract.json')['max_addback_error'],1e-12)
    def test_12_shift_padding(self):
        tr=module('train_event_ssl.py');x=torch.ones(1,5,2);m=torch.tensor([[1,1,1,0,0.]])
        t,v=tr.shifted_future_target(x,m,2);self.assertEqual(v.tolist(),[[1,0,0,0,0]])
    def test_13_clean_input_target_invariance(self):
        self.model.eval();x=torch.randn(2,8,self.data.n_num);cat=torch.zeros(2,8,1,dtype=torch.int64);m=torch.ones(2,8)
        a,_=future_objective(self.model,x,cat,m,self.mask);y=x.clone();y[...,self.mask]=99
        b,_=future_objective(self.model,y,cat,m,self.mask)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    def test_14_clean_target_gradients(self):
        model=copy.deepcopy(self.model);model.zero_grad();x=torch.randn(2,8,self.data.n_num);cat=torch.zeros(2,8,1,dtype=torch.int64)
        loss,_=future_objective(model,x,cat,torch.ones(2,8),self.mask);loss.backward()
        for h in model.future_heads.values():self.assertEqual(float(h.weight.grad[self.mask].abs().max()),0.)
        self.assertIsNone(model.state_head[0].weight.grad);self.assertIsNone(model.delta_head[0].weight.grad)
    def test_15_future_denominator(self):
        tr=module('train_event_ssl.py');x=torch.randn(2,7,self.data.n_num);cat=torch.zeros(2,7,1,dtype=torch.int64);m=torch.ones(2,7)
        self.model.eval();clean=x.clone();clean[...,self.mask]=0;out=self.model(clean,cat)
        expected=[]
        for k,p in out['future'].items():
            t,v=tr.shifted_future_target(clean,m,k);expected.append(tr.masked_mse(p[...,~self.mask],t[...,~self.mask],v))
        loss,_=future_objective(self.model,x,cat,m,self.mask);torch.testing.assert_close(loss,sum(expected)/len(expected))
    def test_16_windows_original(self):
        a=FutureWindows(self.data,8,4,3)
        b=module('train_event_ssl.py').SequenceMemmapDataset(self.data.root,'A_train',8,4,3,False)
        self.assertEqual(a.windows,[(s,e) for s,e,_ in b.windows])
    def test_17_pure_dataset_does_not_load_labels(self):
        d=copy.copy(self.data);original=d.values
        def values(n,i=slice(None)):
            if n in ('X','Y'):raise AssertionError('Labels read in encoder loader')
            return original(n,i)
        d.values=values;b=FutureWindows(d,16,8)[0];self.assertEqual(set(b),{'x','cat','mask'})
    def test_18_paired_heads_init(self):
        h1=initialized_head(self.model,42);h2=initialized_head(self.model,42)
        for a,b in zip(h1.parameters(),h2.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)
    def test_19_H_no_z_dependence(self):
        head=initialized_head(self.model,42).eval();h=torch.randn(10,8);z=torch.randn(10,6);S=torch.zeros(10,2);Y=torch.zeros(10,2)
        _,a=head_loss(head,h,z,S,Y,'H',.5);_,b=head_loss(head,h,z+10,S,Y,'H',.5);torch.testing.assert_close(a,b,atol=0,rtol=0)
    def test_20_HZ_z_sensitive(self):
        head=initialized_head(self.model,42).eval();h=torch.randn(10,8);z=torch.randn(10,6);S=torch.zeros(10,2);Y=torch.zeros(10,2)
        _,a=head_loss(head,h,z,S,Y,'HZ',.5);_,b=head_loss(head,h,z+10,S,Y,'HZ',.5);self.assertGreater(float((a-b).abs().max()),1e-7)
    def test_21_probe_against_sklearn(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        p=self.temp();rng=np.random.default_rng(3);h=rng.normal(size=(60,8)).astype('float32');x=np.tanh(h[:,:2])
        np.save(p/'hb.npy',h);np.save(p/'X.npy',x);g=fit_probe(p,p)
        z=StandardScaler().fit(h.astype(float));r=Ridge(alpha=1).fit(z.transform(h),x)
        np.testing.assert_allclose(probe_predict(h,g),np.clip(r.predict(z.transform(h)),-1,1),atol=1e-6)
    def test_22_grid_endpoints(self):self.assertEqual(cells(np.array([[-1,-1],[1,1],[0,0]] )).tolist(),[0,1599,820])
    def test_23_persistence_correlation_NA(self):
        a=np.ones((1600,2));b=np.zeros((1600,2));W=np.ones(1600);S=np.ones(1600,bool)
        self.assertTrue(np.isnan(field_metrics(a,b,W,S)['vector_r']))
    def test_24_error_identity(self):
        p=predict_native(self.model,self.data,self.temp());X=self.data.values('X').astype(float);Y=self.data.values('Y').astype(float)
        S=p[:,:2].astype(float);T=p[:,2:4].astype(float)
        np.testing.assert_allclose((T-S)-(Y-X),(T-Y)-(S-X),atol=1e-15)
    def test_25_sparse_point_matches_direct(self):
        X=self.data.values('X');Y=self.data.values('Y');cur={'a':X+.01,'b':X-.005};nxt={'a':Y+.02,'b':Y-.003}
        us=UserSufficient(self.data,cur,nxt);v,_=us.summarize(us.point,'fixed')
        _,ui,ct=np.unique(self.data.values('uid'),return_inverse=True,return_counts=True);w=1./ct[ui]
        e=field(X,Y-X,w);p=field(X,nxt['a']-X,w);direct=field_metrics(e['field'],p['field'],e['mass'],e['support'])
        self.assertAlmostEqual(v[('a','endpoint_D2')],direct['D2'],places=9)
        self.assertAlmostEqual(v[('a','next_MSE')],float(np.mean((nxt['a'].astype(float)-Y)**2)),places=12)
    def test_26_interval_incomplete_not_imputed(self):
        r=basic_interval([1.,2.,np.nan],1.,.95);self.assertEqual(r['status'],'not_estimable');self.assertEqual(r['finite_replicates'],2)
    def test_27_interval_raw_not_clipped(self):
        r=basic_interval(np.arange(20)+20.,25.,.95);self.assertGreater(r['basic_high'],1.)
    def test_28_stage_tamper_detected(self):
        d=self.temp();p=stage(d,'a','c',{},lambda q:(q/'x').write_text('ok'))
        (p/'x').write_text('bad')
        with self.assertRaises(AuditError):stage(d,'a','c',{},lambda q:None)
    def test_29_changed_contract_blocked(self):
        d=self.temp();stage(d,'a','first',{},lambda q:None)
        with self.assertRaises(AuditError):stage(d,'a','second',{},lambda q:None)
    def test_30_key_join(self):
        a=np.array([[1,1],[1,3],[2,1]]);idx,ok=key_positions(a,np.array([[1,3],[2,2],[0,1]]))
        self.assertEqual(ok.tolist(),[True,False,False]);self.assertEqual(idx[0],1)
    def test_31_reference_freeze(self):self.assertIsNotNone(discover_reference(self.root))
    def test_32_partition_not_refit(self):
        p=load_partition(self.root);self.assertTrue(p['historical_export_preferred']);self.assertEqual(labels(p['centers'],p).tolist(),list(range(6)))
    def test_33_archive_bridge(self):
        d=Prepared(self.root/'stage4_event_ssl/prepared_inputs','B_confirm');p=self.temp();r=predict_native(self.model,d,p)
        a=archive_bridge(d,self.root/'stage4_event_ssl/evaluation_predictive_state',r,p,self.root/'stage4_event_ssl/models/predictive_state/best_model.pt')
        self.assertEqual(a['status'],'numeric_identity_within_2e-5')
    def test_34_label_gradient_boundary(self):
        tr=module('train_event_ssl.py');m=copy.deepcopy(self.model);m.zero_grad();x=torch.randn(2,8,self.data.n_num);c=torch.zeros(2,8,1,dtype=torch.int64)
        out=m(x,c);loss=tr.masked_huber(out['next_state'],torch.zeros(2,8,2),torch.ones(2,8));loss.backward()
        self.assertGreater(float(m.state_head[0].weight.grad.abs().sum()),0.)
    def test_35_samples_fixed_without_labels(self):
        a=selected_indices(1000,40,42);b=selected_indices(1000,40,42);self.assertTrue(np.array_equal(a,b));self.assertEqual(len(np.unique(a)),40)
    def test_36_no_scheduler(self):
        text=(ROOT/'run_direct.sh').read_text();self.assertNotIn('sbatch',text);self.assertNotIn('srun ',text)
    def test_37_source_unchanged_import(self):
        before={p.name:sha(p) for p in (source_path(n) for n in load_json(ROOT/'source_contract.json')['files'])};module('evaluate_event_ssl_structure.py');after={p.name:sha(p) for p in (source_path(n) for n in load_json(ROOT/'source_contract.json')['files'])};self.assertEqual(before,after)

    def test_38_original_metric_table_identity(self):
        d=Prepared(self.root/'stage4_event_ssl/prepared_inputs','B_confirm');out=self.temp()
        r=predict_native(self.model,d,out);er=self.root/'stage4_event_ssl/evaluation_predictive_state'
        archive_bridge(d,er,r,out,self.root/'stage4_event_ssl/models/predictive_state/best_model.pt')
        old=np.load(out/'historical_S_P.npy');rows=archived_metric_identity(d,er,old,load_partition(self.root),out)
        self.assertTrue(all(x['status'] in ('passed','both_undefined') for x in rows))
    def test_39_historical_metric_change_rejected(self):
        import shutil
        d=Prepared(self.root/'stage4_event_ssl/prepared_inputs','B_confirm');out=self.temp()
        er=out/'evaluation';shutil.copytree(self.root/'stage4_event_ssl/evaluation_predictive_state',er)
        p=er/'tables/stage4_event_ssl_structural_metrics_B_confirm.csv';df=pd.read_csv(p);df['one_step_rmse_M']+=1;df.to_csv(p,index=False)
        a=pd.read_csv(er/'predictions/stage4_event_ssl_predictions_B_confirm.csv.gz')
        old=a[['pred_M','pred_Psi','pred_next_M','pred_next_Psi']].to_numpy('float32')
        with self.assertRaises(AuditError):archived_metric_identity(d,er,old,load_partition(self.root),out)
    def test_40_future_pairs_no_cross_boundary(self):
        out=self.temp();predict_native(self.model,self.data,out,chunk=4,batch_sequences=3,future_common_keep=~self.mask)
        df=pd.read_csv(out/'future_prediction_heldout.csv')
        for k in (1,2,4):
            expected=int(np.maximum(np.diff(self.data.offsets)-k,0).sum())
            self.assertTrue((df.loc[df.horizon==k,'observed_pairs']==expected).all())
    def test_41_future_common_targets_match_manual(self):
        out=self.temp();predict_native(self.model,self.data,out,chunk=4,batch_sequences=3,future_common_keep=~self.mask)
        df=pd.read_csv(out/'future_prediction_heldout.csv');total=0.;count=0
        with torch.inference_mode():
            for a,b in zip(self.data.offsets[:-1],self.data.offsets[1:]):
                x=torch.tensor(self.data.values('num',slice(a,b)).copy())[None]
                c=torch.tensor(self.data.values('cat',slice(a,b)).copy())[None]
                pred=self.model(x,c)['future'][2][0,:-2].numpy().astype(float)
                y=x[0,2:].numpy().astype(float);total+=float(((pred[:,~self.mask]-y[:,~self.mask])**2).sum());count+=len(y)
        actual=df[(df.horizon==2)&(df.target_scope=='common_retained_targets')].iloc[0]
        self.assertAlmostEqual(actual.MSE,total/(count*int((~self.mask).sum())),places=7)
    def test_42_final_receipt_hash_gate(self):
        p=stage(self.temp(),'s','contract',{},lambda d:(d/'model').write_text('frozen'))
        self.assertEqual(verify_committed(p,'contract'),receipt(p));(p/'model').write_text('changed')
        with self.assertRaises(AuditError):verify_committed(p,'contract')
    def test_43_synthetic_target_corruption_rejected(self):
        import shutil
        root=self.temp()/'prepared';shutil.copytree(self.data.root,root)
        a=np.memmap(root/'A_train/y_next.float32.mmap',mode='r+',dtype='float32',shape=(self.data.n,2));a[0,0]+=.1;a.flush()
        with self.assertRaises(AuditError):Prepared(root,'A_train').validate()
    def test_44_bootstrap_repeated_users_matches_explicit(self):
        d=self.data;X=d.values('X');Y=d.values('Y');uid=d.values('uid');N=Y+.01;C=X+.004
        sufficient=UserSufficient(d,{'a':C},{'a':N});mult=np.array([2,0,1,1,1.])
        val,_=sufficient.summarize(np.asarray(sufficient.matrix.T@mult).ravel(),'fixed')
        _,ui,ct=np.unique(uid,return_inverse=True,return_counts=True);w=mult[ui]/ct[ui]
        empirical=field(X,Y-X,w);prediction=field(X,N-X,w)
        fm=field_metrics(empirical['field'],prediction['field'],empirical['mass'],sufficient.fixed)
        self.assertAlmostEqual(val[('a','endpoint_D2')],fm['D2'],places=10)
        se=(N.astype(float)-Y)**2
        self.assertAlmostEqual(val[('a','next_MSE')],float((mult[ui,None]*se).sum()/(2*mult[ui].sum())),places=12)
    def test_45_protocol_scope_frozen(self):
        p=load_json(ROOT/'protocol.json');self.assertEqual(p['s3']['epochs'],8);self.assertEqual(p['new_seeds'],[42,2026,666])
        self.assertEqual(p['bootstrap_replicates'],2000);self.assertNotIn('F10',p['scope'])
        self.assertEqual(p['s3_cpu_seeds'],[42]);self.assertEqual(p['s3_gpu_seeds'],[2026,666,606,37,4669])
        self.assertEqual(p['s3_seeds'],p['s3_cpu_seeds']+p['s3_gpu_seeds'])
        self.assertEqual(p['device'],'CPU preparation and evaluation; S3 seed42 CPU, remaining five seeds GPU')
    def test_46_tensor_digest_stable(self):
        h=initialized_head(self.model,42);self.assertEqual(tensor_digest(h.state_dict()),tensor_digest(copy.deepcopy(h).state_dict()))
    def test_47_empty_mass_weighted_error_not_vector_imputation(self):
        X=self.data.values('X');Y=self.data.values('Y');s=UserSufficient(self.data,{'a':X},{'a':Y+.01})
        v=s.point.copy();cell=np.flatnonzero(v[1600:3200]==0)[0];s.fixed[cell]=True
        r,m=s.summarize(v,'fixed');self.assertEqual(m['empty_support_cells'],1)
        self.assertTrue(np.isfinite(r[('a','endpoint_D2')]))

    def test_48_pure_epoch_resume_exact(self):
        from unittest.mock import patch
        import ssl_neural as ne
        settings={'epochs':2,'seq_len':8,'stride':4,'min_seq_len':3,'warmup_steps':8,'batch_size':4,
                  'lr':2e-4,'weight_decay':1e-4,'future_steps':[1,2,4],'probe_rows':30}
        a=self.temp();b=self.temp()
        ne.train_pure(self.data,self.data,self.ck,42,'clean',self.mask,a,settings,smoke=True)
        original=ne.torch_save
        def interrupt(payload,path):
            original(payload,path)
            if Path(path).name=='last_epoch.pt' and payload.get('epoch')==1:raise RuntimeError('test interruption after atomic epoch')
        with patch.object(ne,'torch_save',side_effect=interrupt):
            with self.assertRaises(RuntimeError):ne.train_pure(self.data,self.data,self.ck,42,'clean',self.mask,b,settings,smoke=True)
        ne.train_pure(self.data,self.data,self.ck,42,'clean',self.mask,b,settings,smoke=True)
        ca=torch_load(a/'fixed_epoch_model.pt');cb=torch_load(b/'fixed_epoch_model.pt')
        self.assertEqual(tensor_digest(ca['model_state_dict']),tensor_digest(cb['model_state_dict']))
        self.assertEqual(ca['history'],cb['history'])

    def test_49_ignore_prior_smoke_reference(self):
        import shutil
        h=self.temp();shutil.copytree(self.root/'four_term_mechanism_revision',h/'four_term_mechanism_revision')
        smoke=h/'four_term_mechanism_revision_smoke';shutil.copytree(h/'four_term_mechanism_revision',smoke)
        save_json({'different':True},smoke/'models/final_models.json')
        r=discover_reference(h);self.assertEqual(Path(r['run']).name,'four_term_mechanism_revision')
    def test_50_unfinished_reference_not_selected(self):
        import shutil
        h=self.temp();r=h/'unfinished';shutil.copytree(self.root/'four_term_mechanism_revision',r)
        (r/'final/COMPLETE.json').unlink();self.assertIsNone(discover_reference(h))

if __name__=='__main__':unittest.main(verbosity=2)
