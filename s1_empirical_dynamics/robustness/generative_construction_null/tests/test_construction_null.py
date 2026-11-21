from __future__ import annotations
import argparse, dataclasses, importlib.util, io, json, os, shlex, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd
from scipy.special import expit
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import cn_common as c
import run_generative_construction_null as run
import summarize_generative_construction_null as summary
import submit_construction_null as submit


def original_repo():
    p=os.environ.get('EDNET_TEST_REPO')
    if p:return Path(p)
    return Path(c.source_paths()['repo'])


def synthetic_fixture(root,train_users=40,val_users=12,confirm_users=12,steps=18):

    root=Path(root);out=root/'outputs_KT4';data=root/'preprocessed_data';ct=data/'contents';ct.mkdir(parents=True)
    questions=pd.DataFrame(dict(question_id=['q1','q2','q3','q4','q5'],bundle_id=['b1','b1','b2','b3','b4'],tags_raw=['1','1;2','2','3','-1'],part=[1]*5,explanation_id=['e1','e1','e2','e3','e4']))
    for name,df in [('questions_clean',questions),('bundles_clean',pd.DataFrame(dict(bundle_id=['b1','b2','b3','b4']))),('lectures_clean',pd.DataFrame(dict(lecture_id=['l1','l2'],tags_raw=['1;2','3']))),('explanations_clean',pd.DataFrame(dict(explanation_id=['e1','e2','e3','e4'],tags_raw=['1;2','2','3','-1'])))]:df.to_csv(ct/(name+'.csv'),index=False)
    sources=c.source_paths(str(original_repo()))
    provisional=dict(sources=sources,data=dict(data_root=str(data),kt4_root=str(data/'kt4'),contents_root=str(ct),metadata_root=str(data/'metadata')),output_root=str(out))
    m=c.source_module(provisional);m.ensure_dirs()
    sets={};start=1
    for name,n in zip(c.SPLITS,[train_users,val_users,confirm_users]):
        sets[name]=list(range(start,start+n));start+=n
        pd.DataFrame(dict(user_id=sets[name])).to_csv(m.SPLIT_ROOT/(name+'_users.csv'),index=False)
    content=m.ContentsBuilder().build();qtab=pd.DataFrame(dict(question_id=['q1','q2','q3','q4','q5'],prior_correct_itemEB_Atrain=[.8,.63,.47,.9,.55]))
    ttab=pd.DataFrame(dict(tag=[1,2,3],prior_correct_Atrain_shrunk=[.7,.55,.8],global_prior_correct_Atrain=[.66]*3))
    qtab.to_csv(m.META_OUT_ROOT/'A_train_question_itemEB_correctness_priors.csv',index=False);ttab.to_csv(m.META_OUT_ROOT/'A_train_tag_correctness_priors.csv',index=False)
    priors=m.TagPriorEstimate(np.array([.7,.55,.8]),ttab,.66,dict(zip(qtab.question_id,qtab.prior_correct_itemEB_Atrain)),qtab)
    builder=m.UserEmpiricalDynamicsBuilder(content,priors)
    b_all=[];q_all=[];s_all=[];rng=np.random.default_rng(103)
    splitmap={u:s for s,us in sets.items() for u in us}
    for uid in splitmap:
        timestamp=1_600_000_000_000+uid*10000
        for k in range(steps):
            bundle=['b1','b2','b3','b4'][k%4];qids={'b1':['q1','q2'],'b2':['q3'],'b3':['q4'],'b4':['q5']}[bundle]
            pre=timestamp;sub=pre+90000
            br={x:0 for x in c.B_REQ+c.B_OPT};br.update(user_id=uid,bundle_attempt_index=k+1,bundle_id=bundle,enter_ts=pre if k%11 else np.nan,submit_ts=sub,source_code=1,platform_code=1,accuracy=.5,total_response_count=len(qids),answered_question_count=len(qids),duration_ms=90000.)
            b_all.append(br)
            for j,qid in enumerate(qids):q_all.append(dict(user_id=uid,bundle_attempt_index=k+1,question_id=qid,is_correct=np.nan if (k+uid)%29==0 else float(rng.random()<priors.question_priors[qid])))
            if k%3==1:
                sr={x:np.nan for x in c.S_REQ+c.S_OPT};sr.update(user_id=uid,study_episode_index=k+1,item_id='e1' if k%2 else 'l1',item_type_code=2 if k%2 else 4,enter_ts=sub+1000,quit_ts=sub+61000,dwell_ms=60000.,related_bundle_id=bundle,media_event_count=0,media_pair_anomaly_count=0)
                s_all.append(sr)
            timestamp+=int((.1+(uid%7)/5)*m.DAY_MS)
    b=pd.DataFrame(b_all);q=pd.DataFrame(q_all);s=pd.DataFrame(s_all)
    frames={name:[] for name in sets}
    for ci,keep in enumerate([set(u for u in splitmap if u%2==0),set(u for u in splitmap if u%2==1)]):
        bb=b[b.user_id.isin(keep)];qq=q[q.user_id.isin(keep)];ss=s[s.user_id.isin(keep)]
        for name,df in [('bundle_attempts',bb),('question_attempts',qq),('study_episodes',ss)]:
            d=data/'kt4'/name;d.mkdir(parents=True,exist_ok=True);df.to_csv(d/f'{name}_chunk_{ci:03d}.csv.gz',index=False,compression='gzip')
        raw=[]
        for uid,bu in bb.groupby('user_id',sort=False):raw.extend(builder.build_user(int(uid),bu,qq[qq.user_id==uid],ss[ss.user_id==uid],splitmap[uid]))
        raw=pd.DataFrame(raw);raw.to_csv(m.RAW_ROOT/f'raw_panel_chunk_{ci:03d}.csv.gz',index=False,compression='gzip')
        fin=m.finalize_panel_chunk(raw)
        for sp in frames:frames[sp].append(fin[fin.split==sp])
    sem=c.semantic_module(provisional)
    for sp,parts in frames.items():
        fin=pd.concat(parts,ignore_index=True);fin.to_csv(m.DYN_ROOT/f'student_dynamics_panel_core_{sp}.csv.gz',index=False,compression='gzip')

        archived=pd.read_csv(m.DYN_ROOT/f'student_dynamics_panel_core_{sp}.csv.gz')
        primary=m.downcast_frame(archived)
        ph,_=sem.add_nonsemantic_coordinate(archived);ph=m.downcast_frame(ph)
        for coord,fr,spec in [('psi',primary,m.coordinate_specs()[0]),('phi',ph,sem.control_spec(m))]:
            f=m.field_stats_from_dict(m.occupancy_drift_stats(fr,spec));table=m.field_grid_table(f,sp)
            dst=c.archived_field_base(out,sp,coord);dst.parent.mkdir(parents=True,exist_ok=True);table.to_csv(Path(str(dst)+'.csv'),index=False)
    for name,rr in [('MR_PsiA',out/'stage1'),('MR_PhiAI',out/'semantic_specificity_control/coordinate/stage1')]:
        dr=rr/'dynamics/candidate_regions'/name;dr.mkdir(parents=True,exist_ok=True)
        core=np.zeros((40,40),bool);core[19:21,8:10]=True;np.save(dr/'A_train_primary_convergence_core_mask.npy',core)
        c.atomic_json(dr/'training_convergence_thresholds.json',dict(shell_radius=.35))
    c.atomic_json(m.META_OUT_ROOT/'bundle_chunk_user_completeness_audit.json',dict(gate_passed=True))
    c.atomic_json(m.SPLIT_ROOT/'split_manifest.json',dict(random_state=42,sizes={k:len(v) for k,v in sets.items()}))
    c.atomic_json(m.META_OUT_ROOT/'stage1_empirical_v3_manifest.json',dict(input_preprocess_manifest=dict(output_root=str(data)),synthetic_fixture=True))
    return out,data,provisional,frames


class MathematicsTests(unittest.TestCase):
    def test_r2_bias_identity(self):
        p=.8;J=np.array([1-p,-p]);A=np.abs(J);pr=np.array([p,1-p]);Z=J/A
        self.assertAlmostEqual(np.dot(pr,J),0.,places=14)
        self.assertAlmostEqual(np.dot(pr,A)*np.dot(pr,Z),.192,places=14)
        np.testing.assert_allclose(A*Z,J)
    def test_null_b_calibration_identical_distribution(self):
        pg=np.array([.001,.05,.3,.8,.999]);effects=np.linspace(-1.8,1.8,256)
        alpha=c.calibrated_intercepts(pg,effects)
        np.testing.assert_allclose(expit(alpha[:,None]+effects).mean(axis=1),pg,atol=1e-12,rtol=0)
    def test_rng_scheduling_invariant(self):
        a={u:c.stable_rng(42,'A_val','null_a',4,u).random(10) for u in [11,20,5]}
        b={u:c.stable_rng(42,'A_val','null_a',4,u).random(10) for u in [5,11,20]}
        for u in a:np.testing.assert_array_equal(a[u],b[u])
        self.assertFalse(np.array_equal(a[5],c.stable_rng(42,'B_confirm','null_a',4,5).random(10)))
    def test_missingness_and_no_truth_dependence(self):
        q=pd.DataFrame(dict(question_id=['a','b','c'],is_correct=[0.,np.nan,1.]));p=np.array([.2,.5,.8])
        a,_=c.simulate_questions(q,p,42,'A_val','null_a',0,7)
        q.loc[[0,2],'is_correct']=[1.,0.]
        b,_=c.simulate_questions(q,p,42,'A_val','null_a',0,7)
        np.testing.assert_array_equal(a.is_correct.to_numpy(),b.is_correct.to_numpy())
        self.assertTrue(np.isnan(a.is_correct.iloc[1]))
    def test_symmetric_score_permutation_equivariance(self):
        r=np.random.default_rng(4);u=r.normal(size=(20,4,4));v=r.normal(size=u.shape);w=r.uniform(.2,1,size=u.shape);mask=np.ones((4,4),bool)
        scores=summary.symmetric_distances(u,v,w,mask);order=r.permutation(20)
        np.testing.assert_allclose(summary.symmetric_distances(u[order],v[order],w[order],mask),scores[order],atol=1e-14)
    def test_rank_never_zero_and_ties(self):
        self.assertEqual(summary.conservative_rank_p([100.,1.,2.]),1/3)
        self.assertEqual(summary.conservative_rank_p([1.,1.,1.]),1.)
        self.assertIsNone(summary.conservative_rank_p([1.,np.nan,2.]))
    def test_bh_preserves_three_metric_family(self):
        q=summary.bh_three([.01,None,.04]);self.assertAlmostEqual(q[0],.03);self.assertIsNone(q[1]);self.assertAlmostEqual(q[2],.06)
    def test_unsupported_is_not_zero(self):
        v=summary.symmetric_distances(np.zeros((5,2,2)),np.zeros((5,2,2)),np.ones((5,2,2)),np.zeros((2,2),bool))
        self.assertTrue(np.isnan(v).all())
    def test_null_a_distribution_not_filtered(self):
        q=pd.DataFrame(dict(question_id=['a']*20000,is_correct=np.zeros(20000)))
        out,ca=c.simulate_questions(q,np.full(20000,.8),42,'A_val','null_a',2,8)
        self.assertLess(abs(out.is_correct.mean()-.8),.015);self.assertEqual(ca[0],20000)


class SourceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='cn_source_fixture_');cls.root=Path(cls.temp.name)
        cls.out,cls.data,cls.provisional,cls.frames=synthetic_fixture(cls.root)
        cls.m=c.source_module(cls.provisional);cls.sem=c.semantic_module(cls.provisional)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_manifest_root_discovery(self):
        d=c.resolve_data(self.out);self.assertEqual(Path(d['data_root']),self.data)
        self.assertEqual(len(d['bundle']),2)
    def test_full_user_block_field_equivalence(self):
        fr=pd.concat(self.frames['A_val'],ignore_index=True);fr=self.m.downcast_frame(fr)
        spec=self.m.coordinate_specs()[0];acc=c.FieldAccumulator(self.m,spec)
        users=sorted(fr.user_id.unique());acc.add(fr[fr.user_id.isin(users[::2])]);acc.add(fr[fr.user_id.isin(users[1::2])])
        got=acc.finish();ref=self.m.occupancy_drift_stats(fr,spec)
        for a,b in [('u','drift_u'),('v','drift_v'),('occ','occupancy_weighted'),('n','drift_count'),('w','drift_weight')]:np.testing.assert_allclose(got[a],ref[b],atol=1e-13,rtol=1e-13)
        np.testing.assert_array_equal(got['drift_mask'],ref['drift_mask']);np.testing.assert_array_equal(got['state_mask'],ref['state_mask'])
    def test_fallback_prior_matches_source(self):
        _,ct,pr=c.frozen_inputs(self.m);b=self.m.UserEmpiricalDynamicsBuilder(ct,pr)
        del b.question_priors['q2']
        q=pd.DataFrame(dict(question_id=['q2','unknown']))
        got=c.probabilities(q,ct,b)
        self.assertAlmostEqual(got[0],np.mean(pr.priors[ct.question_tag_idx['q2']]))
        self.assertAlmostEqual(got[1],pr.global_prior)
    def test_foreign_output_write_refused(self):
        with self.assertRaises(ValueError):run.choose_work(self.out,self.out/'stage1/bad')
    def test_actual_smoke_workflow_source_replay(self):
        args=argparse.Namespace(output_root=str(self.out),work_root=None,repo_root=str(original_repo()),data_root=None,smoke=True,replicates=3,seed=42,smoke_users=12)
        work=run.initialize(args);self.assertEqual(run.initialize(args),work);run.fit_null_b(work);run.prepare_split(work,'A_val')
        for mode in ['null_a','null_b']:
            for rid in range(3):run.simulate(work,'A_val',mode,rid)
        run.smoke_finish(work)
        self.assertTrue(c.read_json(work/'SMOKE_OK.json')['passed'])

        audit=c.read_json(work/'observed/A_val_identity.json')
        self.assertTrue(audit['passed']);self.assertGreater(len(audit['source_replay']),0)

        f=work/'simulations/A_val/null_a/replicate_0000.npz';h=c.sha256(f)
        run.simulate(work,'A_val','null_a',0);self.assertEqual(c.sha256(f),h)
        with self.assertRaises(RuntimeError):run.prepare_split(work,'B_confirm')
        with self.assertRaises(RuntimeError):summary.summarize(work,'A_val','null_a')
        p=c.load_protocol(work)


        for mode in ['null_a','null_b']:
            arrays,_=summary.read_realizations(work,p,'A_val',mode)
            self.assertEqual(len(arrays),4)
            for coord in ['psi','phi']:
                target=work/f'synthetic_summary_test_{mode}_{coord}';target.mkdir()
                result=summary.analyze_coordinate(work,p,'A_val',mode,coord,arrays,target)
                self.assertEqual(len(result['tests']),4)
                self.assertTrue((target/f'{coord}_observed_field.png').exists())
                self.assertTrue((target/f'{coord}_scope_coverage.csv').exists())
                self.assertEqual(len(pd.read_csv(target/f'{coord}_realization_statistics.csv')),4)
                for row in result['tests']:
                    if row['status']=='estimable':self.assertTrue(0<row['p_mc']<=1)
                    else:self.assertIsNone(row['p_mc'])

        with f.open('ab') as handle:handle.write(b'tamper')
        with self.assertRaises(RuntimeError):run.simulate(work,'A_val','null_a',0)
    def test_joint_demand_and_alignment_rebuilt(self):
        users,ct,pr=c.frozen_inputs(self.m);builder=self.m.UserEmpiricalDynamicsBuilder(ct,pr)
        data=c.resolve_data(self.out);p={'data':data}
        uid=min(users['A_val']);ci=0 if uid%2==0 else 1
        b,qg,sg=run.raw_chunk(self.m,p,ci,{uid});q=qg[uid].copy()
        q['is_correct']=1.;positive=pd.DataFrame(builder.build_user(uid,b,q,sg.get(uid,pd.DataFrame()),'A_val'))
        q['is_correct']=0.;negative=pd.DataFrame(builder.build_user(uid,b,q,sg.get(uid,pd.DataFrame()),'A_val'))
        self.assertTrue((positive.unresolved_response_demand_mass_pre.fillna(0)==0).all())
        self.assertGreater(negative.unresolved_response_demand_mass_pre.max(),0)
        self.assertFalse(np.allclose(positive.activity_alignment_order_Psi_pre,negative.activity_alignment_order_Psi_pre,equal_nan=True))
        np.testing.assert_allclose(positive.activity_active_mass_pre,negative.activity_active_mass_pre,atol=1e-12)
        np.testing.assert_allclose(positive.activity_idle_mass_pre,negative.activity_idle_mass_pre,atol=1e-12)
    def test_complete_support_no_replica_selection(self):
        rng=np.random.default_rng(23);n=20;shape=(n,40,40)
        X,Y=np.meshgrid(np.linspace(-.975,.975,40),np.linspace(-.975,.975,40),indexing='ij')
        u=-.05*X[None]+rng.normal(0,.002,shape);v=-.05*Y[None]+rng.normal(0,.002,shape)
        occ=np.ones(shape);w=occ.copy();mask=np.ones((40,40),bool);core=np.zeros_like(mask);core[19:21,19:21]=True
        bank,geom,stencil=summary.metric_bank(u,v,occ,w,mask,mask,core,.35)
        self.assertEqual(geom['interior_cells'],38*38)
        self.assertTrue(np.isfinite(bank['flow_core_to_shell_speed_ratio']).all())
        scores=summary.symmetric_distances(u,v,occ,mask)
        ranks=np.array([summary.conservative_rank_p(scores,index=i) for i in range(n)])
        self.assertEqual(int((ranks<=.05).sum()),1)
    def test_archived_field_audit_after_csv_replay(self):
        raw=pd.read_csv(self.out/'stage1/dynamics/student_dynamics_panel_core_A_val.csv.gz')
        a=c.FieldAccumulator(self.m,self.m.coordinate_specs()[0]);a.add(self.m.downcast_frame(raw))
        result=c.audit_field(a.finish(),c.archived_field_base(self.out,'A_val','psi'))
        self.assertTrue(result['passed'])
    def test_slurm_plan_no_gpu(self):
        a=argparse.Namespace(phase='smoke',output_root=str(self.out),work_root=None,repo_root=str(original_repo()),data_root=None,python=sys.executable,seed=42,smoke_users=12,array_size=999,replicates_per_task=8,mem='16G',prepare_mem='64G',dry_run=True)
        _,jobs=submit.build_jobs(a)
        self.assertEqual([j['name'] for j in jobs],['init_calibrate','prepare_A_val','A_val_null_a_0000','A_val_null_b_0000','smoke_finish'])


    def test_formal_array_batches_cover_each_id_once(self):
        a=argparse.Namespace(phase='all',output_root=str(self.out),work_root=None,repo_root=str(original_repo()),data_root=None,python=sys.executable,seed=42,smoke_users=12,array_size=30,replicates_per_task=8,mem='16G',prepare_mem='64G',dry_run=True)
        _,jobs=submit.build_jobs(a)
        chosen=[j for j in jobs if j['name'].startswith('A_val_null_a_')]
        ids=[]
        for job in chosen:
            ar=job['array']
            for idx in range(ar['count']):
                first=ar['start']+idx*ar['batch'];ids.extend(range(first,min(first+ar['batch'],ar['total'])))
        self.assertEqual(ids,list(range(999)))


    def test_basin_metrics_match_original_functions(self):
        rng=np.random.default_rng(209)
        base=self.m.field_stats_from_dict(self.m.occupancy_drift_stats(self.m.downcast_frame(pd.concat(self.frames['A_val'],ignore_index=True)),self.m.coordinate_specs()[0]))
        u=rng.normal(0,.03,(40,40));v=rng.normal(0,.03,(40,40))
        occ=rng.uniform(.1,5,(40,40));w=rng.uniform(.2,8,(40,40))
        dm=rng.random((40,40))>.1;sm=rng.random((40,40))>.07
        core=np.zeros((40,40),bool);core[16:23,14:21]=True
        f=dataclasses.replace(base,drift_u=u,drift_v=v,occupancy_weighted=occ,occupancy_probability=occ/occ.sum(),drift_weight=w,state_mask=sm,drift_mask=dm)
        bank,geom,stencil=summary.metric_bank(u[None],v[None],occ[None],w[None],sm,dm,core,.35)
        con=self.m.global_field_contraction_summary(f,'test')
        region,_=self.m.evaluate_frozen_convergence_region(f,core,'test',.35)
        for key,ref in [('negative_divergence_occupancy_fraction',con['weighted_negative_divergence_fraction_interior_only']),('flow_weighted_shell_fraction_inward',region.iloc[0]['flow_weighted_shell_fraction_inward']),('flow_core_to_shell_speed_ratio',region.iloc[0]['flow_core_to_shell_speed_ratio'])]:
            self.assertAlmostEqual(float(bank[key][0]),float(ref),places=12)
        _,ref_stencil=self.m.interior_divergence(f);np.testing.assert_array_equal(stencil,ref_stencil)

    def test_generated_slurm_shell_is_valid_bash(self):
        a=argparse.Namespace(phase='all',output_root=str(self.out),work_root=None,repo_root=str(original_repo()),data_root=None,python=sys.executable,seed=42,smoke_users=12,array_size=30,replicates_per_task=8,mem='16G',prepare_mem='64G',dry_run=True,concurrency=8,time='24:00:00',partition='',account='')
        buffer=io.StringIO()
        with redirect_stdout(buffer):submit.submit(a)
        count=0
        for line in buffer.getvalue().splitlines():
            if not line.startswith('sbatch '):continue

            count+=1

        a.phase='smoke';a.dry_run=False
        calls=[]
        def record(command,**kwargs):
            if command[0]=='sbatch':
                shell=shlex.split(command[command.index('--wrap')+1]);self.assertEqual(shell[:2],['bash','-c'])
                check=subprocess.run(['bash','-n','-c',shell[2]],capture_output=True,text=True)
                self.assertEqual(check.returncode,0,check.stderr);calls.append(command)
                return subprocess.CompletedProcess(command,0,stdout=str(100+len(calls))+'\n',stderr='')
            raise AssertionError(command)
        real_run=subprocess.run
        def record_safe(command,**kwargs):
            if command[:2]==['bash','-n']:return real_run(command,**kwargs)
            return record(command,**kwargs)
        real_which=submit.shutil.which
        with mock.patch.object(submit.shutil,'which',side_effect=lambda x:'/usr/bin/sbatch' if x=='sbatch' else real_which(x)), mock.patch.object(submit.subprocess,'run',side_effect=record_safe):
            with redirect_stdout(io.StringIO()):submit.submit(a)
        self.assertEqual(len(calls),5)
        self.assertTrue(all(not any('--gres' in v or '--gpus' in v for v in command) for command in calls))


    def test_evidence_assembly_does_not_require_positive_findings(self):
        work=self.root/'synthetic_report_assembly';work.mkdir(exist_ok=True)
        p=dict(protocol_id='synthetic-report-test',smoke=False,modes=['null_a','null_b'],coordinates=['psi','phi'],replicates=999,seed=42,output_root=str(self.out))
        fixture_spec=importlib.util.spec_from_file_location('assembly_fixture',HERE/'tests/test_execution_summary.py')
        fixture=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixture)
        for split in ('A_val','B_confirm'):
            for mode in p['modes']:
                fixture.write_summary_fixture(work,p,split,mode,estimable=True)


        with mock.patch.object(summary,'load_protocol',return_value=p), mock.patch.object(summary,'read_realizations',return_value=([],[])):
            summary.release_validation(work);self.assertTrue((work/'A_val_release.json').exists())
            summary.final_report(work)
        table=pd.read_csv(work/'final/construction_null_statistics.csv')
        self.assertEqual(len(table),32);self.assertTrue((table.p_mc==1).all())
        self.assertTrue(c.read_json(work/'final/COMPLETE.json')['technical_complete'])
        c.require_confirm(work,p)
        a=work/'summary/A_val/null_a/summary.json';a.write_text(a.read_text()+'\n')
        with self.assertRaises(RuntimeError):c.require_confirm(work,p)

if __name__=='__main__':unittest.main(verbosity=2)
