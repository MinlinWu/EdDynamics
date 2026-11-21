
import ast
import copy
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ssl_common as common
import numpy as np


def contract_functions():
    tree=ast.parse((common.ROOT/'ssl_pipeline.py').read_text(encoding='utf-8'))
    names={'reference_paths','resolved_source_selection','reference_stage_input','s0'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==names
    scope=dict(vars(common))
    scope.update(np=np,existing_results=lambda history:[],alignment_mask=lambda manifest:([],[]))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'ssl_pipeline_contract_functions','exec'),scope)
    return scope


class SourceResumeContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.fn=contract_functions();self.history=self.root/'history';self.history.mkdir()
        self.prepared=self.history/'prepared'
        self.put(self.prepared/'metadata/input.json','{}')
        for split in common.SPLITS:
            self.put(self.prepared/split/'x.mmap','fixed array')
            self.put(self.prepared/split/'sequence_offsets.npy','fixed offsets')
        self.entry={'input_root':str(self.prepared)}
        for kind in ('full','shuffled'):
            checkpoint=self.put(self.history/(kind+'.pt'),'frozen '+kind)
            ev=self.history/(kind+'_evaluation')
            self.put(ev/'metadata/evaluation.json','{}')
            for split in ('A_val','B_confirm'):
                self.put(ev/'predictions'/('stage4_event_ssl_predictions_'+split+'.csv'),'key,pred\n1,0\n')
                self.put(ev/'tables'/('stage4_event_ssl_structural_metrics_'+split+'.csv'),'metric\n0\n')
            self.entry.update({kind:str(checkpoint),kind+'_evaluation':str(ev)})
        self.partition=self.put(self.history/'partition.json','frozen partition')
        self.ref=self.make_reference(self.root/'reference_a')

    def tearDown(self):self.temp.cleanup()

    def put(self,path,text):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8');return path

    def make_reference(self,root):
        files={
            'rows':self.put(root/'confirmation/row_predictions.csv','row,pred\n1,0\n'),
            'models':self.put(root/'models/final_models.json','{"model":1}'),
            'freeze':self.put(root/'models/FIT_FROZEN.json','{"frozen":true}'),
            'complete':self.put(root/'final/COMPLETE.json','{"complete":true}'),
        }
        return {'run':str(root),'rows':str(files['rows']),
                **{k:common.file_record(files[k]) for k in ('models','freeze','complete')}}

    def context(self,ref=None):
        ctx=SimpleNamespace(history=self.history,out=self.history/common.run_name(True),
            paths={'42':self.entry.copy()},part={'source_files':[str(self.partition)]},
            ref=copy.deepcopy(self.ref if ref is None else ref),code='fixed-test-code',
            smoke=True,protocol={'version':common.VERSION},threads=1,loader_workers=0,
            infer_batch=2,chunk=128,seeds=(42,),newseeds=(42,),modelinfo=[])
        def data(seed,split):
            uid=common.SPLITS.index(split)+1
            return SimpleNamespace(manifest={},validate=lambda:{'rows':1},
                                   values=lambda key:np.array([uid]))
        ctx.data=data
        ctx.verify_inputs=lambda:[common.verify_record(r) for r in ctx.records]
        return ctx

    def commit(self,ctx=None):
        ctx=self.context() if ctx is None else ctx
        self.fn['s0'](ctx)
        return ctx

    def test_01_same_sources_resume_without_rewriting_receipt(self):
        a=self.commit();p=a.out/'S0/_SUCCESS.json';before=p.read_bytes()
        b=self.commit();self.assertEqual(a.signature,b.signature)
        self.assertEqual(before,p.read_bytes())

    def test_02_identical_content_different_reference_path_rejected(self):
        self.commit();other=self.root/'reference_b';shutil.copytree(Path(self.ref['run']),other)
        b=copy.deepcopy(self.ref);b['run']=str(other)
        b['rows']=str(other/'confirmation/row_predictions.csv')
        for k,rel in [('models','models/final_models.json'),('freeze','models/FIT_FROZEN.json'),('complete','final/COMPLETE.json')]:
            b[k]=common.file_record(other/rel)
        with self.assertRaisesRegex(common.AuditError,'Stage identity changed'):
            self.commit(self.context(b))

    def test_03_each_reference_artifact_change_rejected_even_with_same_mtime(self):
        self.commit()
        for role in ('rows','models','freeze','complete'):
            p=Path(self.ref['rows'] if role=='rows' else self.ref[role]['path'])
            original=p.read_bytes();st=p.stat()
            try:
                replacement=bytes([original[0]^1])+original[1:]
                p.write_bytes(replacement);os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns))
                with self.subTest(role=role),self.assertRaisesRegex(common.AuditError,'Stage identity changed'):
                    self.commit()
            finally:
                p.write_bytes(original);os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns))

    def test_04_mtime_only_change_is_legal_resume(self):
        a=self.commit();p=Path(self.ref['rows']);st=p.stat()
        os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns+1_000_000_000))
        b=self.commit();self.assertEqual(a.signature,b.signature)

    def test_05_current_source_set_addition_is_not_ignored(self):
        self.commit();self.put(self.prepared/'metadata/new_source.json','{}')
        with self.assertRaisesRegex(common.AuditError,'Stage identity changed'):self.commit()

    def test_06_missing_current_reference_fails_before_reuse(self):
        self.commit();Path(self.ref['complete']['path']).unlink()
        with self.assertRaisesRegex(common.AuditError,'Resolved source missing'):self.commit()

    def test_07_bridge_dependency_has_paths_and_hashes_for_all_roles(self):
        ctx=self.commit();dependency=self.fn['reference_stage_input'](ctx)
        for role in ('rows','models','freeze','complete'):
            self.assertEqual(set(dependency[role]),{'path','bytes','sha256'})
            self.assertEqual(common.sha(dependency[role]['path']),dependency[role]['sha256'])
        ctx.ref['run']=str(self.root/'switched')
        with self.assertRaisesRegex(common.AuditError,'reference changed after S0'):
            self.fn['reference_stage_input'](ctx)

    def test_08_partial_stage_same_identity_can_resume_changed_identity_cannot(self):
        root=self.root/'partial_run'
        def interrupted(p):
            (p/'epoch.txt').write_text('saved epoch');raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):common.stage(root,'stage','a',{'source':'one'},interrupted)
        with self.assertRaisesRegex(common.AuditError,'Partial stage contract changed'):
            common.stage(root,'stage','b',{'source':'two'},lambda p:None)
        def resume(p):self.assertEqual((p/'epoch.txt').read_text(),'saved epoch')
        common.stage(root,'stage','a',{'source':'one'},resume)
        self.assertTrue((root/'stage/_SUCCESS.json').exists())

    def test_09_v2_namespace_and_default_shell_smoke(self):
        self.assertEqual(common.VERSION,'ssl-information-controls-v1')
        self.assertEqual(common.run_name(), 'ssl_information_controls')
        self.assertEqual(common.run_name(True),'ssl_information_controls_smoke')
        self.assertIn('MODE="${1:-smoke}"',(common.ROOT/'run_direct.sh').read_text())


if __name__=='__main__':unittest.main(verbosity=2)
