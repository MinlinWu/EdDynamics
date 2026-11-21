
from pathlib import Path
import sys, json, shutil
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ssl_common import *
from ssl_data import ALIGNMENT_SOURCES, Prepared
from ssl_neural import make_model


def create(root,users_train=200,users_eval=40,length=10):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    save_json({'synthetic':True,'not_EdNet':True,'purpose':'implementation integration tests'},root/'_SYNTHETIC_FIXTURE.json')
    prep=module('prepare_event_ssl_inputs.py');tr=module('train_event_ssl.py');ev=module('evaluate_event_ssl_structure.py')
    p=root/'stage1/dynamics/fixed_k6_mesostates';p.mkdir(parents=True)
    raw=np.array([[-.9,.02],[-.7,-.67],[-.3,-.44],[.1,-.71],[.17,-.13],[.96,0.]])
    meta={'coordinate':'MR_PsiA','macrostate_k':6,'macrostate_k_rule':'fixed a priori',
      'features':[prep.COL_M,prep.COL_PSI],'fit_split':'A_train','user_balanced_sampling':True,
      'user_balanced_kmeans_fit':True,'fit_max_rows':500000,'kmeans_n_init':20,'random_state':42,
      'raw_to_ordered_label':{str(i):i for i in range(6)},'scaler_mean':[0.,0.],'scaler_scale':[1.,1.]}
    save_json(meta,p/'fixed_k6_model_metadata.json')
    centers=pd.DataFrame({'macrostate':np.arange(6),'center_M':raw[:,0],'center_Psi':raw[:,1],
          'center_M_standardized':raw[:,0],'center_Psi_standardized':raw[:,1]});centers.to_csv(p/'fixed_k6_centers.csv',index=False)
    fixed=prep.fixed_k6_contract(root/'stage1')
    numeric=list(ALIGNMENT_SOURCES)+['next_gap_days','current_accuracy_diagnostic_only','response_active_mass_interval']
    names=[]
    for col in numeric:
        names.append(col)
        if any(t in col for t in prep.LOG1P_NUMERIC_TOKENS):names.append('log1p_'+col)
    nnum=len(names);ncat=1
    inputroot=root/'stage4_event_ssl/prepared_inputs';(inputroot/'metadata').mkdir(parents=True)
    summaries={}
    for si,split in enumerate(SPLITS):
        ucount=users_train if split=='A_train' else users_eval;n=ucount*length
        rng=np.random.default_rng(10+si);uids=np.repeat(np.arange(ucount)+si*100000+1,length).astype('int64')
        step=np.tile(np.arange(length),ucount).astype('int64')

        states=np.empty((ucount,length+1,2),np.float32)
        states[:,:,0]=.12+np.cumsum(rng.normal(0,.0008,(ucount,length+1)),axis=1)
        states[:,:,1]=-.63+np.cumsum(rng.normal(0,.0007,(ucount,length+1)),axis=1)
        X=states[:,:-1].reshape(n,2);Y=states[:,1:].reshape(n,2)
        num=rng.normal(0,.3,(n,nnum)).astype('float32');cat=rng.integers(0,31,(n,ncat),dtype='int64')
        d=inputroot/split;d.mkdir(parents=True)
        for name,a in [('x_num.float32.mmap',num),('x_cat.int64.mmap',cat),('y_current.float32.mmap',X),
                       ('y_next.float32.mmap',Y),('user_id.int64.mmap',uids),('bundle_step_index.int64.mmap',step)]:
            m=np.memmap(d/name,mode='w+',dtype=a.dtype,shape=a.shape);m[:]=a;m.flush()
        offsets=prep.sequence_offsets(uids,step);np.save(d/'sequence_offsets.npy',offsets)
        summaries[split]={'rows':n,'numeric_shape':[n,nnum],'categorical_shape':[n,ncat],'users':ucount,
           'sequence_count':ucount,'target_shape':[n,2]}
    normal={'fitted_on':'A_train only','numeric_feature_names':names,'mean':[0.]*nnum,'std':[1.]*nnum,'finite_count':[users_train*length]*nnum}
    save_json(normal,inputroot/'metadata/normalizer.json')
    manifest={'primary_coordinates':['M','Psi'],'numeric_input_source_columns':numeric,
       'numeric_feature_names_after_expansion':names,'categorical_input_source_columns':['bundle_id'],
       'categorical_hash_buckets':32,'stage1_fixed_k6_contract':fixed,'split_summaries':summaries,
       'stage1_root':str(root/'stage1'),'sequence_boundary_policy':'user_id change or non-contiguous bundle_step_index'}
    save_json(manifest,inputroot/'metadata/stage4_input_manifest.json')
    cfg={'hidden_dim':8,'input_dim':6,'num_layers':2,'dropout':.1,'categorical_emb_dim':3,
      'future_steps':[1,2,4],'delta_scale':.5,'epochs':8,'seq_len':256,'stride':128,'min_seq_len':3,
      'warmup_steps':8,'batch_size':192,'lr':2e-4,'weight_decay':1e-4,'amp_dtype':'bf16',
      'lambda_future':1.,'lambda_state':.5,'lambda_closure':.5,'model_kind':'predictive_state'}
    shapes={'n_num':nnum,'n_cat':ncat,'hash_buckets':32}
    torch.set_num_threads(1)
    for seed in ALL_SEEDS:
        base=root if seed==42 else root/f'random_seed_experiments/seed_{seed}'
        if seed!=42:
            dst=base/'stage4_event_ssl/prepared_inputs';dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(inputroot,dst)
        for kind in ('full','shuffled'):
            modeldir=base/('stage4_event_ssl/models/predictive_state' if kind=='full' else 'stage4_event_ssl_time_shuffle_control/model')
            evaldir=base/('stage4_event_ssl/evaluation_predictive_state' if kind=='full' else 'stage4_event_ssl_time_shuffle_control/evaluation_on_ordered_inputs')
            torch.manual_seed(seed+(0 if kind=='full' else 200));cc=dict(cfg,seed=seed)
            model=make_model(cc,shapes).eval();modeldir.mkdir(parents=True)
            cp=modeldir/'best_model.pt';torch_save({'config':cc,'model_shapes':shapes,'model_state_dict':model.state_dict(),'best_epoch':4},cp)
            (evaldir/'metadata').mkdir(parents=True);(evaldir/'predictions').mkdir();(evaldir/'tables').mkdir()
            centers.to_csv(evaldir/'tables/stage4_event_ssl_fixed_k6_macrostate_centers.csv',index=False)
            em={'checkpoint_sha256':sha(cp),'train_script_sha256':sha(source_path('train_event_ssl.py')),
                'primary_coordinates':['M','Psi'],'splits':{},'convergence_reference':{'reference_M':0.,'reference_Psi':-.5}}
            for split in ('A_val','B_confirm'):
                data=Prepared(base/'stage4_event_ssl/prepared_inputs',split)
                arrays=data.as_original_arrays();S,P,_,_=ev.predict_split(model,arrays,torch.device('cpu'),512)
                df=pd.DataFrame({'user_id':data.values('uid'),'bundle_step_index':data.values('step'),
                  'M':data.values('X')[:,0],'Psi':data.values('X')[:,1],
                  'target_M_next':data.values('Y')[:,0],'target_Psi_next':data.values('Y')[:,1],
                  'pred_M':S[:,0],'pred_Psi':S[:,1],'pred_next_M':P[:,0],'pred_next_Psi':P[:,1]})
                df.to_csv(evaldir/'predictions'/f'stage4_event_ssl_predictions_{split}.csv.gz',index=False)
                part=ev.MacroPartition(np.zeros(2),np.ones(2),raw,raw,6,{})
                met,_=ev.metrics_for_predictions(arrays,S,P,part,(0.,-.5));met['split']=split
                pd.DataFrame([met]).to_csv(evaldir/'tables'/f'stage4_event_ssl_structural_metrics_{split}.csv',index=False)
            save_json(em,evaldir/'metadata/stage4_event_ssl_evaluation_manifest.json')

    fr=root/'four_term_mechanism_revision';(fr/'models').mkdir(parents=True);(fr/'confirmation').mkdir()
    (fr/'final').mkdir();save_json({'technical_completion':True,'smoke':False,'synthetic':True},fr/'final/COMPLETE.json')
    models={'C3':{'coefficients':[-.3378988759721414,-.8926628365969478,1.]},'synthetic':True}
    save_json(models,fr/'models/final_models.json');save_json({'model_payload_sha256':sha(fr/'models/final_models.json')},fr/'models/FIT_FROZEN.json')
    d=Prepared(inputroot,'B_confirm');X=d.values('X');Y=d.values('Y')
    df=pd.DataFrame({'user_id':d.values('uid'),'bundle_step_index':d.values('step'),
       'M':X[:,0],'Psi':X[:,1],'target_M_next':Y[:,0],'target_Psi_next':Y[:,1],
       'four_term_next_M':Y[:,0]+.001,'four_term_next_Psi':Y[:,1]+.001,
       'C3_next_M':Y[:,0]+.003,'C3_next_Psi':Y[:,1]+.002})
    df.to_csv(fr/'confirmation/row_predictions.csv.gz',index=False)
    return root

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args();create(args.root)
