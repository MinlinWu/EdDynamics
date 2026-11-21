
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ssl_common import *

MAP=[
('same_readout','S1','S1/*/seed_*/B_confirm/evaluation/metrics.csv',
 'Frozen readout applied before and after an event; conditional proposals and actual readout changes are separate quantities.'),
('state_and_event_information','S2','S2/seed_*/B_confirm/evaluation/metrics.csv',
 'Matched fresh prediction heads share initialization and budget on a frozen encoder; this is not an isolated objective-loss comparison.'),
('derived_information','S3','S3/seed_*/*/B_confirm/evaluation/metrics.csv',
 'Alignment-derived inputs and corresponding future targets are jointly removed; probes are fitted only after encoder freezing.'),
('record_order','S1 shuffled','S0/shuffle_scope.json',
 'Complete records are shuffled and evaluated on ordered records; upstream state variables are not reconstructed.'),
('common_reference','endpoint bridge','final/C3_bridge_all.csv',
 'Common-reference endpoint comparisons do not identify a shared internal mechanism or replace a generative null.'),
('uncertainty','paired whole-learner resampling','final/paired_intervals_all.csv',
 'Seed42 fitted models are held fixed; training-seed spread is reported separately from learner-bootstrap ranges.')]

IDENTITY_COLUMNS=['block','seed','model']
FIELD_DISPLAY_METRICS=['vector_r','speed_r','local_cosine','amplitude_slope',
    'D2','D2_M','D2_Psi','reference_energy','reference_relative_skill',
    'cells','support_mass','support_mass_fraction','cosine_mass_fraction','status']
INTERVAL_DIAGNOSTICS=['point','requested_replicates','finite_replicates','bootstrap_mean',
    'bootstrap_bias','bootstrap_sd','bias_over_sd','basic_low','basic_high',
    'percentile_low','percentile_high','se_low','se_high','method_sign_conflict','status']
INFERENCE_BOUNDARY=(
    'The frozen two primary error contrasts per block are unchanged. Fixed-support nominal 97.5% basic ranges '
    'retain that prespecified role; reselected support, nominal 95% ranges and alternative methods are sensitivities. '
    'The nominal adjustment concerns the two contrasts within each block, not a joint guarantee across blocks or support rules. '
    'Population coverage has not been verified. Finite draws, a small bootstrap displacement, or agreement of interval signs '
    'do not establish coverage. Resampling conditions on frozen fitted models, heads, probes and declared processing; '
    'training-seed variation is separate. All methods, negative results and NA are retained. '
    'Nominal ranges crossing zero do not establish equivalence; ranges on one side do not by themselves establish a population direction.')
C3_BOUNDARY=(
    'C3 bridge rows concern endpoint N-X versus Y-X on the exact historical mechanism-export key intersection. '
    'For same_readout, N is T, so this is T-X, not the actual trajectory T-S. '
    'C3-correction rows subtract the same fitted C3 field from both endpoint fields. '
    'Neither bridge is a response-generative construction-null test or independent mechanism residual; '
    'model displacement N-C is reported separately, identifying actual same-readout T-S by model. '
    'C3 and H-only also have different information conditions.')


def markdown(df,cols):

    frame=df.reindex(columns=cols)
    def fmt(v):
        if v is None or (np.isscalar(v) and pd.isna(v)):return 'NA'
        if isinstance(v,(bool,np.bool_)):return str(bool(v))
        if isinstance(v,(int,float,np.number)):return f'{v:.7g}' if np.isfinite(v) else 'NA'
        return str(v).replace('|','\\|').replace('\n',' ')
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join(
        '| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))


def metric_sections(metrics):

    metrics=metrics.copy()
    def quantity(row):
        model=str(row.get('model',''));block=str(row.get('block',''))
        if model=='same_readout' or (block=='S3' and model in ('aug','clean')):
            return 'actual same-readout change T-S'
        if model in ('proposal','archived_proposal','H','HZ'):
            return 'conditional proposal displacement P-S'
        return 'N-C; inspect model-specific readout contract'
    metrics['velocity_quantity']=metrics.apply(quantity,axis=1)
    sections=[]
    for view,definition in [('endpoint','N-X versus Y-X, binned at empirical X'),
                            ('velocity','Model displacement N-C versus Y-X, binned at empirical X')]:
        sections.append(f'### {view}: {definition}\n')
        ids=IDENTITY_COLUMNS+(['velocity_quantity'] if view=='velocity' else [])
        for label,fields in [
            ('Direction, speed and amplitude',['vector_r','speed_r','local_cosine','amplitude_slope']),
            ('Absolute error and reference scale',['D2','D2_M','D2_Psi','reference_energy','reference_relative_skill']),
            ('Support and availability',['cells','support_mass','support_mass_fraction','cosine_mass_fraction','status'])]:
            sections.append(label+'\n\n'+markdown(metrics,ids+[view+'_'+m for m in fields])+'\n')
    sections.append('### Current-state readability\n\n'+markdown(metrics,IDENTITY_COLUMNS+[
        'current_MSE','current_rmse_M','current_rmse_Psi','current_r_M','current_r_Psi'])+'\n')
    sections.append('### Next-state reconstruction and interval change\n\n'+markdown(metrics,IDENTITY_COLUMNS+[
        'next_MSE','one_step_rmse_M','one_step_rmse_Psi','next_r_M','next_r_Psi',
        'user_equal_next_rmse_M','user_equal_next_rmse_Psi','interval_velocity_MSE'])+'\n')
    sections.append('### Frozen K6 summaries (six-state correlations descriptive)\n\n'+markdown(metrics,IDENTITY_COLUMNS+[
        'anchor_transition_mean_row_TV','anchor_self_transition_r_descriptive',
        'readout_transition_mean_row_TV','readout_self_transition_r_descriptive'])+'\n')
    sections.append('Interval errors are row-equal; fields use equal total valid-row mass per learner; '
        'K6 transitions use interval counts then row normalization. These are distinct estimands. '
        'For S1 proposal and S2 H/HZ, N-C=P-S is a conditional proposal displacement. '
        'Only S1 same_readout and the S3 single-probe rows measure actual same-readout change T-S. '
        'The separate learned_plane and matched_origin gauges remain in confirmation_metrics_all.csv with their original labels; '
        'they are not substituted for actual velocity. This is the declared diagnostic set, not a claim to include every original occupancy diagnostic.\n')
    return '\n'.join(sections)


def interval_interpretations(intervals):

    rows=[]
    for _,r in intervals.iterrows():
        if r.get('status')!='complete' or not np.isfinite(r.get('basic_low',np.nan)) or not np.isfinite(r.get('basic_high',np.nan)):
            label='not_estimable'
        elif r.basic_low>0:label='nominal_basic_range_above_zero'
        elif r.basic_high<0:label='nominal_basic_range_below_zero'
        else:label='nominal_basic_range_includes_zero_not_equivalence'
        row={name:r.get(name,np.nan) for name in ['block','contrast','model_a','model_b','metric',
            'support_contract','level','primary_level','primary_analysis','primary_support','support_role',
            'family_scope','family_size','inference_condition']+INTERVAL_DIAGNOSTICS}
        row.update(interpretation=label,coverage_status='not_verified',
                   availability_status_is_not_coverage=True,
                   scientific_outcome_does_not_control_completion=True,
                   interpretation_scope='nominal method range, not a certified population direction')
        rows.append(row)
    return pd.DataFrame(rows)


def interval_sections(intervals):
    diagnostics=interval_interpretations(intervals)
    primary=(diagnostics.support_contract=='fixed')&np.isclose(diagnostics.level,.975)
    sections=['### Paired learner-resampling method diagnostics\n\n'+INFERENCE_BOUNDARY+'\n']
    keys=['block','contrast','support_contract','level','primary_analysis']
    for label,frame in [('Prespecified fixed-support primary contrasts',diagnostics[primary]),
                        ('Other levels and reselected-support sensitivities',diagnostics[~primary])]:
        sections.append('#### '+label+'\n')
        sections.append(markdown(frame,keys+['point','requested_replicates','finite_replicates','bootstrap_mean',
            'bootstrap_bias','bootstrap_sd','bias_over_sd','status'])+'\n')
        sections.append(markdown(frame,keys+['basic_low','basic_high','percentile_low','percentile_high',
            'se_low','se_high','method_sign_conflict','interpretation'])+'\n')
    return '\n'.join(sections)


def scope_status():
    return {'full_loss_factor_design':'not performed; no isolated-loss claims',
        'new_physical_shuffle':'not performed; record-level transfer only',
        'old_CN_C3_ARX_P2':'available files indexed; C3 comparisons use matched rows from the required frozen mechanism export',
        'original_pure_SSL_and_task_only':'pre-existing comparisons of encoder training configurations; reported separately from S3',
        'S1_role':'needed for a stronger actual same-readout trajectory claim, not a prerequisite for the existing empirical/conditional-model chain',
        'S3_role':'supplementary joint-information boundary; not a core-success requirement',
        'informative_censoring':'stated limitation retained; not resolved by this package',
        'external_population_validation':'not performed'}


def make_figures(out,metrics,intervals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=Path(out)/'figures';figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','svg','pdf'):fig.savefig(figdir/(name+'.'+ext),dpi=300,bbox_inches='tight')
        plt.close(fig)

    if len(intervals):
        q=intervals[(intervals.support_contract=='fixed')&np.isclose(intervals.level,.975)]
        for i,r in q.iterrows():
            fig,ax=plt.subplots(figsize=(6.4,2.7));p=float(r.point)
            if r.status=='complete':

                ax.plot([r.basic_low,r.basic_high,np.nan,r.basic_low,r.basic_low,np.nan,r.basic_high,r.basic_high],
                        [0,0,np.nan,-.06,.06,np.nan,-.06,.06],linewidth=2)
            ax.plot([p],[0],'o');ax.axvline(0,linestyle='--',linewidth=.8)
            ax.set_yticks([]);ax.set_ylim(-.4,.4);ax.set_xlabel('Paired error difference (unscaled coordinate units squared)')
            ax.set_title(str(r.contrast).replace('_',' '),wrap=True)
            shift=float(r.get('bias_over_sd',np.nan))
            shift_text=f'{shift:.3g}' if np.isfinite(shift) else 'NA'
            ax.text(.02,.05,f"Nominal 97.5% basic; {r.status}; finite {int(r.finite_replicates)}/{int(r.requested_replicates)}\n"
                    f"bootstrap displacement/SD {shift_text}; population coverage unverified",transform=ax.transAxes,fontsize=8)
            save(fig,f'{r.block}_{r.contrast}')
    if len(metrics):
        keep=metrics[(metrics.seed==42)&(metrics.block.isin(['S1_full','S2','S3']))]
        for metric in ['endpoint_D2','velocity_D2','current_MSE','next_MSE']:
            if metric not in keep:continue
            y=pd.to_numeric(keep[metric],errors='coerce').to_numpy();x=np.arange(len(keep))
            fig,ax=plt.subplots(figsize=(7.5,3.8));ax.bar(x,y)
            ax.set_xticks(x,[f'{a}/{b}' for a,b in zip(keep.block,keep.model)],rotation=25,ha='right')
            ax.set_ylabel(metric.replace('_',' '));ax.set_title('Seed 42; distinct models and information conditions')
            save(fig,'overview_'+metric)


def make_field_figures(ctx,out):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=[('S1_model_displacement',ctx.out/'S1/full/seed_42/B_confirm/evaluation/field_grids.csv.gz','velocity'),
            ('S2_endpoint',ctx.out/'S2/seed_42/B_confirm/evaluation/field_grids.csv.gz','endpoint')]
    cleanfiles=[ctx.out/f'S3/seed_42/{v}/B_confirm/evaluation/field_grids.csv.gz' for v in ('aug','clean')]
    frames=[]
    for name,path,view in groups:
        df=pd.read_csv(path);frames.append((name,df[df.view==view]))
    both=pd.concat([pd.read_csv(p) for p in cleanfiles],ignore_index=True)
    frames.append(('S3_actual_velocity',both[both.view=='velocity']))
    target=Path(out)/'figures';source=Path(out)/'figure_source';source.mkdir(exist_ok=True)
    for block,df in frames:
        df=df.copy();write_csv(df,source/(block+'.csv.gz'))
        supported=df[df.supported.astype(bool)]
        norm=np.hypot(supported.model_M,supported.model_Psi).to_numpy()
        en=np.hypot(supported.emp_M,supported.emp_Psi).to_numpy()
        lengths=np.r_[norm,en];lengths=lengths[np.isfinite(lengths)]
        unit=max(float(np.quantile(lengths,.9)) if len(lengths) else .1,1e-4)
        scale=unit/.12
        variants=[('empirical',df[df.model==df.model.iloc[0]],'emp')]
        variants += [(str(model),d,'model') for model,d in df.groupby('model',sort=False)]
        for model,d,prefix in variants:
            mask=d.supported.astype(bool).to_numpy()

            if mask.sum()>300:mask &= ((d.cell.to_numpy()//40)%2==0)&((d.cell.to_numpy()%40)%2==0)
            g=d.cell.to_numpy()[mask];x=-1+(g//40+.5)*.05;y=-1+(g%40+.5)*.05
            u=d[prefix+'_M'].to_numpy()[mask];v=d[prefix+'_Psi'].to_numpy()[mask]
            fig,ax=plt.subplots(figsize=(5.7,5.2))
            if len(g):
                q=ax.quiver(x,y,u,v,angles='xy',scale_units='xy',scale=scale,width=.003)
                ax.quiverkey(q,.90,.95,unit,f'{unit:.3g} / event',labelpos='W',coordinates='axes',fontproperties={'size':9})
            else:ax.text(.5,.5,'No jointly supported cells',transform=ax.transAxes,ha='center')
            ax.axhline(0,linewidth=.5,linestyle=':');ax.axvline(0,linewidth=.5,linestyle=':')
            quantity=('empirical Y-X' if prefix=='emp' else
                      'conditional proposal P-S' if block=='S1_model_displacement' and model=='proposal' else
                      'actual same-readout T-S' if block in ('S1_model_displacement','S3_actual_velocity') else 'endpoint N-X')
            ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='Response evidence M',ylabel='Activity alignment Ψ',
                   title=f'{block.replace("_"," ")}\n{model.replace("_"," ")}: {quantity}')
            ax.set_aspect('equal');fig.tight_layout()
            for ext in ('png','svg','pdf'):fig.savefig(target/f'{block}_{model}.{ext}',dpi=300,bbox_inches='tight')
            plt.close(fig)


def finalize(ctx):
    from ssl_pipeline import assert_complete
    dependencies=assert_complete(ctx)
    def produce(dst):
        rows=[];ivs=[];bridges=[];replay=[]
        for p in sorted(ctx.out.glob('S*/**/B_confirm/evaluation/metrics.csv')):
            rel=p.relative_to(ctx.out);parts=rel.parts
            seed=int(next(x[5:] for x in parts if x.startswith('seed_')))
            block='S1_'+parts[1] if parts[0]=='S1' else parts[0]
            df=pd.read_csv(p);df['seed']=seed;df['block']=block;df['source']=str(rel)
            rows.append(df)
        for p in sorted(ctx.out.glob('S*/**/inference/paired_intervals.csv')):
            df=pd.read_csv(p);df['block']=p.relative_to(ctx.out).parts[0];df['source']=str(p.relative_to(ctx.out));ivs.append(df)
        for p in sorted(ctx.out.glob('S*/**/B_confirm/bridge/C3_bridge_metrics.csv')):
            df=pd.read_csv(p);df['source']=str(p.relative_to(ctx.out));bridges.append(df)
        for p in ctx.out.glob('S1/**/S0_archive_bridge.json'):
            r=load_json(p);replay.append({'source':str(p.relative_to(ctx.out)),'status':r['status'],
                 'max_absolute_error':max(r['max_absolute_cpu_minus_archive'])})
        metrics=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
        intervals=pd.concat(ivs,ignore_index=True) if ivs else pd.DataFrame()
        require(len(metrics)>0 and len(intervals)>0,'Cannot create empty final results')
        write_csv(metrics,dst/'confirmation_metrics_all.csv');write_csv(intervals,dst/'paired_intervals_all.csv')
        write_csv(pd.concat(bridges,ignore_index=True) if bridges else pd.DataFrame(),dst/'C3_bridge_all.csv')
        write_csv(replay,dst/'CPU_vs_archive_precision.csv')
        write_csv([dict(zip(['analysis','comparison','evidence','interpretation_boundary'],r)) for r in MAP],dst/'analysis_index.csv')
        write_csv(interval_interpretations(intervals),dst/'result_interpretation.csv')
        make_figures(dst,metrics,intervals)
        make_field_figures(ctx,dst)
        future=[]
        for p in sorted(ctx.out.glob('S3/**/B_confirm/future_prediction_heldout.csv')):
            df=pd.read_csv(p);rel=p.relative_to(ctx.out).parts
            df['seed']=int(rel[1][5:]);df['variant']=rel[2];df['source']=str(p.relative_to(ctx.out));future.append(df)
        if future:write_csv(pd.concat(future,ignore_index=True),dst/'common_future_targets_confirmation.csv')
        table=metric_sections(metrics)
        uncertainty=interval_sections(intervals)
        title='SMOKE — reduced-data software check' if ctx.smoke else 'SSL supplemental numerical evidence'
        text=f'# {title}\n\nContract: `{ctx.contract}`. CPU float32. Fixed parameters; no confirmation fitting.\n\n'+table+'\n\n'
        text+='## Interpretation\n\nState accessibility, next-state reconstruction, actual readout change and field calibration are different endpoints. '
        text+='All unfavorable outcomes and nonestimable ranges are retained. A positive point contrast means the first named model has larger observed error. '
        text+='Paired error differences are reported separately for each prespecified contrast. '
        text+=f'The training-seed set for the paired prediction-head and information-removal comparisons is {", ".join(map(str,ctx.newseeds))}. '
        text+=f'The intervals use {ctx.nboot:,} paired whole-learner draws with the seed-42 models held fixed.\n\n'+uncertainty+'\n\n'
        text+='## C3 endpoint bridges\n\n'+C3_BOUNDARY+'\n\n'
        text+='## Historical precision bridge\n\nHistorical GPU/BF16 outputs are immutable. CPU recomputations have their own row/field values. '
        text+='CPU_vs_archive_precision.csv summarizes the maximum absolute differences. Each S0_archive_bridge.json records the prediction export, checkpoint hash and per-column differences for the current-state readout and next-state proposal.\n\n'
        text+='## Related analyses\n\nS0/existing_evidence_inventory.json lists the earlier analysis files found in the input tree. The list can be incomplete or empty. '
        text+='The standard workflow requires a frozen mechanism prediction export, identified in S0/reference_identity.json. C3_bridge_all.csv compares endpoint fields for confirmation intervals matched to that export. '
        text+='The original future-prediction-only and task-only models provide separate comparisons of training configurations. S3 examines the joint removal of alignment-derived inputs and their future targets. '
        text+='The response-generating controls are evaluated separately. Unavailable fixed-support field estimates remain missing.\n'
        if ctx.smoke:text+=f'\nTechnical smoke: up to {getattr(ctx,"maxusers",32)} whole learners per split, one training epoch, and {ctx.nboot} paired learner draws. These are reduced-data checks, not formal scientific evidence.\n'
        (dst/'numeric_report.md').write_text(text,encoding='utf-8')
        save_json(scope_status(),dst/'SCOPE_STATUS.json')
        save_json({'technical_completion':True,'scientific_success_required':False,'smoke':ctx.smoke,
           'synthetic':(ctx.history/'_SYNTHETIC_FIXTURE.json').exists(),'contract':ctx.contract,
           'new_encoder_conditions':['F00-clean'],'matched_reference_replay':['F00-aug CPU fixed epoch'],
           'actual_encoder_runs':2*len(ctx.newseeds),'seed42_bootstrap_replicates':ctx.nboot,
           'CPU_archive_distinct_views':sum(x['status']=='distinct_CPU_precision_view' for x in replay),
           'outputs_are_real_computations_not_filled_results':True},dst/'COMPLETE.json')
    return stage(ctx.out,'final',ctx.contract,dependencies,produce)
