#!/usr/bin/env python3


import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description='Run the construction-control workflow using bounded CPU subprocesses.')
    parser.add_argument('--code-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--repo-root', required=True)
    parser.add_argument('--work-root', required=True)
    parser.add_argument('--smoke-root', required=True)
    parser.add_argument('--phase', choices=['smoke', 'prepare', 'formal'], required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke-users', type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.workers <= len(os.sched_getaffinity(0)):
        parser.error('workers must be within available CPU affinity')
    code = Path(args.code_root).resolve()
    work = Path(args.smoke_root if args.phase == 'smoke' else args.work_root).resolve()
    run = code / 'run_generative_construction_null.py'
    summary = code / 'summarize_generative_construction_null.py'
    logs = work / 'direct_logs'
    logs.mkdir(parents=True, exist_ok=True)
    lock_handle = (work / 'direct_controller.lock').open('a')
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update({k: '1' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                               'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']})
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    env['MPLCONFIGDIR'] = str(work / 'matplotlib_cache')
    status = {'phase': args.phase, 'pid': os.getpid(), 'started': utcnow(),
              'workers': args.workers, 'python': sys.executable,
              'execution': 'direct CPU subprocesses; no Slurm', 'state': 'running',
              'stages_completed': [], 'active': [], 'work_root': str(work)}
    active = {}

    def save():
        status['updated'] = utcnow()
        status['active'] = [{'name': x['name'], 'pid': p.pid,
                             'elapsed_seconds': round(time.monotonic()-x['start'], 1)}
                            for p, x in active.items()]
        atomic_json(work / 'direct_status.json', status)

    def command(script, subcommand, *extra, target=work):
        return [sys.executable, '-B', str(script), subcommand,
                '--work-root', str(target), *map(str, extra)]

    def launch(name, cmd):
        log = logs / (name + '.log')
        resource_log = logs / (name + '.resources.txt')
        handle = log.open('a', buffering=1)
        handle.write('\nDIRECT_LAUNCH '+json.dumps({'at': utcnow(), 'command': cmd})+'\n')
        actual = ['/usr/bin/time', '-a', '-v', '-o', str(resource_log), *cmd]
        proc = subprocess.Popen(actual, stdout=handle, stderr=subprocess.STDOUT,
                                env=env, cwd=code, start_new_session=True)
        active[proc] = {'name': name, 'handle': handle, 'start': time.monotonic(), 'command': cmd}
        save()
        return proc

    def finish(proc):
        record = active.pop(proc)
        record['handle'].close()
        item = {'name': record['name'], 'returncode': proc.returncode,
                'elapsed_seconds': round(time.monotonic()-record['start'], 3), 'finished': utcnow()}
        with (work / 'direct_receipts.jsonl').open('a') as stream:
            stream.write(json.dumps(item)+'\n')
        if proc.returncode:
            raise RuntimeError('Failed '+record['name']+'; inspect '+str(logs/(record['name']+'.log')))
        return item

    def serial(name, cmd):
        status['stage'] = name
        proc = launch(name, cmd)
        while proc.poll() is None:
            time.sleep(2)
            save()
        item = finish(proc)
        status['stages_completed'].append(item)
        print(json.dumps({'stage_complete': item}), flush=True)
        save()

    def simulations(split, mode, replicates):
        status['stage'] = split+'_'+mode
        status['replicates_total'] = replicates
        status['replicates_completed'] = 0
        next_id = 0
        while next_id < replicates or active:
            resource_file = work / 'direct_resources.json'
            workers = args.workers
            if resource_file.exists():
                workers = int(json.loads(resource_file.read_text())['workers'])
                if not 1 <= workers <= len(os.sched_getaffinity(0)):
                    raise ValueError('Invalid direct CPU resource cap')
            if status['workers'] != workers:
                print(json.dumps({'resource_cap_changed': workers, 'at': utcnow()}), flush=True)
                status['workers'] = workers
            while next_id < replicates and len(active) < workers:
                name = f'{split}_{mode}_{next_id:04d}'
                launch(name, command(run, 'simulate', '--split', split,
                                     '--mode', mode, '--replicate', next_id))
                next_id += 1
            for proc in list(active):
                if proc.poll() is not None:
                    item = finish(proc)
                    status['replicates_completed'] += 1
                    print(json.dumps({'replicate_complete': item,
                                      'group_completed': status['replicates_completed']}), flush=True)
            save()
            if active:
                time.sleep(2)
        status['stages_completed'].append({'name': status['stage'],
                                          'replicates': replicates, 'finished': utcnow()})
        save()

    def terminate(signum, frame):
        raise KeyboardInterrupt('Signal '+str(signum))

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        save()
        if args.phase != 'smoke':

            serial('verify_smoke', command(run, 'smoke-finish', target=Path(args.smoke_root)))
            sys.path.insert(0, str(code))
            import cn_common as common
            sp = common.load_protocol(Path(args.smoke_root))
            if sp['seed'] != args.seed or Path(sp['output_root']).resolve() != Path(args.output_root).resolve():
                raise RuntimeError('Smoke seed/output differs from formal run')
            sources = common.source_paths(args.repo_root)
            if sp['source_hashes'] != {k: common.sha256(sources[k]) for k in ('stage1','semantic','original_null')}:
                raise RuntimeError('Smoke sources differ from formal run')
        init = command(run, 'init', '--output-root', args.output_root, '--repo-root', args.repo_root,
                       '--seed', args.seed, '--replicates', 3 if args.phase == 'smoke' else 999)
        if args.phase == 'smoke':
            init += ['--smoke', '--smoke-users', str(args.smoke_users)]
        serial('init', init)
        serial('calibrate', command(run, 'calibrate'))
        serial('prepare_A_val', command(run, 'prepare', '--split', 'A_val'))
        if args.phase != 'prepare':
            splits = ['A_val'] if args.phase == 'smoke' else ['A_val', 'B_confirm']
            for split in splits:
                if split == 'B_confirm':
                    serial('prepare_B_confirm', command(run, 'prepare', '--split', split))
                for mode in ['null_a', 'null_b']:
                    simulations(split, mode, 3 if args.phase == 'smoke' else 999)
                    if args.phase != 'smoke':
                        serial('summary_'+split+'_'+mode,
                               command(summary, 'summarize', '--split', split, '--mode', mode))
                if split == 'A_val' and args.phase != 'smoke':
                    serial('release_A_val', command(summary, 'release'))
            serial('smoke_finish' if args.phase == 'smoke' else 'final_report',
                   command(run, 'smoke-finish') if args.phase == 'smoke' else command(summary, 'final'))
        status['state'] = 'completed'
        status['finished'] = utcnow()
        save()
    except BaseException as error:
        status['state'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        status['error'] = repr(error)
        status['traceback'] = traceback.format_exc()
        for proc in list(active):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for proc, record in list(active.items()):
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            record['handle'].close()
        active.clear()
        save()
        raise


if __name__ == '__main__':
    main()
