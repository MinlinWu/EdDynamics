from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def require(value, message):
    if not bool(value):
        raise RuntimeError(message)

def render(grid_path, output):
    grid_path = Path(grid_path).resolve()
    output = Path(output).resolve()
    formal_root = grid_path.parent.parent
    require(grid_path.is_file(), 'Frozen input grid is missing')
    require(not output.exists(), 'Presentation output must be a new directory')
    require(not output.is_relative_to(formal_root) and (not formal_root.is_relative_to(output)), 'Presentation output must be separate from the formal tree')
    original_hash = sha(grid_path)
    grid = pd.read_csv(grid_path, float_precision='round_trip')
    require(set(grid.model) == {'four_term', 'C3', 'SSL'}, 'Unexpected grid models')
    base = grid[grid.model == 'four_term'].sort_values('cell')
    require(len(base) == 1600 and np.array_equal(base.cell.to_numpy(), np.arange(1600)), 'Expected the complete original 40x40 grid')
    supported = base.supported.astype(str).str.lower().isin(['true', '1', '1.0']).to_numpy()
    require(np.array_equal(supported, base['count'].to_numpy() >= 30), 'Support contract differs')
    pos = base[['M_center', 'Psi_center']].to_numpy(float)
    emp = base[['emp_M', 'emp_Psi']].to_numpy(float)
    fields = {}
    shared = ['cell', 'M_center', 'Psi_center', 'count', 'mass', 'emp_M', 'emp_Psi']
    for name in ('four_term', 'C3', 'SSL'):
        part = grid[grid.model == name].sort_values('cell')
        require(len(part) == 1600 and np.array_equal(part[shared].to_numpy(), base[shared].to_numpy(), equal_nan=True), 'Models do not share the identical frozen field grid: ' + name)
        mask = part.supported.astype(str).str.lower().isin(['true', '1', '1.0']).to_numpy()
        require(np.array_equal(mask, supported), 'Model support differs: ' + name)
        fields[name] = part[['model_M', 'model_Psi']].to_numpy(float)
        require(np.isfinite(fields[name][supported]).all(), 'Non-finite supported model field')
    require(np.isfinite(emp[supported]).all(), 'Non-finite supported empirical field')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True)
    panels = [('Empirical', emp), ('C3', fields['C3']), ('Four-term', fields['four_term']), ('Event-SSL (empirical anchor)', fields['SSL'])]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, (name, field) in zip(axs, panels):
        ax.quiver(pos[supported, 0], pos[supported, 1], field[supported, 0], field[supported, 1], angles='xy', scale_units='xy', scale=1, width=0.003)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect='equal', title=name, xlabel='Response evidence M')
    axs[0].set_ylabel('Exposure alignment Psi')
    fig.suptitle('Frozen conditional fields: identical support and arrow scale; seed 42')
    fig.savefig(output / 'raw_fields_seed42.png', dpi=220, bbox_inches='tight', pad_inches=0.25)
    fig.savefig(output / 'raw_fields_seed42.svg', bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    panels = [('Empirical - C3', emp - fields['C3']), ('Four-term - C3', fields['four_term'] - fields['C3']), ('Event-SSL - C3', fields['SSL'] - fields['C3'])]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.7), sharex=True, sharey=True, constrained_layout=True)
    for ax, (name, field) in zip(axs, panels):
        ax.quiver(pos[supported, 0], pos[supported, 1], field[supported, 0], field[supported, 1], angles='xy', scale_units='xy', scale=1, width=0.003)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect='equal', title=name, xlabel='Response evidence M')
    axs[0].set_ylabel('Exposure alignment Psi')
    fig.suptitle('C3-reference fields: identical support and arrow scale; no per-panel rescaling')
    fig.savefig(output / 'C3_excess_fields_seed42.png', dpi=220, bbox_inches='tight', pad_inches=0.25)
    fig.savefig(output / 'C3_excess_fields_seed42.svg', bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    require(sha(grid_path) == original_hash, 'Frozen input changed during presentation rendering')
    manifest = {'role': 'presentation-only re-render; no analysis recomputation or model fitting', 'input': str(grid_path), 'input_sha256': original_hash, 'script_sha256': sha(__file__), 'supported_cells': int(supported.sum()), 'unchanged': ['field values', 'support mask', 'panel order', 'axes [-1,1]', 'equal aspect', 'quiver angles=xy', 'scale_units=xy', 'scale=1', 'width=.003', 'figure sizes', 'titles', 'PNG dpi=220'], 'presentation_changes': ['constrained_layout=True', "bbox_inches='tight'", 'pad_inches=.25'], 'formal_tree_modified': False, 'outputs': {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / 'PRESENTATION_COMPLETE.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    print(json.dumps(manifest, indent=2))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render frozen population fields using common axes and scales.')
    parser.add_argument('--grid', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.grid, args.output)
