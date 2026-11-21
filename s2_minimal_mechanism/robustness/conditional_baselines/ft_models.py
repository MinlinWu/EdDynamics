from __future__ import annotations
from dataclasses import fields
from itertools import product
import math
import numpy as np
from ft_common import EPS, FEATURES, PARAMS, require

def calibration_object(p1, c):
    return p1.Calibration(**{x.name: float(c[x.name]) for x in fields(p1.Calibration) if x.name in c and c[x.name] is not None})

def check_features(F):
    F = np.asarray(F, dtype=np.float64)
    require(F.ndim == 2 and F.shape[1] == len(FEATURES), 'Predictor accepts exactly the declared 12 features')
    require(np.isfinite(F).all(), 'Non-finite predictor input; refusing row deletion')
    return F

def four_predict(p1, F, keys, params, cal):
    F = check_features(F)
    a = p1._simulate_core_dispatch(np.asarray(keys[:, 0]), np.asarray(keys[:, 1]), F[:, 0], F[:, 1], F[:, 2], F[:, 3], F[:, 4], F[:, 11], F[:, 5], F[:, 6], F[:, 7], F[:, 8], F[:, 9], F[:, 10], np.asarray([params[k] for k in PARAMS]), calibration_object(p1, cal))
    pred = np.column_stack([a[6], a[7]])
    phase = np.column_stack([a[12], a[14], a[15]])
    require(np.isfinite(pred).all(), 'Frozen prediction is non-finite')
    return (pred, phase)

def constant_predict(F, c, params, cal):
    F = check_features(F)
    c = np.asarray(c, dtype=float)
    require(c.shape == (3,) and np.isfinite(c).all() and (np.max(np.abs(c)) <= 1 + 1e-12), 'C3 coefficients out of bounds')
    M = np.clip(F[:, 0], -1, 1)
    E = np.maximum(F[:, 2], 0)
    B = np.maximum(F[:, 3], EPS)
    G = np.clip(F[:, 4], -B, B)
    gap = np.maximum(F[:, 11], 0)
    rR = np.exp(-gap / max(cal['tau_response_days'], EPS))
    rA = np.exp(-gap / max(cal['tau_activity_days'], EPS))
    R = np.maximum(params['lambdaR'] * cal['residual_mass_per_answer'] * np.maximum(F[:, 5], 0), 0)
    Sn = rR * (M * E + max(cal['response_signed_gain'], 0) * R * c[0])
    En = rR * np.maximum(E + R, 0)
    Mn = np.where(En > EPS, Sn / np.maximum(En, EPS), 0)
    Ar = np.maximum(params['lambdaA'] * F[:, 6], 0)
    As = np.maximum(params['lambdaA'] * F[:, 7], 0)
    A0r = np.maximum(params['lambdaA'] * F[:, 8], 0)
    A0s = np.maximum(params['lambdaA'] * F[:, 9], 0)
    I = np.maximum(params['lambdaI'] * F[:, 10], 0)
    Ba = np.maximum(B + (Ar + As) + (A0r + A0s), EPS)
    Ga = np.clip(G + max(cal['alignment_signed_gain'], 0) * (Ar * c[1] + As * c[2]), -Ba, Ba)
    Bi = np.maximum(Ba + I, EPS)
    Bn = rA * Bi
    Gn = np.clip(rA * Ga, -Bn, Bn)
    Pn = Gn / np.maximum(Bn, EPS)
    pred = np.clip(np.column_stack([Mn, Pn]), -1, 1)
    require(np.isfinite(pred).all(), 'Non-finite C3 prediction')
    return pred

def c3_basis(F, params, cal):
    base = constant_predict(F, [0, 0, 0], params, cal)
    A = np.stack([constant_predict(F, np.eye(3)[j], params, cal) - base for j in range(3)], axis=2)
    return (base, A)

def c3_coefficients_to_four(c, base_params):
    c = np.asarray(c)
    require(np.max(np.abs(c)) < 1, 'atanh test embedding requires interior constants')
    p = dict(base_params)
    p.update(theta0=float(np.arctanh(c[0])), thetaM=0.0, thetaPsi=0.0, thetaMPsi=0.0, phi0=float(np.arctanh(c[1])), deltaS=float(np.arctanh(c[2]) - np.arctanh(c[1])), phiPsi=0.0)
    return p

def bounded_min_norm(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    require(A.ndim == 2 and A.shape[1] == 3 and (len(A) == len(b)), 'Invalid C3 system')
    require(np.isfinite(A).all() and np.isfinite(b).all() and (len(b) > 0), 'Empty/non-finite C3 fit')
    candidates = []
    for state in product((-1, 0, 1), repeat=3):
        state = np.array(state)
        fixed = state != 0
        free = ~fixed
        c = state.astype(float)
        if free.any():
            c[free] = np.linalg.lstsq(A[:, free], b - A[:, fixed] @ c[fixed], rcond=1e-12)[0]
        if np.max(np.abs(c)) > 1 + 1e-10:
            continue
        c = np.clip(c, -1, 1)
        residual = A @ c - b
        candidates.append((float(residual @ residual), float(c @ c), tuple(c)))
    require(bool(candidates), 'C3 constrained solve has no feasible candidate')
    best = min((x[0] for x in candidates))
    tol = 1e-13 * max(float(b @ b), best, 1e-12)
    tied = [x for x in candidates if x[0] <= best + tol]
    chosen = min(tied, key=lambda v: (v[1], v[2]))
    sv = np.linalg.svd(A, compute_uv=False)
    return (np.array(chosen[2]), {'sse': chosen[0], 'rank': int(np.linalg.matrix_rank(A)), 'singular_values': sv.tolist(), 'feasible_faces': len(candidates), 'tie_sse_tolerance': tol, 'solution_rule': 'active_faces_minimum_euclidean_norm', 'status': 'solved', 'identifiable_all_coefficients': bool(np.linalg.matrix_rank(A) == 3)})

def ridge_fit(datasets, alpha, block_size=200000):
    n = 0
    mean = np.zeros(len(FEATURES))
    M2 = np.zeros(len(FEATURES))
    ymean = np.zeros(2)
    for d in datasets:
        for s in d.blocks(block_size):
            x = np.asarray(d.F[s])
            y = np.asarray(d.Y[s]) - x[:, :2]
            nb = len(x)
            mb = x.mean(axis=0)
            delta = mb - mean
            total = n + nb
            M2 += np.sum((x - mb) ** 2, axis=0) + delta ** 2 * (n * nb / max(total, 1))
            mean += delta * (nb / max(total, 1))
            ymean += (y.mean(axis=0) - ymean) * (nb / max(total, 1))
            n = total
    require(n > 0, 'No ARX training observations')
    scale = np.sqrt(np.maximum(M2 / n, 0))
    active = scale > 1e-12
    require(active.any(), 'All ARX predictors have zero variance')
    gram = np.zeros((active.sum(), active.sum()))
    cross = np.zeros((active.sum(), 2))
    zmean = np.zeros(active.sum())
    for d in datasets:
        for s in d.blocks(block_size):
            x = np.asarray(d.F[s])
            z = (x[:, active] - mean[active]) / scale[active]
            y = np.asarray(d.Y[s]) - x[:, :2] - ymean
            gram += z.T @ z / n
            cross += z.T @ y / n
            zmean += z.sum(axis=0) / n
    gram -= np.outer(zmean, zmean)
    return ridge_from_moments(mean, scale, active, ymean, zmean, gram, cross, n, alpha)

def ridge_from_moments(mean, scale, active, ymean, zmean, gram, cross, n, alpha):
    gram = (gram + gram.T) / 2
    ev, Q = np.linalg.eigh(gram)
    tol = max(float(ev.max(initial=0)), 1) * 1e-12
    require(ev.min(initial=0) > -100 * tol, 'ARX covariance not positive semidefinite')
    ev = np.maximum(ev, 0)
    if alpha == 0:
        inv = np.divide(1.0, ev, out=np.zeros_like(ev), where=ev > tol)
    else:
        inv = 1 / (ev + alpha)
    coef = Q * inv @ (Q.T @ cross)
    intercept = np.asarray(ymean) - np.asarray(zmean) @ coef
    return {'alpha': float(alpha), 'features': list(FEATURES), 'n_fit': int(n), 'mean': np.asarray(mean).tolist(), 'scale': np.asarray(scale).tolist(), 'active': np.asarray(active).tolist(), 'coef': coef.tolist(), 'intercept': intercept.tolist(), 'gram': gram.tolist(), 'cross': cross.tolist(), 'zmean': np.asarray(zmean).tolist(), 'ymean': np.asarray(ymean).tolist(), 'eigenvalues': ev.tolist(), 'rank': int((ev > tol).sum()), 'loss': 'mean interval squared error plus alpha*Frobenius(W)^2; intercept unpenalized'}

def ridge_change_alpha(model, alpha):
    return ridge_from_moments(np.array(model['mean']), np.array(model['scale']), np.array(model['active']), np.array(model['ymean']), np.array(model['zmean']), np.array(model['gram']), np.array(model['cross']), model['n_fit'], alpha)

def ridge_predict(F, model, return_raw=False):
    F = check_features(F)
    require(tuple(model['features']) == FEATURES, 'ARX feature order changed')
    active = np.array(model['active'], dtype=bool)
    z = (F[:, active] - np.array(model['mean'])[active]) / np.array(model['scale'])[active]
    raw = F[:, :2] + np.array(model['intercept']) + z @ np.array(model['coef'])
    require(np.isfinite(raw).all(), 'Non-finite ARX prediction')
    result = np.clip(raw, -1, 1)
    return (result, raw) if return_raw else result

def decompose(F, pred, params, cal):
    F = check_features(F)
    M, P, E, B, G, q, AR, AS, AR0, AS0, I, gap = F.T
    R = params['lambdaR'] * cal['residual_mass_per_answer'] * q
    C = params['lambdaA'] * (AR + AS + AR0 + AS0) + params['lambdaI'] * I
    dM = np.maximum(E + R, EPS)
    dP = np.maximum(B + C, EPS)
    out = np.zeros((len(F), 8))
    out[:, 0] = -R * M / dM
    out[:, 1] = cal['response_signed_gain'] * R * np.tanh(params['theta0'] - params['thetaM'] * M) / dM
    out[:, 2] = pred[:, 0] - M - out[:, 0] - out[:, 1]
    out[:, 3] = -C * P / dP
    out[:, 4] = cal['alignment_signed_gain'] * params['lambdaA'] * AR * np.tanh(params['phi0']) / dP
    out[:, 5] = cal['alignment_signed_gain'] * params['lambdaA'] * AS * np.tanh(params['phi0'] + params['deltaS']) / dP
    out[:, 6] = (G - P * B) / dP
    out[:, 7] = pred[:, 1] - P - out[:, 3:7].sum(axis=1)
    require(np.allclose(out[:, :3].sum(axis=1), pred[:, 0] - M, atol=1e-12, rtol=1e-10), 'M component add-back failure')
    require(np.allclose(out[:, 3:].sum(axis=1), pred[:, 1] - P, atol=1e-12, rtol=1e-10), 'Psi component add-back failure')
    return out
COMPONENTS = ('M_dilution', 'M_signed_drive', 'M_numeric_branch_residual', 'Psi_dilution', 'Psi_response_drive', 'Psi_support_drive', 'Psi_G_consistency', 'Psi_numeric_branch_residual')
