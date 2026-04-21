"""Quick single-layer benchmark for fast iteration."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from benchmark import build_modules, load_flux_data, benchmark_kernel, compute_gemm_ops, check_correctness, COSINE_THRESHOLDS, GROUP_SIZE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', default='attn_to_qkv', choices=['attn_to_qkv','attn_to_out','ff_up','ff_down'])
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--no-correctness', action='store_true')
    args = parser.parse_args()

    ref, sol = build_modules()
    data = load_flux_data(os.path.join(SCRIPT_DIR, 'flux_dump'))
    weights, activations = data['weights'], data['activations']

    name = args.layer
    w, a = weights[name], activations[name]
    M, K, N = a.shape[0], a.shape[1], w.shape[0]
    gs = GROUP_SIZE

    # Offline weight quantization
    sys.path.insert(0, os.path.join(SCRIPT_DIR, 'your_solution'))
    import quantize as qmod
    result = qmod.quantize_weights(w, group_size=gs)
    wgt_packed, wgt_scales = result['quantized_weights'], result['scales']

    if not args.no_correctness:
        passed, cos, _ = check_correctness(ref, sol, a, w, wgt_packed, wgt_scales, gs, COSINE_THRESHOLDS.get(name))
        status = 'PASS' if passed else 'FAIL'
        print(f'  {name}: cosine={cos:.6f} [{status}]')
        if not passed:
            print('CORRECTNESS FAILED'); return

    sol_act_p, sol_act_s = sol.quantize_int4(a, gs)
    gemm_time = benchmark_kernel(
        lambda ap,wp,as_,ws,g: sol.gemm_int4(ap,wp,as_,ws,g),
        [sol_act_p, wgt_packed, sol_act_s, wgt_scales, gs],
        warmup=args.warmup, iters=args.iters)
    tops = compute_gemm_ops(M, N, K) / gemm_time / 1e12
    print(f'  {name} ({M}x{N}x{K}): {tops:.1f} TOPs  ({gemm_time*1e6:.1f} us)')

if __name__ == '__main__':
    main()
