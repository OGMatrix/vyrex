"""
VYREX Training Log Analyzer
Parses rlgym-ppo output.log and produces a detailed training report.
"""
import re
import statistics
from collections import defaultdict

LOG_PATH = r"G:\Dev\ai\vyrex\wandb\run-20260215_115549-9ue9rans\files\output.log"

def parse_european_number(s):
    """Convert European number format (dots=thousands, comma=decimal) to float."""
    s = s.strip()
    if s.lower() == 'nan':
        return float('nan')
    # Remove dots (thousands separators), replace comma with dot (decimal)
    s = s.replace('.', '').replace(',', '.')
    return float(s)

def parse_log(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into iteration blocks
    blocks = re.split(r'-+BEGIN ITERATION REPORT-+', content)
    
    iterations = []
    for block in blocks[1:]:  # skip before first block
        data = {}
        
        def extract(pattern, key, block=block):
            m = re.search(pattern, block)
            if m:
                data[key] = parse_european_number(m.group(1))
        
        extract(r'Policy Reward:\s*(.+)', 'reward')
        extract(r'Policy Entropy:\s*(.+)', 'entropy')
        extract(r'Value Function Loss:\s*(.+)', 'vf_loss')
        extract(r'Mean KL Divergence:\s*(.+)', 'kl_div')
        extract(r'SB3 Clip Fraction:\s*(.+)', 'clip_frac')
        extract(r'Policy Update Magnitude:\s*(.+)', 'policy_update_mag')
        extract(r'Value Function Update Magnitude:\s*(.+)', 'vf_update_mag')
        extract(r'Collected Steps per Second:\s*(.+)', 'collected_sps')
        extract(r'Overall Steps per Second:\s*(.+)', 'overall_sps')
        extract(r'Cumulative Model Updates:\s*(.+)', 'cum_updates')
        extract(r'Cumulative Timesteps:\s*(.+)', 'cum_timesteps')
        extract(r'Timesteps Collected:\s*(.+)', 'steps_collected')
        extract(r'Timestep Collection Time:\s*(.+)', 'collection_time')
        extract(r'Timestep Consumption Time:\s*(.+)', 'consumption_time')
        extract(r'Total Iteration Time:\s*(.+)', 'iter_time')
        
        if 'cum_timesteps' in data:
            iterations.append(data)
    
    return iterations

def safe_list(values):
    """Filter out NaN values."""
    return [v for v in values if v == v]  # NaN != NaN

def fmt(v, decimals=4):
    if v != v: return "NaN"
    if abs(v) >= 1_000_000:
        return f"{v:,.0f}"
    if abs(v) >= 1000:
        return f"{v:,.1f}"
    return f"{v:.{decimals}f}"

def phase_stats(iters, label):
    if not iters:
        return f"  {label}: No data\n"
    rewards = safe_list([i.get('reward', float('nan')) for i in iters])
    vf_losses = safe_list([i.get('vf_loss', float('nan')) for i in iters])
    entropies = safe_list([i.get('entropy', float('nan')) for i in iters])
    sps_vals = safe_list([i.get('overall_sps', float('nan')) for i in iters])
    csps_vals = safe_list([i.get('collected_sps', float('nan')) for i in iters])
    policy_mag = safe_list([i.get('policy_update_mag', float('nan')) for i in iters])
    vf_mag = safe_list([i.get('vf_update_mag', float('nan')) for i in iters])
    kl_divs = safe_list([i.get('kl_div', float('nan')) for i in iters])
    clip_fracs = safe_list([i.get('clip_frac', float('nan')) for i in iters])
    
    start_ts = iters[0].get('cum_timesteps', 0)
    end_ts = iters[-1].get('cum_timesteps', 0)
    
    lines = []
    lines.append(f"  {label} (steps {start_ts/1e6:.1f}M - {end_ts/1e6:.1f}M, {len(iters)} iterations):")
    if rewards:
        lines.append(f"    Avg Reward:       {statistics.mean(rewards):.4f}  (min: {min(rewards):.4f}, max: {max(rewards):.4f})")
    if vf_losses:
        lines.append(f"    Avg VF Loss:      {statistics.mean(vf_losses):.5f}  (min: {min(vf_losses):.5f}, max: {max(vf_losses):.5f})")
    if entropies:
        lines.append(f"    Avg Entropy:      {statistics.mean(entropies):.5f}  (min: {min(entropies):.5f}, max: {max(entropies):.5f})")
    if sps_vals:
        lines.append(f"    Avg SPS:          {statistics.mean(sps_vals):,.0f}  (min: {min(sps_vals):,.0f}, max: {max(sps_vals):,.0f})")
    if csps_vals:
        lines.append(f"    Avg Coll SPS:     {statistics.mean(csps_vals):,.0f}")
    if policy_mag:
        lines.append(f"    Avg Policy Mag:   {statistics.mean(policy_mag):.5f}")
    if vf_mag:
        lines.append(f"    Avg VF Mag:       {statistics.mean(vf_mag):.5f}")
    if kl_divs:
        lines.append(f"    Avg KL Div:       {statistics.mean(kl_divs):.6f}")
    if clip_fracs:
        lines.append(f"    Avg Clip Frac:    {statistics.mean(clip_fracs):.5f}")
    return '\n'.join(lines) + '\n'

def compute_linear_regression(xs, ys):
    """Simple linear regression: returns slope, intercept."""
    n = len(xs)
    if n < 2: return 0, 0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    if ss_xx == 0: return 0, y_mean
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    return slope, intercept

def main():
    print("Parsing log file...")
    iterations = parse_log(LOG_PATH)
    print(f"Parsed {len(iterations)} iterations.\n")
    
    if not iterations:
        print("No iterations found!")
        return
    
    # Extract all series
    rewards = [i.get('reward', float('nan')) for i in iterations]
    vf_losses = [i.get('vf_loss', float('nan')) for i in iterations]
    entropies = [i.get('entropy', float('nan')) for i in iterations]
    overall_sps = [i.get('overall_sps', float('nan')) for i in iterations]
    collected_sps = [i.get('collected_sps', float('nan')) for i in iterations]
    cum_timesteps = [i.get('cum_timesteps', 0) for i in iterations]
    cum_updates = [i.get('cum_updates', 0) for i in iterations]
    policy_mag = [i.get('policy_update_mag', float('nan')) for i in iterations]
    vf_mag = [i.get('vf_update_mag', float('nan')) for i in iterations]
    iter_times = [i.get('iter_time', float('nan')) for i in iterations]
    kl_divs = [i.get('kl_div', float('nan')) for i in iterations]
    clip_fracs = [i.get('clip_frac', float('nan')) for i in iterations]
    
    total_timesteps = cum_timesteps[-1]
    total_iterations = len(iterations)
    total_time_s = sum(safe_list(iter_times))
    total_time_m = total_time_s / 60
    total_time_h = total_time_s / 3600
    
    safe_sps = safe_list(overall_sps)
    safe_csps = safe_list(collected_sps)
    safe_rewards = safe_list(rewards)
    safe_vf = safe_list(vf_losses)
    safe_ent = safe_list(entropies)
    safe_pmag = safe_list(policy_mag)
    safe_vmag = safe_list(vf_mag)
    safe_kl = safe_list(kl_divs)
    safe_clip = safe_list(clip_fracs)
    
    # ============================================================
    # OVERALL STATS
    # ============================================================
    print("=" * 70)
    print("  VYREX TRAINING LOG — FULL ANALYSIS REPORT")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("  1. OVERALL STATS")
    print("=" * 70)
    print(f"  Total Timesteps:         {total_timesteps:,.0f}")
    print(f"  Total Iterations:        {total_iterations}")
    print(f"  Total Model Updates:     {cum_updates[-1]:,.0f}")
    print(f"  Total Training Time:     {total_time_h:.2f} hours ({total_time_m:.1f} min)")
    print(f"  Steps per Iteration:     ~{total_timesteps / total_iterations:,.0f}")
    print()
    print(f"  Overall SPS (avg):       {statistics.mean(safe_sps):,.0f}")
    print(f"  Overall SPS (peak):      {max(safe_sps):,.0f}")
    print(f"  Overall SPS (min):       {min(safe_sps):,.0f}")
    print(f"  Overall SPS (median):    {statistics.median(safe_sps):,.0f}")
    print(f"  Overall SPS (stdev):     {statistics.stdev(safe_sps):,.0f}")
    print()
    print(f"  Collected SPS (avg):     {statistics.mean(safe_csps):,.0f}")
    print(f"  Collected SPS (peak):    {max(safe_csps):,.0f}")
    print(f"  Collected SPS (min):     {min(safe_csps):,.0f}")
    
    # ============================================================
    # REWARD TRAJECTORY
    # ============================================================
    print("\n" + "=" * 70)
    print("  2. REWARD TRAJECTORY")
    print("=" * 70)
    
    reward_phases = [
        ("0-1M", 0, 1e6),
        ("1M-5M", 1e6, 5e6),
        ("5M-10M", 5e6, 10e6),
        ("10M-20M", 10e6, 20e6),
        ("20M-40M", 20e6, 40e6),
        ("40M-60M", 40e6, 60e6),
        ("60M-80M", 60e6, 80e6),
        ("80M-89.3M", 80e6, 90e6),
    ]
    
    print()
    for label, lo, hi in reward_phases:
        phase_iters = [it for it in iterations if lo <= it.get('cum_timesteps', 0) < hi]
        phase_rewards = safe_list([it.get('reward', float('nan')) for it in phase_iters])
        if phase_rewards:
            avg_r = statistics.mean(phase_rewards)
            min_r = min(phase_rewards)
            max_r = max(phase_rewards)
            print(f"  {label:>12s}: avg={avg_r:8.4f}  min={min_r:8.4f}  max={max_r:8.4f}  n={len(phase_rewards)}")
        else:
            print(f"  {label:>12s}: No data")
    
    # Best/worst reward
    best_idx = max(range(len(safe_rewards)), key=lambda i: safe_rewards[i])
    worst_idx = min(range(len(safe_rewards)), key=lambda i: safe_rewards[i])
    # Map back to iterations
    valid_iters = [(i, it) for i, it in enumerate(iterations) if it.get('reward', float('nan')) == it.get('reward', float('nan'))]
    best_iter = valid_iters[best_idx][1]
    worst_iter = valid_iters[worst_idx][1]
    
    print()
    print(f"  Best Reward:   {max(safe_rewards):.4f} at step {best_iter['cum_timesteps']:,.0f} ({best_iter['cum_timesteps']/1e6:.1f}M)")
    print(f"  Worst Reward:  {min(safe_rewards):.4f} at step {worst_iter['cum_timesteps']:,.0f} ({worst_iter['cum_timesteps']/1e6:.1f}M)")
    
    # Trend
    valid_steps = [it['cum_timesteps'] for _, it in valid_iters]
    valid_rewards = [it['reward'] for _, it in valid_iters]
    slope, intercept = compute_linear_regression(valid_steps, valid_rewards)
    direction = "INCREASING" if slope > 0 else "DECREASING" if slope < 0 else "FLAT"
    reward_per_million = slope * 1e6
    print()
    print(f"  Overall Trend: {direction}")
    print(f"  Rate: {reward_per_million:+.4f} reward per million steps")
    print(f"  Start (fitted): {intercept:.4f}, End (fitted): {intercept + slope * total_timesteps:.4f}")
    
    # Moving average reward (window=20)
    window = 50
    if len(safe_rewards) >= window:
        first_ma = statistics.mean(safe_rewards[:window])
        last_ma = statistics.mean(safe_rewards[-window:])
        mid_ma = statistics.mean(safe_rewards[len(safe_rewards)//2 - window//2 : len(safe_rewards)//2 + window//2])
        print()
        print(f"  Moving Avg (w={window}):")
        print(f"    First {window}:  {first_ma:.4f}")
        print(f"    Middle {window}: {mid_ma:.4f}")
        print(f"    Last {window}:   {last_ma:.4f}")
    
    # ============================================================
    # VALUE FUNCTION LOSS
    # ============================================================
    print("\n" + "=" * 70)
    print("  3. VALUE FUNCTION LOSS")
    print("=" * 70)
    
    vf_valid = [(it['cum_timesteps'], it['vf_loss']) for it in iterations if it.get('vf_loss', float('nan')) == it.get('vf_loss', float('nan'))]
    
    if vf_valid:
        vf_steps = [v[0] for v in vf_valid]
        vf_vals = [v[1] for v in vf_valid]
        
        print(f"\n  Total VF Loss data points: {len(vf_valid)}")
        print(f"  Starting VF Loss (first 10): {statistics.mean(vf_vals[:10]):.6f}")
        print(f"  Ending VF Loss (last 10):    {statistics.mean(vf_vals[-10:]):.6f}")
        print(f"  Peak VF Loss:                {max(vf_vals):.6f} at step {vf_steps[vf_vals.index(max(vf_vals))]/1e6:.1f}M")
        print(f"  Min VF Loss:                 {min(vf_vals):.6f}")
        print(f"  Overall Avg:                 {statistics.mean(vf_vals):.6f}")
        
        # VF loss per reward phase
        print("\n  VF Loss by Phase:")
        for label, lo, hi in reward_phases:
            phase_vf = [it['vf_loss'] for it in iterations if lo <= it.get('cum_timesteps', 0) < hi and it.get('vf_loss', float('nan')) == it.get('vf_loss', float('nan'))]
            if phase_vf:
                print(f"    {label:>12s}: avg={statistics.mean(phase_vf):.6f}  min={min(phase_vf):.6f}  max={max(phase_vf):.6f}")
        
        # Convergence check
        last_100_vf = vf_vals[-100:] if len(vf_vals) >= 100 else vf_vals
        last_50_vf = vf_vals[-50:] if len(vf_vals) >= 50 else vf_vals
        vf_std_last = statistics.stdev(last_100_vf) if len(last_100_vf) > 1 else 0
        print(f"\n  Convergence Analysis:")
        print(f"    Last 100 iters avg: {statistics.mean(last_100_vf):.6f}, stdev: {vf_std_last:.6f}")
        print(f"    Last 50 iters avg:  {statistics.mean(last_50_vf):.6f}")
        converged = vf_std_last < 0.01
        print(f"    Converged: {'YES' if converged else 'NO'} (stdev {'<' if converged else '>'} 0.01)")
    
    # ============================================================
    # POLICY ENTROPY
    # ============================================================
    print("\n" + "=" * 70)
    print("  4. POLICY ENTROPY")
    print("=" * 70)
    
    ent_valid = [(it['cum_timesteps'], it['entropy']) for it in iterations if it.get('entropy', float('nan')) == it.get('entropy', float('nan'))]
    ent_steps = [v[0] for v in ent_valid]
    ent_vals = [v[1] for v in ent_valid]
    
    print(f"\n  Starting Entropy (first 10):   {statistics.mean(ent_vals[:10]):.5f}")
    print(f"  Ending Entropy (last 10):      {statistics.mean(ent_vals[-10:]):.5f}")
    print(f"  Min Entropy:                   {min(ent_vals):.5f}")
    print(f"  Max Entropy:                   {max(ent_vals):.5f}")
    
    total_decay = ent_vals[0] - ent_vals[-1]
    decay_rate = total_decay / (ent_steps[-1] / 1e6) if ent_steps[-1] > 0 else 0
    print(f"\n  Total Decay:     {total_decay:.5f}")
    print(f"  Decay Rate:      {decay_rate:.5f} per million steps")
    
    # Entropy per phase
    print("\n  Entropy by Phase:")
    for label, lo, hi in reward_phases:
        phase_ent = [it['entropy'] for it in iterations if lo <= it.get('cum_timesteps', 0) < hi and it.get('entropy', float('nan')) == it.get('entropy', float('nan'))]
        if phase_ent:
            print(f"    {label:>12s}: avg={statistics.mean(phase_ent):.5f}  min={min(phase_ent):.5f}  max={max(phase_ent):.5f}")
    
    # Health assessment
    ent_ratio = ent_vals[-1] / ent_vals[0] if ent_vals[0] != 0 else 0
    print(f"\n  Entropy Retention: {ent_ratio*100:.2f}% of initial")
    if ent_ratio > 0.90:
        print(f"  Assessment: VERY SLOW DECAY — model is barely specializing")
    elif ent_ratio > 0.70:
        print(f"  Assessment: HEALTHY — gradual decay with good exploration")
    elif ent_ratio > 0.40:
        print(f"  Assessment: MODERATE DECAY — acceptable but monitor")
    else:
        print(f"  Assessment: COLLAPSING — entropy too low, risk of mode collapse")
    
    # ============================================================
    # UPDATE MAGNITUDES
    # ============================================================
    print("\n" + "=" * 70)
    print("  5. UPDATE MAGNITUDES")
    print("=" * 70)
    
    print(f"\n  Policy Update Magnitude:")
    print(f"    Avg:    {statistics.mean(safe_pmag):.5f}")
    print(f"    Min:    {min(safe_pmag):.5f}")
    print(f"    Max:    {max(safe_pmag):.5f}")
    print(f"    Stdev:  {statistics.stdev(safe_pmag):.5f}")
    print(f"    First 10 avg: {statistics.mean(safe_pmag[:10]):.5f}")
    print(f"    Last 10 avg:  {statistics.mean(safe_pmag[-10:]):.5f}")
    
    # Detect spikes (>3 stdev from mean)
    pmag_mean = statistics.mean(safe_pmag)
    pmag_std = statistics.stdev(safe_pmag)
    pmag_spikes = [(i, it['cum_timesteps'], it['policy_update_mag']) 
                   for i, it in enumerate(iterations) 
                   if it.get('policy_update_mag', float('nan')) == it.get('policy_update_mag', float('nan'))
                   and abs(it['policy_update_mag'] - pmag_mean) > 3 * pmag_std]
    print(f"    Spikes (>3σ): {len(pmag_spikes)}")
    if pmag_spikes:
        for idx, ts, val in pmag_spikes[:10]:
            print(f"      iter {idx}: step {ts/1e6:.1f}M, mag={val:.5f}")
    
    print(f"\n  Value Function Update Magnitude:")
    print(f"    Avg:    {statistics.mean(safe_vmag):.5f}")
    print(f"    Min:    {min(safe_vmag):.5f}")
    print(f"    Max:    {max(safe_vmag):.5f}")
    print(f"    Stdev:  {statistics.stdev(safe_vmag):.5f}")
    print(f"    First 10 avg: {statistics.mean(safe_vmag[:10]):.5f}")
    print(f"    Last 10 avg:  {statistics.mean(safe_vmag[-10:]):.5f}")
    
    vmag_mean = statistics.mean(safe_vmag)
    vmag_std = statistics.stdev(safe_vmag)
    vmag_spikes = [(i, it['cum_timesteps'], it['vf_update_mag']) 
                   for i, it in enumerate(iterations) 
                   if it.get('vf_update_mag', float('nan')) == it.get('vf_update_mag', float('nan'))
                   and abs(it['vf_update_mag'] - vmag_mean) > 3 * vmag_std]
    print(f"    Spikes (>3σ): {len(vmag_spikes)}")
    if vmag_spikes:
        for idx, ts, val in vmag_spikes[:10]:
            print(f"      iter {idx}: step {ts/1e6:.1f}M, mag={val:.5f}")
    
    # ============================================================
    # SPS ANALYSIS
    # ============================================================
    print("\n" + "=" * 70)
    print("  6. SPS ANALYSIS")
    print("=" * 70)
    
    # Divide into 10 segments
    seg_size = len(iterations) // 10
    print(f"\n  SPS over time (10 segments):")
    for seg in range(10):
        start = seg * seg_size
        end = start + seg_size if seg < 9 else len(iterations)
        seg_sps = safe_list([it.get('overall_sps', float('nan')) for it in iterations[start:end]])
        seg_csps = safe_list([it.get('collected_sps', float('nan')) for it in iterations[start:end]])
        if seg_sps:
            ts_start = iterations[start]['cum_timesteps'] / 1e6
            ts_end = iterations[min(end-1, len(iterations)-1)]['cum_timesteps'] / 1e6
            print(f"    Seg {seg+1:2d} ({ts_start:5.1f}M-{ts_end:5.1f}M): "
                  f"SPS avg={statistics.mean(seg_sps):,.0f}  Coll.SPS avg={statistics.mean(seg_csps):,.0f}")
    
    # Degradation detection
    first_quarter_sps = safe_list([it.get('overall_sps', float('nan')) for it in iterations[:len(iterations)//4]])
    last_quarter_sps = safe_list([it.get('overall_sps', float('nan')) for it in iterations[3*len(iterations)//4:]])
    if first_quarter_sps and last_quarter_sps:
        sps_change = (statistics.mean(last_quarter_sps) - statistics.mean(first_quarter_sps)) / statistics.mean(first_quarter_sps) * 100
        print(f"\n  SPS Change (first vs last quarter): {sps_change:+.1f}%")
        if sps_change < -10:
            print(f"  WARNING: Significant SPS degradation detected!")
        elif sps_change < -5:
            print(f"  Note: Mild SPS degradation.")
        else:
            print(f"  SPS is STABLE.")
    
    # ============================================================
    # KL DIVERGENCE & CLIP FRACTION
    # ============================================================
    print("\n" + "=" * 70)
    print("  7. KL DIVERGENCE & CLIP FRACTION")
    print("=" * 70)
    
    if safe_kl:
        print(f"\n  KL Divergence:")
        print(f"    Avg:    {statistics.mean(safe_kl):.6f}")
        print(f"    Max:    {max(safe_kl):.6f}")
        print(f"    Last 10 avg: {statistics.mean(safe_kl[-10:]):.6f}")
    
    if safe_clip:
        print(f"\n  Clip Fraction:")
        print(f"    Avg:    {statistics.mean(safe_clip):.5f}")
        print(f"    Max:    {max(safe_clip):.5f}")
        print(f"    Last 10 avg: {statistics.mean(safe_clip[-10:]):.5f}")
    
    # ============================================================
    # PHASE ANALYSIS (5 phases)
    # ============================================================
    print("\n" + "=" * 70)
    print("  8. PHASE ANALYSIS (5 Equal Phases)")
    print("=" * 70)
    print()
    
    phase_size = len(iterations) // 5
    for p in range(5):
        start = p * phase_size
        end = start + phase_size if p < 4 else len(iterations)
        phase_iters = iterations[start:end]
        print(phase_stats(phase_iters, f"Phase {p+1}"))
    
    # ============================================================
    # KEY OBSERVATIONS & RECOMMENDATIONS
    # ============================================================
    print("=" * 70)
    print("  9. KEY OBSERVATIONS & RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    observations = []
    
    # 1. Learning check
    first_50_reward = statistics.mean(safe_list([it.get('reward', float('nan')) for it in iterations[:50]]))
    last_50_reward = statistics.mean(safe_list([it.get('reward', float('nan')) for it in iterations[-50:]]))
    reward_change = last_50_reward - first_50_reward
    if reward_change > 0.5:
        observations.append(f"[POSITIVE] Model IS learning. Reward improved by {reward_change:.4f} from first to last 50 iterations.")
    elif reward_change > 0:
        observations.append(f"[NEUTRAL] Marginal reward improvement ({reward_change:.4f}). Learning is slow.")
    else:
        observations.append(f"[CONCERNING] Reward has NOT improved (change: {reward_change:.4f}). Model may not be learning effectively.")
    
    # 2. Entropy health
    observations.append(f"[INFO] Entropy retention: {ent_ratio*100:.2f}% — {'Healthy' if ent_ratio > 0.7 else 'Monitor closely' if ent_ratio > 0.4 else 'DANGER: Collapsing'}.")
    
    # 3. VF Loss
    if vf_valid:
        vf_final = statistics.mean(vf_vals[-50:])
        vf_peak = max(vf_vals)
        if vf_final < 0.01:
            observations.append(f"[POSITIVE] VF Loss has converged to {vf_final:.6f}. Value function is well-fit.")
        elif vf_final < 0.1:
            observations.append(f"[NEUTRAL] VF Loss at {vf_final:.6f} — still converging.")
        else:
            observations.append(f"[CONCERNING] VF Loss remains high at {vf_final:.6f}.")
    
    # 4. Reward variance
    reward_std = statistics.stdev(safe_rewards) if len(safe_rewards) > 1 else 0
    observations.append(f"[INFO] Reward stdev: {reward_std:.4f} — {'High variance' if reward_std > 2 else 'Moderate variance' if reward_std > 1 else 'Low variance'}.")
    
    # 5. SPS stability
    if sps_change < -10:
        observations.append(f"[WARNING] SPS degraded by {abs(sps_change):.1f}% over training — possible memory leak or increased computation.")
    else:
        observations.append(f"[POSITIVE] SPS remained stable (change: {sps_change:+.1f}%).")
    
    # 6. Reward hacking check
    if reward_change > 5 and ent_ratio < 0.5:
        observations.append("[WARNING] Possible reward hacking: large reward gain with entropy collapse.")
    else:
        observations.append("[POSITIVE] No signs of reward hacking detected.")
    
    # 7. Update magnitude stability
    if len(pmag_spikes) > 5:
        observations.append(f"[WARNING] {len(pmag_spikes)} policy update magnitude spikes detected — training may be unstable.")
    else:
        observations.append(f"[POSITIVE] Policy updates are stable ({len(pmag_spikes)} spikes).")
    
    # 8. KL divergence check
    if safe_kl:
        max_kl = max(safe_kl)
        if max_kl > 0.05:
            observations.append(f"[WARNING] KL divergence peaked at {max_kl:.6f} — policy updates may be too aggressive.")
        else:
            observations.append(f"[POSITIVE] KL divergence well-controlled (max: {max_kl:.6f}).")
    
    # 9. Plateau detection
    # Check if reward has plateaued in the last 30% of training
    last_30_pct = iterations[int(len(iterations)*0.7):]
    last_30_rewards = safe_list([it.get('reward', float('nan')) for it in last_30_pct])
    if last_30_rewards:
        last_30_slope, _ = compute_linear_regression(
            list(range(len(last_30_rewards))), last_30_rewards)
        if abs(last_30_slope) < 0.001:
            observations.append("[INFO] Reward appears PLATEAUED in the last 30% of training.")
        elif last_30_slope > 0:
            observations.append(f"[POSITIVE] Reward still INCREASING in the last 30% (slope: {last_30_slope:.5f}/iter).")
        else:
            observations.append(f"[CONCERNING] Reward DECLINING in the last 30% (slope: {last_30_slope:.5f}/iter).")
    
    # 10. Efficiency
    effective_sps = total_timesteps / total_time_s if total_time_s > 0 else 0
    observations.append(f"[INFO] Effective throughput: {effective_sps:,.0f} steps/sec over {total_time_h:.2f} hours.")
    
    for obs in observations:
        print(f"  {obs}")
    
    # Recommendations
    print()
    print("  RECOMMENDATIONS:")
    print("  " + "-" * 50)
    
    if reward_change <= 0:
        print("  - Consider adjusting reward function — model isn't improving.")
    if ent_ratio > 0.95:
        print("  - Entropy is barely decaying. Consider increasing learning rate or reducing entropy coefficient.")
    elif ent_ratio < 0.4:
        print("  - Entropy is collapsing. Increase entropy coefficient to maintain exploration.")
    if vf_valid and vf_final > 0.1:
        print("  - VF Loss is high. Consider increasing value function learning rate or network capacity.")
    if sps_change < -10:
        print("  - SPS degradation detected. Check for memory leaks or consider reducing batch size.")
    if reward_std > 2:
        print("  - High reward variance. Consider reward normalization or clipping.")
    if abs(last_30_slope) < 0.001 and len(last_30_rewards) > 50:
        print("  - Reward has plateaued. Consider: adjusting LR schedule, modifying reward function, or increasing model capacity.")
    if max(safe_kl) > 0.05:
        print("  - KL divergence too high. Reduce learning rate or increase clip range.")
    
    print(f"  - Continue training — at {total_timesteps/1e6:.1f}M of 2B target, only {total_timesteps/2e9*100:.1f}% complete.")
    print(f"  - ETA to 2B steps: ~{(2e9 - total_timesteps) / effective_sps / 3600:.0f} hours at current throughput.")
    
    print("\n" + "=" * 70)
    print("  END OF REPORT")
    print("=" * 70)

if __name__ == '__main__':
    main()
