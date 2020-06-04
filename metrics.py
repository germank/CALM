# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python
import json
import numpy as np


def compute_metrics(filename):
    parsed_results = parse_results(filename)
    results = extract_measures(parsed_results)
    return results

def get_loss_history(parsed_results):
    losses = np.array([r['loss'] for r in parsed_results])
    return losses

def get_switch_times(parsed_results):
    sequences = np.array([r['sequence'] for r in parsed_results])
    sequences_ids, switch_times = np.unique(sequences, return_index=True)
    return switch_times

def get_domain_history(parsed_results, switch_times):
    domain_history = np.array([parsed_results[t]['domain'] for t in switch_times])
    return domain_history

def get_domain_names(parsed_results):
    domain_names = {}
    for r in parsed_results:
        domain_names[r['domain']] = r['domain_name']
    return domain_names

def parse_results(filename):
    with open(filename) as f:
        parsed_results = []
        for line in f:
            parsed_line = json.loads(line)
            if parsed_line['sequence'] >= 50:
                parsed_results.append(parsed_line)
    if not parsed_results or parsed_line['sequence'] <  99:
        return []
    return parsed_results

def calc_surprisal_intensity(losses, _):
    return np.mean(losses[:10])

def calc_surprisal_duration(losses, prev_loss):
    duration = 0
    #loss_avg = np.average(losses)
    for i in range(len(losses)):
        if losses[i] < prev_loss:
            break
        duration += 1
    return duration

def get_mean_loss_by_domain(parsed_results):
    losses_by_domain = stitch_losses_by_domain(parsed_results)
    domain_names = get_domain_names(parsed_results)
    return {domain_names[d]: np.mean(losses) for d, losses in losses_by_domain.items()}

def stitch_losses_by_domain(parsed_results):
    loss_history = get_loss_history(parsed_results)
    switch_times = get_switch_times(parsed_results)
    dom_names = get_domain_history(parsed_results, switch_times)
    loss_per_domain = {}
    for i in range(len(switch_times)-1):
        if dom_names[i] not in loss_per_domain:
            loss_per_domain[dom_names[i]] = []
        local_losses = loss_history[switch_times[i]:switch_times[i+1]]
        loss_per_domain[dom_names[i]].extend(local_losses)
    loss_per_domain[dom_names[-1]].extend(loss_history[switch_times[-1]:])
    return loss_per_domain

def get_surprisal_by_domain(parsed_results, surprisal_measure):
    loss_history = get_loss_history(parsed_results)
    switch_times = get_switch_times(parsed_results)
    dom_names = get_domain_history(parsed_results, switch_times)
    real_domain_names = get_domain_names(parsed_results)
    surprisal_per_domain = {}
    avg_surprisal_per_domain = {}
    prev_losses = {}
    for i in range(len(switch_times)-1):
        if dom_names[i] not in surprisal_per_domain:
            surprisal_per_domain[dom_names[i]] = []
        local_losses = loss_history[switch_times[i]:switch_times[i+1]]
        if dom_names[i] in prev_losses:
            surprisal_per_domain[dom_names[i]].append(surprisal_measure(local_losses, prev_losses[dom_names[i]]))
        prev_losses[dom_names[i]] = np.mean(local_losses)
    #surprisal_per_domain[dom_names[-1]].append(surprisal_measure(local_losses))
    all_surprisals = []
    for el in surprisal_per_domain:
        avg_surprisal_per_domain[real_domain_names[el]] = np.average(surprisal_per_domain[el])
        all_surprisals.extend(surprisal_per_domain[el])
    gen_avg_surprisal = np.average(all_surprisals)
    return gen_avg_surprisal, avg_surprisal_per_domain

def extract_measures(parsed_results):
    loss_history = get_loss_history(parsed_results)
    if len(loss_history) == 0:
        return {}
    loss_per_domain = stitch_losses_by_domain(parsed_results)
    loss = np.mean(loss_history)
    total_pp = np.exp(loss) if loss < 20 else float('inf')
    results = {'loss': loss, 'total_pp': total_pp}
    loss_by_domain = get_mean_loss_by_domain(parsed_results)
    surprisal_intensity, surprisal_intensity_per_domain = get_surprisal_by_domain(parsed_results, calc_surprisal_intensity)
    results['ppl@sw'] = np.exp(surprisal_intensity)
    surprisal_duration, surprisal_duration_per_domain = get_surprisal_by_domain(parsed_results, calc_surprisal_duration)
    results['rec'] = surprisal_duration
    for domain, dloss in loss_by_domain.items():
        results[f'loss_{domain}'] = dloss
        results[f'total_pp_{domain}'] = np.exp(dloss) if loss < 20 else float('inf')
    for domain, dsurprisal in surprisal_intensity_per_domain.items():
        results[f'ppl@sw_{domain}'] = np.exp(dsurprisal)
    for domain, dsurprisal in surprisal_duration_per_domain.items():
        results[f'rec_{domain}'] = dsurprisal
    return results
