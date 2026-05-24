#!/usr/bin/env python3
"""
Synthetic Table Tennis Serve Dataset Generator (~500 rows)
Chopper's perspective: serves are designed to set up chop rallies or draw errors.
"""

import pandas as pd
import numpy as np
from numpy.random import default_rng

rng = default_rng(42)

# ── Categorical options ─────────────────────────────────────────────────────
SERVE_TYPES       = ['pendulum', 'reverse_pendulum', 'tomahawk',
                     'short_backspin', 'backhand', 'no_spin_disguised']
SPIN_TYPES        = ['backspin', 'sidespin-back', 'sidespin-top', 'float/no-spin']
SERVE_LENGTHS     = ['short', 'half_long', 'long']
PLACEMENT_ZONES   = ['wide_FH', 'middle_FH', 'elbow', 'middle_BH', 'wide_BH']
TOSS_HEIGHTS      = ['low', 'medium', 'high']
CONTACT_POINTS    = ['hip_level_forehand', 'hip_level_backhand',
                     'body_center', 'low_table_level',
                     'extended_forehand', 'high_forehand']
RETURN_TYPES      = ['push', 'flip', 'loop', 'banana_flip', 'pop_up', 'miss', 'net']
RETURN_PLACEMENTS = ['short_FH', 'short_middle', 'wide_BH',
                     'wide_FH', 'deep_BH', 'deep_FH']

# ── Opponent pool ───────────────────────────────────────────────────────────
OPPONENTS = {
    1:  ('intermediate', 'looper'),
    2:  ('advanced',     'looper'),
    3:  ('intermediate', 'allround'),
    4:  ('beginner',     'blocker'),
    5:  ('advanced',     'defender'),
    6:  ('competitive',  'looper'),
    7:  ('intermediate', 'allround'),
    8:  ('competitive',  'blocker'),
    9:  ('beginner',     'allround'),
    10: ('advanced',     'allround'),
}

# ── Serve mechanics mappings ────────────────────────────────────────────────

# serve_type → (spin_options, weights_or_None)
SERVE_SPIN = {
    'pendulum':          (['backspin', 'backspin', 'sidespin-back'], None),
    'reverse_pendulum':  (['sidespin-back', 'sidespin-top', 'sidespin-back'], None),
    'tomahawk':          (['sidespin-top', 'sidespin-top', 'sidespin-back'], None),
    'short_backspin':    (['backspin', 'backspin', 'backspin', 'float/no-spin'],
                          [0.55, 0.25, 0.12, 0.08]),
    'backhand':          (['backspin', 'float/no-spin', 'sidespin-back'],
                          [0.45, 0.30, 0.25]),
    'no_spin_disguised': (['float/no-spin', 'float/no-spin', 'backspin'],
                          [0.45, 0.40, 0.15]),
}

# [P(short), P(half_long), P(long)]
SERVE_LENGTH_PROBS = {
    'pendulum':          [0.45, 0.32, 0.23],
    'reverse_pendulum':  [0.22, 0.33, 0.45],
    'tomahawk':          [0.18, 0.27, 0.55],
    'short_backspin':    [0.62, 0.26, 0.12],
    'backhand':          [0.35, 0.38, 0.27],
    'no_spin_disguised': [0.55, 0.30, 0.15],
}

TOSS_HEIGHT_PROBS = {
    'pendulum':          [0.55, 0.35, 0.10],   # [low, medium, high]
    'reverse_pendulum':  [0.20, 0.50, 0.30],
    'tomahawk':          [0.10, 0.35, 0.55],
    'short_backspin':    [0.65, 0.30, 0.05],
    'backhand':          [0.55, 0.35, 0.10],
    'no_spin_disguised': [0.70, 0.25, 0.05],
}

# (contact_options, weights)
CONTACT_PROBS = {
    'pendulum':          (['hip_level_forehand', 'low_table_level', 'extended_forehand'],
                          [0.50, 0.30, 0.20]),
    'reverse_pendulum':  (['extended_forehand', 'hip_level_forehand', 'body_center'],
                          [0.45, 0.35, 0.20]),
    'tomahawk':          (['high_forehand', 'extended_forehand', 'hip_level_forehand'],
                          [0.50, 0.30, 0.20]),
    'short_backspin':    (['low_table_level', 'hip_level_forehand', 'body_center'],
                          [0.55, 0.30, 0.15]),
    'backhand':          (['hip_level_backhand', 'body_center', 'low_table_level'],
                          [0.55, 0.28, 0.17]),
    'no_spin_disguised': (['low_table_level', 'hip_level_forehand', 'body_center'],
                          [0.45, 0.35, 0.20]),
}

# Intended setup from chopper's perspective
INTENDED_SETUP_PROBS = {
    'pendulum':          {'force_weak_push': 0.38, 'start_chop_rally': 0.35,
                          'force_pop_up': 0.15,    'direct_point': 0.12},
    'reverse_pendulum':  {'force_pop_up': 0.38, 'direct_point': 0.32,
                          'force_weak_push': 0.20, 'start_chop_rally': 0.10},
    'tomahawk':          {'force_pop_up': 0.38, 'direct_point': 0.30,
                          'force_weak_push': 0.22, 'start_chop_rally': 0.10},
    'short_backspin':    {'force_weak_push': 0.52, 'start_chop_rally': 0.30,
                          'direct_point': 0.10,    'force_pop_up': 0.08},
    'backhand':          {'start_chop_rally': 0.45, 'force_weak_push': 0.35,
                          'direct_point': 0.12,    'force_pop_up': 0.08},
    'no_spin_disguised': {'direct_point': 0.45, 'force_pop_up': 0.40,
                          'force_weak_push': 0.10, 'start_chop_rally': 0.05},
}

# Baseline serve type distribution for a defensive chopper
BASE_SERVE_PROBS = {
    'pendulum': 0.28, 'reverse_pendulum': 0.12, 'tomahawk': 0.10,
    'short_backspin': 0.22, 'backhand': 0.16, 'no_spin_disguised': 0.12,
}

# ── Sampling helpers ────────────────────────────────────────────────────────
def norm(arr):
    a = np.array(arr, dtype=float)
    a = np.clip(a, 0.005, None)
    return a / a.sum()

def wc(options, weights=None):
    """Weighted choice from a list."""
    if weights is None:
        return rng.choice(options)
    return rng.choice(options, p=norm(weights))

def dict_wc(d):
    """Weighted choice from a dict of {option: weight}."""
    keys = list(d.keys())
    return rng.choice(keys, p=norm(list(d.values())))

# ── Return type ─────────────────────────────────────────────────────────────
def sample_return_type(spin_type, spin_intensity, serve_length, opp_style, opp_skill):
    # Base distribution: [push, flip, loop, banana_flip, pop_up, miss, net]
    p = np.array([0.28, 0.14, 0.13, 0.08, 0.12, 0.13, 0.12])

    # Spin type influence
    spin_deltas = {
        'backspin':      [ 0.18, -0.05, -0.05, -0.04,  0.04,  0.04, -0.02],
        'sidespin-back': [ 0.04,  0.02,  0.02,  0.02,  0.01, -0.02, -0.04],
        'sidespin-top':  [-0.08,  0.10,  0.10,  0.04, -0.02, -0.05, -0.04],
        'float/no-spin': [-0.10,  0.08,  0.07,  0.03,  0.06, -0.06, -0.03],
    }
    p += np.array(spin_deltas.get(spin_type, [0]*7))

    # Spin intensity
    if spin_intensity == 3:   # heavy → harder to flip, more pushes/errors
        p += [ 0.12, -0.08, -0.05, -0.05,  0.06,  0.05,  0.04]
    elif spin_intensity == 1: # light → easier to attack
        p += [-0.08,  0.07,  0.05,  0.04, -0.03, -0.03, -0.02]

    # Serve length
    if serve_length == 'short':
        p += [ 0.18, -0.05, -0.14,  0.01,  0.05, -0.02, -0.03]
    elif serve_length == 'long':
        p += [-0.12, -0.04,  0.22,  0.04, -0.05, -0.03, -0.02]
    elif serve_length == 'half_long':
        p += [ 0.04,  0.04, -0.03,  0.02,  0.02, -0.04, -0.05]

    # Opponent style
    style_deltas = {
        'looper':   [-0.08,  0.08,  0.10,  0.06, -0.05, -0.04, -0.05],
        'defender': [ 0.18, -0.08, -0.08, -0.05, -0.02,  0.05,  0.00],
        'blocker':  [ 0.10,  0.04, -0.08, -0.04,  0.04, -0.02, -0.04],
        'allround': [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
    }
    p += np.array(style_deltas.get(opp_style, [0]*7))

    # Skill level
    skill_deltas = {
        'beginner':     [-0.04, -0.04, -0.08, -0.04,  0.09,  0.08,  0.05],
        'intermediate': [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
        'advanced':     [ 0.03,  0.03,  0.03,  0.02, -0.04, -0.05, -0.02],
        'competitive':  [ 0.05,  0.05,  0.05,  0.04, -0.07, -0.08, -0.04],
    }
    p += np.array(skill_deltas.get(opp_skill, [0]*7))

    return wc(RETURN_TYPES, p)

# ── Return quality ──────────────────────────────────────────────────────────
def sample_return_quality(return_type, spin_intensity, opp_skill):
    if return_type in ('miss', 'net'):
        return 0
    base = {'push': 1.8, 'flip': 2.0, 'loop': 2.5,
            'banana_flip': 2.2, 'pop_up': 1.0}[return_type]
    # Heavy spin reduces return quality
    base += (2 - spin_intensity) * 0.25
    skill_adj = {'beginner': -0.6, 'intermediate': 0.0,
                 'advanced': 0.4,  'competitive': 0.7}
    base += skill_adj.get(opp_skill, 0)
    base += rng.normal(0, 0.30)
    return int(np.clip(round(base), 1, 3))

# ── Point outcome ───────────────────────────────────────────────────────────
def sample_point_outcome(return_type, return_quality, opp_skill):
    if return_type in ('miss', 'net'):
        return 'won', 'forced_error'

    if return_type == 'pop_up':
        win_p = 0.88
    elif return_quality == 1:
        win_p = 0.68
    elif return_quality == 2:
        win_p = 0.42
    else:
        win_p = 0.22

    skill_adj = {'beginner': 0.15, 'intermediate': 0.0,
                 'advanced': -0.08, 'competitive': -0.16}
    win_p = float(np.clip(win_p + skill_adj.get(opp_skill, 0), 0.05, 0.95))
    won = rng.random() < win_p

    if won:
        if return_quality == 1 or return_type == 'pop_up':
            end = wc(['winner', 'forced_error', 'chop_rally_won'], [0.45, 0.35, 0.20])
        else:
            end = wc(['winner', 'forced_error', 'chop_rally_won'], [0.20, 0.40, 0.40])
    else:
        end = wc(['unforced_error', 'forced_error', 'chop_rally_lost'], [0.20, 0.45, 0.35])

    return ('won' if won else 'lost'), end

# ── Rally type ──────────────────────────────────────────────────────────────
def rally_type_for(return_type):
    if return_type in ('miss', 'net'):
        return 'direct_point'
    if return_type in ('loop', 'flip', 'banana_flip', 'pop_up'):
        return 'attack_rally'
    if return_type == 'push':
        return wc(['chop_rally', 'mixed'], [0.65, 0.35])
    return 'mixed'

# ── Rally length ────────────────────────────────────────────────────────────
def sample_rally_length(return_type, rally_type, end_type):
    if return_type in ('miss', 'net'):
        return 1
    if rally_type == 'chop_rally':
        return int(rng.integers(5, 22))
    if end_type == 'winner':
        return int(rng.integers(2, 5))
    return int(rng.integers(2, 9))

# ── Game state ──────────────────────────────────────────────────────────────
def game_state(s, r):
    if s >= 10 and r >= 10:
        return 'deuce'
    if max(s, r) >= 10:
        return 'game_point'
    if s - r >= 4:
        return 'leading'
    if r - s >= 4:
        return 'trailing'
    return 'neutral'

# ── Serve selection with pressure + streak logic ─────────────────────────
def choose_serve(gs, prev_type, streak):
    probs = dict(BASE_SERVE_PROBS)
    if gs == 'game_point':
        probs['short_backspin'] *= 1.4     # safer under pressure
    elif gs == 'trailing':
        probs['no_spin_disguised'] *= 1.5  # go for surprise when behind
    elif gs == 'deuce':
        probs['short_backspin'] *= 1.2
        probs['pendulum'] *= 1.2

    # Streak: 35% chance to repeat same serve (up to 3 in a row)
    if prev_type and streak < 3 and rng.random() < 0.35:
        return prev_type, streak + 1

    chosen = rng.choice(list(probs.keys()), p=norm(list(probs.values())))
    return chosen, 1

# ── Main generation ─────────────────────────────────────────────────────────
TARGET     = 500
N_MATCHES  = 20
rows       = []
match_id   = 0

# Ensure all 4 opponent styles appear: cycle through a guaranteed set first
_guaranteed_opps = [1, 5, 8, 3, 2, 6, 4, 7, 9, 10]  # covers looper/defender/blocker/allround

for match_idx in range(N_MATCHES):
    if len(rows) >= TARGET:
        break

    match_id += 1
    if match_idx < len(_guaranteed_opps):
        opp_id = _guaranteed_opps[match_idx]
    else:
        opp_id = int(rng.integers(1, 11))
    opp_skill, opp_style = OPPONENTS[opp_id]
    n_games             = int(rng.choice([3, 4, 5], p=[0.38, 0.32, 0.30]))
    side                = rng.choice(['forehand_side', 'backhand_side'], p=[0.60, 0.40])
    prev_serve          = None
    streak              = 0

    for game_num in range(1, n_games + 1):
        if len(rows) >= TARGET:
            break

        s, r       = 0, 0
        game_over  = False

        while not game_over and len(rows) < TARGET:
            gs = game_state(s, r)

            # ── Serve mechanics ─────────────────────────────────────────────
            serve_type, streak = choose_serve(gs, prev_serve, streak)
            prev_serve         = serve_type

            spin_opts, spin_w  = SERVE_SPIN[serve_type]
            spin_type          = wc(spin_opts, spin_w)

            if spin_type == 'float/no-spin':
                spin_intensity = int(rng.choice([1, 2], p=[0.72, 0.28]))
            elif serve_type == 'short_backspin':
                spin_intensity = int(rng.choice([2, 3], p=[0.40, 0.60]))
            else:
                spin_intensity = int(rng.choice([1, 2, 3], p=[0.25, 0.45, 0.30]))

            serve_length  = wc(SERVE_LENGTHS, SERVE_LENGTH_PROBS[serve_type])
            toss_height   = wc(TOSS_HEIGHTS,  TOSS_HEIGHT_PROBS[serve_type])
            cp_opts, cp_w = CONTACT_PROBS[serve_type]
            contact_point = wc(cp_opts, cp_w)

            # Placement targeting: attack elbow/wide-BH vs loopers
            if opp_style == 'looper':
                pz_w = [0.12, 0.13, 0.32, 0.22, 0.21]
            elif opp_style == 'defender':
                pz_w = [0.20, 0.20, 0.20, 0.20, 0.20]
            elif opp_style == 'blocker':
                pz_w = [0.18, 0.18, 0.28, 0.22, 0.14]
            else:
                pz_w = [0.20, 0.20, 0.24, 0.20, 0.16]
            placement_zone = wc(PLACEMENT_ZONES, pz_w)

            intended_setup = dict_wc(INTENDED_SETUP_PROBS[serve_type])

            # ── Outcome ─────────────────────────────────────────────────────
            return_type      = sample_return_type(
                spin_type, spin_intensity, serve_length, opp_style, opp_skill)
            return_quality   = sample_return_quality(return_type, spin_intensity, opp_skill)
            return_placement = ('not_applicable' if return_type in ('miss', 'net')
                                else wc(RETURN_PLACEMENTS))
            point_outcome, point_end_type = sample_point_outcome(
                return_type, return_quality, opp_skill)
            rally_type       = rally_type_for(return_type)
            rally_length     = sample_rally_length(return_type, rally_type, point_end_type)
            chop_outcome     = ('not_applicable' if rally_type != 'chop_rally'
                                else ('won' if point_outcome == 'won' else 'lost'))

            rows.append({
                # ── Serve mechanics ──────────────────────────────────────────
                'serve_type':           serve_type,
                'spin_type':            spin_type,
                'spin_intensity':       spin_intensity,
                'serve_length':         serve_length,
                'placement_zone':       placement_zone,
                'toss_height':          toss_height,
                'contact_point':        contact_point,
                # ── Match context ────────────────────────────────────────────
                'match_id':             match_id,
                'game_number':          game_num,
                'server_score':         s,
                'receiver_score':       r,
                'game_state':           gs,
                'opponent_id':          opp_id,
                'opponent_skill_level': opp_skill,
                'opponent_style':       opp_style,
                'side':                 side,
                # ── Outcomes ─────────────────────────────────────────────────
                'return_type':          return_type,
                'return_quality':       return_quality,
                'return_placement':     return_placement,
                'rally_length':         rally_length,
                'point_outcome':        point_outcome,
                'point_end_type':       point_end_type,
                # ── Chopper-specific ──────────────────────────────────────────
                'intended_setup':       intended_setup,
                'rally_type_achieved':  rally_type,
                'chop_rally_outcome':   chop_outcome,
            })

            if point_outcome == 'won':
                s += 1
            else:
                r += 1

            if max(s, r) >= 11 and abs(s - r) >= 2:
                game_over = True

# ── Finalise ────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows).head(TARGET)
df.to_csv('table_tennis_serves.csv', index=False)

print(f"Rows generated : {len(df)}")
print(f"Columns        : {list(df.columns)}\n")
print("Serve type distribution:")
print(df['serve_type'].value_counts().to_string(), "\n")
print("Return type distribution:")
print(df['return_type'].value_counts().to_string(), "\n")
print("Point outcome:")
print(df['point_outcome'].value_counts().to_string(), "\n")
print("Rally type achieved:")
print(df['rally_type_achieved'].value_counts().to_string(), "\n")
print("Chop rally outcome:")
print(df['chop_rally_outcome'].value_counts().to_string(), "\n")
print("Opponent styles:")
print(df['opponent_style'].value_counts().to_string(), "\n")
print("Spin intensity avg by spin_type:")
print(df.groupby('spin_type')['spin_intensity'].mean().round(2).to_string(), "\n")
print("Return quality avg by spin_intensity:")
print(df[df['return_quality'] > 0].groupby('spin_intensity')['return_quality'].mean().round(2).to_string())
