# components/attacks.py
import math
import random

ABORT_QBER = 0.15     # typical abort threshold for BB84 in sims/report
DEFAULT_SIFT_ERR = 0.25  # MITM intercept-resend yields ~25% QBER on sifted bits (ideal)

def apply_mitm(results, strength=1.0):
    """
    Simulate an intercept-resend MITM on a fraction 'strength' of the raw key.
    We mix baseline QBER with ~25% to emulate BB84 detection.
    If QBER crosses abort, set final_key_rate=0 to emulate protocol abort.
    """
    base_qber = float(results['qber'])
    attacked_qber = (1 - strength) * base_qber + strength * DEFAULT_SIFT_ERR
    results['qber'] = attacked_qber
    # crude but effective: if above abort threshold, no secure key
    if attacked_qber >= ABORT_QBER:
        results['final_key_rate'] = 0.0
    else:
        # otherwise privacy amplification + EC overhead increases => drop rate
        results['final_key_rate'] *= max(1e-9, (1 - 2 * attacked_qber))
    return results

def apply_dos(results, mode="block", extra_loss_db=60.0):
    """
    Denial-of-Service on a link:
      - 'block': drop key rate to zero (link unusable)
      - 'jam': inflate channel loss hugely, which will drive rate to ~0
    """
    if mode == "block":
        results['final_key_rate'] = 0.0
    else:  # 'jam'
        results['channel_loss_db'] = float(results['channel_loss_db']) + extra_loss_db
        results['final_key_rate'] = 0.0
    return results

def apply_passive_eavesdrop(results, leak_fraction=0.3, qber_bump=0.0):
    """
    Passive tapping (e.g., beam-splitting/PNS on weak coherent pulses):
    - reduce the *secure* key rate by a privacy-leak factor,
    - optionally add a tiny QBER bump (defaults to none).
    """
    results['final_key_rate'] *= max(0.0, (1.0 - leak_fraction))
    results['qber'] = float(results['qber']) + qber_bump
    return results
