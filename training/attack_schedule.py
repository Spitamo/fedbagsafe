# Exact per-round attack schedule of the final paper notebook (100 rounds).
NOTEBOOK_SCHEDULE = [
    # Phase 1: Stable Initialization (Rounds 1-10)
    'benign', 'benign', 'benign', 'benign', 'benign',
    'benign', 'benign', 'benign', 'benign', 'benign',
    # Phase 2: Gradual Attack Introduction (Rounds 11-25)
    'benign', 'benign', 'backdoor', 'benign', 'benign',
    'labelflip', 'benign', 'benign', 'modelpoisoning', 'benign',
    'benign', 'backdoor', 'benign', 'labelflip', 'benign',
    # Phase 3: Moderate Attack Frequency (Rounds 26-50)
    'benign', 'backdoor', 'benign', 'modelpoisoning', 'benign',
    'labelflip', 'benign', 'backdoor', 'benign', 'modelpoisoning',
    'benign', 'labelflip', 'benign', 'backdoor', 'benign',
    'modelpoisoning', 'benign', 'labelflip', 'benign', 'backdoor',
    'benign', 'modelpoisoning', 'benign', 'labelflip', 'benign',
    # Phase 4: High-Intensity Attacks (Rounds 51-75)
    'backdoor', 'modelpoisoning', 'benign', 'labelflip', 'backdoor',
    'benign', 'modelpoisoning', 'labelflip', 'benign', 'backdoor',
    'modelpoisoning', 'benign', 'labelflip', 'backdoor', 'benign',
    'modelpoisoning', 'labelflip', 'benign', 'backdoor', 'modelpoisoning',
    'benign', 'labelflip', 'backdoor', 'benign', 'modelpoisoning',
    # Phase 5: Consecutive Attack Stress Test (Rounds 76-85)
    'backdoor', 'backdoor', 'modelpoisoning', 'modelpoisoning', 'labelflip',
    'labelflip', 'backdoor', 'modelpoisoning', 'labelflip', 'benign',
    # Phase 6: Recovery and Final Validation (Rounds 86-100)
    'benign', 'benign', 'backdoor', 'benign', 'benign',
    'modelpoisoning', 'benign', 'benign', 'labelflip', 'benign',
    'benign', 'backdoor', 'benign', 'benign', 'benign',
]