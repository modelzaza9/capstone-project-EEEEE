import numpy as np
from pyscf import gto, scf

def rhf_scf(mol, max_iter=500, tol=1e-10, debug = True):
    import numpy as np

    # --- Basic quantities ---
    n_elec = mol.nelectron
    n_occ = n_elec // 2
    n_ao = mol.nao_nr()

    # --- Integrals ---
    S = mol.intor('int1e_ovlp')
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    h_core = T + V
    eri = mol.intor('int2e')

    # --- Orthogonalization matrix X = S^(-1/2) ---
    s, U = np.linalg.eigh(S)
    X = U @ np.diag(s**-0.5) @ U.T

    # --- Initial guess (core Hamiltonian) ---
    F0 = h_core
    F0_prime = X.T @ F0 @ X
    eps, C_prime = np.linalg.eigh(F0_prime)
    C = X @ C_prime
    C_occ = C[:, :n_occ]
    P = 2.0 * C_occ @ C_occ.T
    
    alpha = 0.1
    E_old = 0.0
    R_norm_old = np.inf
    
    # --- SCF Loop ---
    for iteration in range(max_iter):

        # Build Fock matrix
        J = np.einsum('uvls,ls->uv', eri, P)
        K = np.einsum('ulvs,ls->uv', eri, P)
        F = h_core + J - 0.5 * K

        # Residual (use current P)
        R = F @ P @ S - S @ P @ F
        R_norm = np.linalg.norm(R)

        # Transform and solve
        F_prime = X.T @ F @ X
        eps, C_prime = np.linalg.eigh(F_prime)
        C = X @ C_prime

        # New density
        C_occ = C[:, :n_occ]
        P_new = 2.0 * C_occ @ C_occ.T

        # Energy
        E_elec = 0.5 * np.sum(P_new * (h_core + F))
        E_tot = E_elec + mol.energy_nuc()

        if debug == True:
            print(f"Iter {iteration:3d}  E = {E_tot:.12f}  |R| = {R_norm:.6e}")

        # Convergence (BOTH conditions)
        if abs(E_tot - E_old) < tol and R_norm < 1e-6:
            print(f"\nSCF converged in {iteration} iterations")
            return P_new, F, E_tot, J, K, eps, n_occ, h_core, S
        
        # Added a small epsilon (1e-12) to avoid division by zero
        
        progress = (R_norm_old / (R_norm + 1e-12)) - 1.0
        #adjustable to prevent diverging and slow converge
        log_residual = - np.log10(R_norm)
        # Damping DO NOT REMOVE OR IT NEVER CONVERGE
        # greedy Damping
        '''
        if E_tot > E_old or progress < 0:
            # Energy rose OR Residual grew: Slam brakes
            alpha = max(0.10, alpha * 0.5)
        elif progress < 0.5:
            # Stagnant: If we aren't moving, try to wake it up
            alpha = min(1, alpha * 1.5) 
        else:
             #Healthy: Keep the momentum but don't go crazy
            alpha = min(1, alpha * 1.2)
        '''
        # adaptive_damping
        if E_tot > E_old or progress < 0:
            alpha = min(1 , 0.30 + 0.10 * log_residual)
        else:
            alpha = 1
        P_mixed = alpha * P_new + (1.0 - alpha) * P
        P = P_mixed
        E_old = E_tot
        R_norm_old = R_norm
    print("SCF failed to converge")
    return P, F, E_tot, J, K, eps, n_occ, h_core, S

def energy_decomposition_check(mol, P, F, J, K, eps, n_occ, h_core, tol=1e-10):

    # --- One-electron term ---
    E1 = np.sum(P * h_core)

    # --- Coulomb & Exchange ---
    EJ = 0.5 * np.sum(P * J)
    EK = 0.25 * np.sum(P * K)

    # --- Electronic energies ---
    E_elec_1 = E1 + EJ - EK
    E_elec_2 = 0.5 * np.sum(P * (h_core + F))
    E_orb = 2.0 * np.sum(eps[:n_occ])
    E_elec_3 = E_orb - EJ + EK

    # --- Nuclear ---
    E_nuc = mol.energy_nuc()

    # --- Total energies ---
    E_tot_1 = E_elec_1 + E_nuc
    E_tot_2 = E_elec_2 + E_nuc
    E_tot_3 = E_elec_3 + E_nuc

    # --- Checks ---
    check_12 = np.allclose(E_elec_1, E_elec_2, atol=tol)
    check_13 = np.allclose(E_elec_1, E_elec_3, atol=tol)
    check_23 = np.allclose(E_elec_2, E_elec_3, atol=tol)

    # --- Output ---
    print("\n=== Energy Decomposition ===")
    print(f"E1 (one-electron)       = {E1:.12f}")
    print(f"EJ (Coulomb)            = {EJ:.12f}")
    print(f"EK (Exchange)           = {EK:.12f}")

    print("\n=== Electronic Energy ===")
    print(f"E1 + EJ - EK            = {E_elec_1:.12f}")
    print(f"1/2 Tr[P(h+F)]          = {E_elec_2:.12f}")
    print(f"Orbital expression      = {E_elec_3:.12f}")

    print("\n=== Total Energy ===")
    print(f"Decomposition           = {E_tot_1:.12f}")
    print(f"SCF formula             = {E_tot_2:.12f}")
    print(f"Orbital formula         = {E_tot_3:.12f}")

    print("\n=== Consistency Checks ===")
    print(f"(E1+EJ-EK) == SCF       → {check_12}")
    print(f"(E1+EJ-EK) == Orbital   → {check_13}")
    print(f"SCF == Orbital          → {check_23}")

    return {
        "E1": E1,
        "EJ": EJ,
        "EK": EK,
        "E_elec": E_elec_1,
        "E_tot": E_tot_1,
        "checks": (check_12, check_13, check_23)
    }

def compute_dipole(mol, P):
    import numpy as np

    # --- Electronic dipole integrals ---
    dip_ints = mol.intor('int1e_r')   # shape (3, nao, nao)

    # Electronic contribution
    mu_elec = np.einsum('xuv,uv->x', dip_ints, P)

    # Nuclear contribution
    coords = mol.atom_coords()
    charges = mol.atom_charges()
    mu_nuc = np.einsum('i,ix->x', charges, coords)

    # Total dipole (atomic units)
    mu_total_au = mu_nuc - mu_elec
    print(f'{mu_nuc},{mu_elec}')
    # Convert to Debye
    mu_total_debye = mu_total_au * 2.54174623

    # Magnitude
    mu_mag_au = np.linalg.norm(mu_total_au)
    mu_mag_debye = np.linalg.norm(mu_total_debye)

    # --- PySCF reference ---
    from pyscf import scf
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    mu_pyscf_au = mf.dip_moment(unit='au')
    mu_pyscf = mf.dip_moment(unit='Debye', verbose=0)

    # --- Print ---
    print("\n=== Dipole Moment ===")
    print(f"mu (a.u.)        = {mu_total_au}")
    print(f"|mu| (a.u.)      = {mu_mag_au:.6f}")
    print(f"mu (Debye)       = {mu_total_debye}")
    print(f"|mu| (Debye)     = {mu_mag_debye:.6f}")

    print("\n=== PySCF Reference (Debye) ===")
    print(f"PySCF mu (a.u.): {mu_pyscf_au}")
    print(f"mu_pyscf         = {mu_pyscf}")
    print(f"|mu|_pyscf       = {np.linalg.norm(mu_pyscf):.6f}")

    # --- Check ---
    check = np.allclose(mu_total_debye, mu_pyscf, atol=1e-6)

    print("\n=== Consistency Check ===")
    print(f"Our vs PySCF match → {check}")

    return {
        "mu_au": mu_total_au,
        "mu_debye": mu_total_debye,
        "magnitude_debye": mu_mag_debye,
        "pyscf_debye": mu_pyscf,
        "check": check
    }

def mulliken_population(mol, P, S, tol=1e-6):
    import numpy as np

    # Symmetrized Mulliken population matrix
    PS = 0.5 * (P @ S + S @ P)

    aoslices = mol.aoslice_by_atom()
    charges = mol.atom_charges()
    atom_labels = [mol.atom_symbol(i) for i in range(len(charges))]

    qA = []

    print("\n=== Mulliken Population Analysis ===")

    for A, (_, _, p0, p1) in enumerate(aoslices):
        pop_A = np.sum(np.diag(PS)[p0:p1])

        q = charges[A] - pop_A
        qA.append(q)

        print(f"Atom {A:2d} ({atom_labels[A]:>2s})  Charge = {q:.6f}")

    qA = np.array(qA)

    total_charge = np.sum(qA)
    is_neutral = np.allclose(total_charge, 0.0, atol=tol)

    print("\n=== Charge Check ===")
    print(f"Sum of charges = {total_charge:.10f}")
    print(f"Neutral molecule → {is_neutral}")

    return qA, total_charge, is_neutral

def virial_ratio(mol, P, T, V, J, K):
    import numpy as np

    # --- Kinetic energy ---
    T_exp = np.sum(P * T)

    # --- Electron-nuclear ---
    V_en = np.sum(P * V)

    # --- Electron-electron ---
    E_J = 0.5 * np.sum(P * J)
    E_K = 0.25 * np.sum(P * K)
    V_ee = E_J - E_K

    # --- Nuclear-nuclear ---
    V_nn = mol.energy_nuc()

    # --- Total potential ---
    V_total = V_en + V_ee + V_nn

    # --- Virial ratio ---
    eta = -V_total / T_exp

    # --- Print ---
    print("\n=== Virial Theorem Analysis ===")
    print(f"<T>        = {T_exp:.12f}")
    print(f"V_en       = {V_en:.12f}")
    print(f"V_ee       = {V_ee:.12f}")
    print(f"V_nn       = {V_nn:.12f}")
    print(f"<V> total  = {V_total:.12f}")
    print(f"Virial η   = {eta:.8f}")

    return {
        "T": T_exp,
        "V_en": V_en,
        "V_ee": V_ee,
        "V_nn": V_nn,
        "V_total": V_total,
        "eta": eta
    }

mol_H2O_sto3g = gto.M(
    atom = "O 0 0 0; H 0 0 0.96; H 0.92 0 0",
    basis = "sto-3g",
    unit = "Angstrom"
)

mol_H2O_ccpvdz = gto.M(
    atom = "O 0 0 0; H 0 0 0.96; H 0.92 0 0",
    basis = "cc-pvdz",
    unit = "Angstrom"
)

mol_HF_ccpvdz = gto.M(
    atom = "H 0 0 0; F 0 0 0.92",
    basis = "cc-pvdz",
    unit = "Angstrom"
)

mol_CO_ccpvdz = gto.M(
    atom = '''
    O   0.5285   0.0000   0.0000
    C  -0.5285   0.0000   0.0000
    ''',
    basis = "cc-pvdz",
    unit = "Angstrom",
    charge = 0,
    spin = 0,
    symmetry = True,
    verbose = 0
)

mol_Adenosine_sto_3g = gto.M(
    atom = '''
    O   1.9998   -0.5205   -0.9524
    O   1.2085    2.2652    1.1770
    O   3.4012    2.1355   -0.4232
    O   3.7891   -2.6031   -0.3835
    N  -0.2134   -0.2591   -0.1379
    N  -1.6397   -1.8913    0.3913
    N  -1.7853    1.5466   -0.5513
    N  -4.0890    0.8118   -0.2214
    N  -4.6438   -1.4234    0.4295
    C   1.7170    0.9876    0.8516
    C   1.0575    0.4149   -0.3937
    C   3.1742    1.0172    0.4346
    C   3.3045   -0.2641   -0.3829
    C   3.7054   -1.4672    0.4592
    C  -1.4639    0.2870   -0.2203
    C  -0.3728   -1.5695    0.2323
    C  -2.3326   -0.7403    0.1117
    C  -3.6915   -0.4420    0.1040
    C  -3.1253    1.7148   -0.5244
    H   1.5809    0.3255    1.7146
    H   0.8615    1.1833   -1.1517
    H   3.8662    1.0958    1.2779
    H   3.9968   -0.1521   -1.2252
    H   2.9710   -1.6864    1.2404
    H   4.6819   -1.3126    0.9273
    H   1.3811    2.8579    0.4257
    H   4.3301    2.0980   -0.7086
    H   0.4627   -2.2408    0.3766
    H   2.9169   -2.7275   -0.7954
    H  -3.4792    2.7078   -0.7798
    H  -4.3498   -2.3596    0.6729
    H  -5.6254   -1.1820    0.4170
    ''',
    basis = 'sto-3g',
    charge = 0,
    spin = 0,
    unit = 'Angstrom',
    symmetry = True,
    verbose = 0,
    max_memory = 16384
)

mol_Nitroglycerin_sto3g = gto.M(
    atom = '''
    O   0.0424   -1.3569    0.4110
    O   2.4896   -0.1938   -0.1771
    O  -1.5272    0.9543    0.2726
    O  -1.1430   -2.6329   -0.9457
    O  -1.5454   -2.7561    1.2440
    O   3.3033    1.8382    0.1442
    O  -3.2965    1.5003   -0.9379
    O   4.6890    0.1002    0.3197
    O  -3.1490    2.2897    1.1422
    N  -0.9724   -2.3340    0.2369
    N   3.5993    0.6429    0.1229
    N  -2.7640    1.6465    0.1631
    C   0.1644   -0.4658   -0.7000
    C   1.2787    0.5245   -0.3881
    C  -1.1692    0.2429   -0.9077
    H   0.4619   -1.0291   -1.5943
    H   1.0315    1.0753    0.5284
    H   1.3888    1.2205   -1.2291
    H  -1.9707   -0.4513   -1.1776
    H  -1.0471    0.9527   -1.7372
    ''',
    basis = 'sto-3g',
    charge = 0,
    spin = 0,
    unit = 'Angstrom',
    symmetry = True,
    verbose = 0,
    max_memory = 16384
)

# Hexanitrogen (Diazide) N6
# Coordinates calculated from:
# N1=N2: 1.138 A, N2=N3: 1.251 A, N3-N4: 1.460 A
# Angles: N1-N2-N3: 172.5°, N2-N3-N4: 107° (trans geometry)
mol_N6_6_31G_star = gto.M(
    atom = '''
    N  -1.2835   2.3188   0.0000
    N  -1.0957   1.1964   0.0000
    N  -0.7300   0.0000   0.0000
    N   0.7300   0.0000   0.0000
    N   1.0957  -1.1964   0.0000
    N   1.2835  -2.3188   0.0000
    ''',
    basis = '6-31G*',
    charge = 0,
    spin = 0,
    unit = 'Angstrom',
    symmetry = True,
    verbose = 0,
    max_memory = 16384
)

# -----------------------------
# Molecule registry (explicit names)
# -----------------------------
molecule_cases = [
    ("Adenosine", mol_Adenosine_sto_3g),
    ("Nitroglycerin", mol_Nitroglycerin_sto3g),
    ("N6_6_31G", mol_N6_6_31G_star),
]

# =========================================================
# RHF SCF TESTS (≥ 3 tests)
# =========================================================

def test_rhf_scf_energy_accuracy_H2O_sto3g():
    P, F, E_tot, *_ = rhf_scf(mol_H2O_sto3g)

    mf = scf.RHF(mol_H2O_sto3g)
    mf.conv_tol = 1e-10
    mf.kernel()

    assert abs(E_tot - mf.e_tot) < 1e-8, "SCF energy mismatch"

def test_rhf_scf_energy_accuracy_HF_ccpvdz():
    P, F, E_tot, *_ = rhf_scf(mol_HF_ccpvdz)

    mf = scf.RHF(mol_HF_ccpvdz)
    mf.conv_tol = 1e-10
    mf.kernel()

    assert abs(E_tot - mf.e_tot) < 1e-8, "SCF energy mismatch"

def test_rhf_scf_energy_accuracy_Adenosine():
    _, mol = molecule_cases[0]

    P, F, E_tot, *_ = rhf_scf(mol)

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()

    assert abs(E_tot - mf.e_tot) < 1e-8, "SCF energy mismatch"


def test_rhf_scf_energy_accuracy_Nitroglycerin():
    _, mol = molecule_cases[1]

    P, F, E_tot, *_ = rhf_scf(mol)

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()

    assert abs(E_tot - mf.e_tot) < 1e-8, "SCF energy mismatch"


def test_rhf_scf_energy_accuracy_N6():
    _, mol = molecule_cases[2]

    P, F, E_tot, *_ = rhf_scf(mol)

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()

    assert abs(E_tot - mf.e_tot) < 1e-8, "SCF energy mismatch"


# =========================================================
# ENERGY DECOMPOSITION TESTS (≥ 3 tests)
# =========================================================

def test_energy_decomposition_H2O():
    P, F, _, J, K, eps, n_occ, h_core, _ = rhf_scf(mol_H2O_ccpvdz)

    res = energy_decomposition_check(
        mol_H2O_ccpvdz, P, F, J, K, eps, n_occ, h_core, tol=1e-10
    )

    check_12, check_13, check_23 = res["checks"]
    assert check_12
    assert check_13
    assert check_23

def test_energy_decomposition_Adenosine():
    _, mol = molecule_cases[0]

    P, F, _, J, K, eps, n_occ, h_core, _ = rhf_scf(mol)

    res = energy_decomposition_check(
        mol, P, F, J, K, eps, n_occ, h_core
    )

    check_12, check_13, check_23 = res["checks"]
    assert check_12
    assert check_13
    assert check_23


def test_energy_decomposition_Nitroglycerin():
    _, mol = molecule_cases[1]

    P, F, _, J, K, eps, n_occ, h_core, _ = rhf_scf(mol)

    res = energy_decomposition_check(
        mol, P, F, J, K, eps, n_occ, h_core
    )

    check_12, check_13, check_23 = res["checks"]
    assert check_12
    assert check_13
    assert check_23


def test_energy_decomposition_N6():
    _, mol = molecule_cases[2]

    P, F, _, J, K, eps, n_occ, h_core, _ = rhf_scf(mol)

    res = energy_decomposition_check(
        mol, P, F, J, K, eps, n_occ, h_core
    )

    check_12, check_13, check_23 = res["checks"]
    assert check_12
    assert check_13
    assert check_23


# =========================================================
# DIPOLE TESTS (≥ 3 tests)
# =========================================================

def test_compute_dipole_HF():
    P, *_ = rhf_scf(mol_HF_ccpvdz)

    res = compute_dipole(mol_HF_ccpvdz, P)

    assert res["check"]
    assert np.linalg.norm(res["mu_debye"] - res["pyscf_debye"]) < 1e-6

def test_compute_dipole_CO():
    P, *_ = rhf_scf(mol_CO_ccpvdz)

    res = compute_dipole(mol_CO_ccpvdz, P)

    assert res["check"]
    assert np.linalg.norm(res["mu_debye"] - res["pyscf_debye"]) < 1e-6

def test_compute_dipole_Adenosine():
    _, mol = molecule_cases[0]

    P, *_ = rhf_scf(mol)

    res = compute_dipole(mol, P)

    assert res["check"]
    assert res["magnitude_debye"] >= 0


def test_compute_dipole_Nitroglycerin():
    _, mol = molecule_cases[1]

    P, *_ = rhf_scf(mol)

    res = compute_dipole(mol, P)

    assert res["check"]
    assert res["magnitude_debye"] >= 0


def test_compute_dipole_N6():
    _, mol = molecule_cases[2]

    P, *_ = rhf_scf(mol)

    res = compute_dipole(mol, P)

    assert res["check"]
    assert res["magnitude_debye"] >= 0


# =========================================================
# MULLIKEN POPULATION TESTS (≥ 3 tests)
# =========================================================

def test_mulliken_population_H2O():
    P, _, _, _, _, _, _, _, S = rhf_scf(mol_H2O_sto3g)

    _, total_charge, _ = mulliken_population(mol_H2O_sto3g, P, S)

    assert abs(total_charge) < 1e-10

def test_mulliken_population_Adenosine():
    _, mol = molecule_cases[0]

    P, _, _, _, _, _, _, _, S = rhf_scf(mol)

    qA, total_charge, is_neutral = mulliken_population(mol, P, S)

    assert is_neutral
    assert abs(total_charge) < 1e-6


def test_mulliken_population_Nitroglycerin():
    _, mol = molecule_cases[1]

    P, _, _, _, _, _, _, _, S = rhf_scf(mol)

    qA, total_charge, is_neutral = mulliken_population(mol, P, S)

    assert is_neutral
    assert abs(total_charge) < 1e-6


def test_mulliken_population_N6():
    _, mol = molecule_cases[2]

    P, _, _, _, _, _, _, _, S = rhf_scf(mol)

    qA, total_charge, is_neutral = mulliken_population(mol, P, S)

    assert is_neutral
    assert abs(total_charge) < 1e-6

# =========================================================
# VIRIAL RATIO TESTS (≥ 3 tests)
# =========================================================

def test_virial_ratio_H2O_ccpvdz():
    P, _, E_tot, J, K, _, _, _, _ = rhf_scf(mol_H2O_ccpvdz)

    T = mol_H2O_ccpvdz.intor('int1e_kin')
    V = mol_H2O_ccpvdz.intor('int1e_nuc')

    res = virial_ratio(mol_H2O_ccpvdz, P, T, V, J, K)

    # Virial theorem: η ≈ 2
    assert abs(res["eta"] - 2.0) < 0.05

    # Energy consistency: T + V = E_total
    assert abs((res["T"] + res["V_total"]) - E_tot) < 1e-8


def test_virial_ratio_HF_ccpvdz():
    P, _, E_tot, J, K, _, _, _, _ = rhf_scf(mol_HF_ccpvdz)

    T = mol_HF_ccpvdz.intor('int1e_kin')
    V = mol_HF_ccpvdz.intor('int1e_nuc')

    res = virial_ratio(mol_HF_ccpvdz, P, T, V, J, K)

    assert abs(res["eta"] - 2.0) < 0.05
    assert abs((res["T"] + res["V_total"]) - E_tot) < 1e-8


def test_virial_ratio_CO_ccpvdz():
    P, _, E_tot, J, K, _, _, _, _ = rhf_scf(mol_CO_ccpvdz)

    T = mol_CO_ccpvdz.intor('int1e_kin')
    V = mol_CO_ccpvdz.intor('int1e_nuc')

    res = virial_ratio(mol_CO_ccpvdz, P, T, V, J, K)

    assert abs(res["eta"] - 2.0) < 0.05
    assert abs((res["T"] + res["V_total"]) - E_tot) < 1e-8

def test_virial_ratio_Adenosine():
    _, mol = molecule_cases[0]

    P, _, E_tot, J, K, _, _, _, _ = rhf_scf(mol)

    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')

    res = virial_ratio(mol, P, T, V, J, K)

    # Larger tolerance due to STO-3G + size
    assert abs(res["eta"] - 2.0) < 0.1
    assert abs((res["T"] + res["V_total"]) - E_tot) < 1e-6

# Run all tests manually
'''
# RHF SCF
test_rhf_scf_energy_accuracy_H2O_sto3g()
test_rhf_scf_energy_accuracy_HF_ccpvdz()
test_rhf_scf_energy_accuracy_Adenosine()
test_rhf_scf_energy_accuracy_Nitroglycerin()
test_rhf_scf_energy_accuracy_N6()

# Energy decomposition
test_energy_decomposition_H2O()
test_energy_decomposition_Adenosine()
test_energy_decomposition_Nitroglycerin()
test_energy_decomposition_N6()

# Dipole
test_compute_dipole_HF()
test_compute_dipole_CO()
test_compute_dipole_Adenosine()
test_compute_dipole_Nitroglycerin()
test_compute_dipole_N6()

# Mulliken
test_mulliken_population_H2O()
test_mulliken_population_Adenosine()
test_mulliken_population_Nitroglycerin()
test_mulliken_population_N6()

# virial_ratio
test_virial_ratio_Adenosine()
test_virial_ratio_CO_ccpvdz()
test_virial_ratio_HF_ccpvdz()
test_virial_ratio_H2O_ccpvdz()
'''
