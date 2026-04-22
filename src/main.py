import numpy as np
from pyscf import gto, scf
import matplotlib.pyplot as plt

def rhf_scf(mol, max_iter=500, tol=1e-10, debug = False, damping_min = 0.3, damping_number = 0.1):
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
            alpha = min(1 , damping_min + damping_number * log_residual)
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

print("\n==============================")
print("TASK I: CORE IMPLEMENTATION")
print("==============================")

# =========================================
# (i) RHF SCF VALIDATION
# =========================================

def validate_energy(mol, label):
    print(f"\n--- Energy Validation: {label} ---")
    
    P, F, E_tot, J, K, eps, n_occ, h_core, S = rhf_scf(mol)

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    E_ref = mf.e_tot

    error = abs(E_tot - E_ref)
    print(f"Our Energy   = {E_tot:.12f}")
    print(f"PySCF Energy = {E_ref:.12f}")
    print(f"Error        = {error:.3e}")

    assert error < 1e-8, "Energy validation failed!"

    return P, F, E_tot, J, K, eps, n_occ, h_core, S


# Required systems
mol_H2O_sto3g = gto.M(
    atom='''
    O 0 0 0
    H 0 0.757 0.587
    H 0 -0.757 0.587
    ''',
    basis='sto-3g'
)

mol_HF_ccpVDZ = gto.M(
    atom='H 0 0 0; F 0.917 0 0',
    basis='ccpvdz'
)

# Run validation
validate_energy(mol_H2O_sto3g, "H2O / STO-3G")
validate_energy(mol_HF_ccpVDZ, "HF / cc-pVDZ")


# =========================================
# (ii) ENERGY DECOMPOSITION
# =========================================

print("\n--- Energy Decomposition (H2O / cc-pVDZ) ---")

mol_H2O_ccpvdz = gto.M(
    atom='''
    O 0 0 0
    H 0 0.757 0.587
    H 0 -0.757 0.587
    ''',
    basis='ccpvdz'
)

P, F, E_tot, J, K, eps, n_occ, h_core, S = rhf_scf(mol_H2O_ccpvdz)
energy_result = energy_decomposition_check(
    mol_H2O_ccpvdz, P, F, J, K, eps, n_occ, h_core
)

print(f"EJ/|E1| = {energy_result['EJ']/abs(energy_result['E1']):.6f}")
print(f"EK/|E1| = {energy_result['EK']/abs(energy_result['E1']):.6f}")
print(f"EK/EJ   = {energy_result['EK']/energy_result['EJ']:.6f}")

# =========================================
# (iii) DIPOLE MOMENT (WITH EXPERIMENT)
# =========================================

print("\n--- Dipole Comparison ---")

# Experimental values (Debye)
exp_values = {
    "H2O": 1.85,
    "HF": 1.83,
    "CO": 0.122
}

def dipole_analysis(mol, label):
    P, *_ = rhf_scf(mol)
    result = compute_dipole(mol, P)

    mu = result["magnitude_debye"]
    exp = exp_values[label]

    print(f"{label}:")
    print(f"  Computed = {mu:.4f} D")
    print(f"  Exp      = {exp:.4f} D")
    print(f"  Error    = {abs(mu-exp):.4f} D\n")


# Systems
mol_CO = gto.M(atom='C 0 0 0; O 1.128 0 0', basis='ccpvdz')

dipole_analysis(mol_H2O_ccpvdz, "H2O")
dipole_analysis(mol_HF_ccpVDZ, "HF")
dipole_analysis(mol_CO, "CO")


# =========================================
# (iv) MULLIKEN ANALYSIS
# =========================================

print("\n--- Mulliken Charges ---")

for mol, name in [
    (mol_H2O_sto3g, "H2O STO-3G"),
    (mol_HF_ccpVDZ, "HF cc-pVDZ")
]:
    print(f"\n{name}")
    P, F, E_tot, J, K, eps, n_occ, h_core, S = rhf_scf(mol)
    mulliken_population(mol, P, S)

# =========================================
# (v) Virial theorem
# =========================================

print("\n--- Virial Check (H2O / cc-pVDZ) ---")

P, F, E_tot, J, K, eps, n_occ, h_core, S = rhf_scf(mol_H2O_ccpvdz)

T = mol_H2O_ccpvdz.intor('int1e_kin')
V = mol_H2O_ccpvdz.intor('int1e_nuc')

virial_ratio(mol_H2O_ccpvdz, P, T, V, J, K)

print("\n==============================")
print("TASK II: ANALYSIS")
print("==============================")

# =========================================
# (i) ENERGY RATIOS + CORRELATION
# =========================================

print("\n--- Energy Analysis ---")

E1 = energy_result['E1']
EJ = energy_result['EJ']
EK = energy_result['EK']

print(f"EJ/|E1| = {EJ/abs(E1):.4f}")
print(f"EK/|E1| = {EK/abs(E1):.4f}")
print(f"EK/EJ   = {EK/EJ:.4f}")

# Correlation estimate
E_corr_est = -0.2
fraction = abs(E_corr_est) / abs(E_tot)

print(f"\nEstimated correlation fraction ≈ {fraction:.4f}")


# =========================================
# (ii) HF DIPOLE CURVE + PEAK DETECTION
# =========================================

print("\n--- HF Dipole Curve ---")

def get_mol_HF(R):
    return gto.M(atom=f'H 0 0 0; F {R} 0 0', basis='ccpvdz')

bond_lengths = []
dipoles = []

R = 0.7
delta_R = 0.3

while R <= 2.51:
    print(f"R = {R:.2f}")
    mol = get_mol_HF(R)

    # --- Use your adaptive damping strategy ---
    if R < 1.6:
        P, F, E_tot, J, K, eps, n_occ, h_core, S = rhf_scf(mol)

    elif R < 2.5:
        # Slightly stronger damping in mid-region (harder SCF zone)
        P, F, E_tot, J, K, eps, n_occ, h_core, S = rhf_scf(
            mol,
            damping_min=0.25,
            damping_number=0.05
        )

    else:
        # Very stretched bond → strongest stabilization
        P, F, E_tot, J, K, eps, n_occ, h_core, S = rhf_scf(
            mol,
            damping_min=0.09,
            damping_number=0.05
        )

    # --- Dipole computation ---
    mu = compute_dipole(mol, P)["magnitude_debye"]

    bond_lengths.append(R)
    dipoles.append(mu)

    R += delta_R


# =========================================
# Analysis: Peak detection
# =========================================

idx_max = np.argmax(dipoles)
print(f"\nMax dipole at R = {bond_lengths[idx_max]:.2f} Å")
print(f"Max dipole value = {dipoles[idx_max]:.4f} Debye")


# =========================================
# Plot
# =========================================

plt.figure(figsize=(8, 5))
plt.plot(bond_lengths, dipoles, marker='o', linestyle='-')

plt.xlabel("Bond Length R (Å)")
plt.ylabel("|μ| (Debye)")
plt.title("HF Dipole vs Bond Length (RHF)")
plt.grid(True)

plt.savefig("HF_dipole_vs_R.png")
print("Plot saved as HF_dipole_vs_R.png")

# =========================================
# (iii) MULLIKEN BASIS SET DEPENDENCE
# =========================================

print("\n--- Mulliken Basis Sensitivity ---")

basis_list = ['sto-3g', '6-31g*', 'ccpvtz']

for basis in basis_list:
    mol = gto.M(
        atom='''
        O 0 0 0
        H 0 0.757 0.587
        H 0 -0.757 0.587
        ''',
        basis=basis
    )
    print(f"\nBasis: {basis}")
    P, F, E_tot, J, K, eps, n_occ, h_core, S = rhf_scf(mol)
    mulliken_population(mol, P, S)


mol_H2O_ccpvdz = gto.M(
    atom = '''
    O 0.000000  0.000000  0.000000
    H 0.000000  0.757000  0.587000
    H 0.000000 -0.757000  0.587000
    ''',
    basis = 'ccpvdz',
    charge = 0,
    spin = 0,
    unit = 'Angstrom',
    symmetry = True,
    verbose = 0,
    max_memory = 16384
)
