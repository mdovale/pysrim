"""
pysrim.dose
===========

Functions for calculating Total Ionizing Dose (TID) using SRIM output,
including CSDA-based residual energy estimation and LET-based dose models.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from pysrim import SR, Ion, Layer
import concurrent.futures
from tqdm import tqdm

def safe_ion_energy(energy_MeV, precision_keV=10):
    """
    Rounds energy to the nearest multiple of `precision_keV` [keV] 
    to avoid floating-point edge cases that may cause SRIM to fail.

    Parameters:
    -----------
    energy_MeV : float
        Ion energy in MeV.
    precision_keV : float
        Precision to round to, in keV. Default is 10 keV.

    Returns:
    --------
    rounded_MeV : float
        Rounded energy in MeV.
    """
    keV = energy_MeV * 1000
    rounded_keV = round(keV / precision_keV) * precision_keV
    rounded_MeV = rounded_keV / 1000
    return rounded_MeV

def run_srim_with_timeout(srim, timeout):
    """
    Runs SRIM with a timeout. Uses a separate thread to allow timing out long or frozen runs.

    Parameters:
    -----------
    srim : srim.SR
        Configured SRIM object.
    timeout : int
        Timeout in seconds.

    Returns:
    --------
    srim_output : object
        The result of srim.run()

    Raises:
    -------
    concurrent.futures.TimeoutError
        If the SRIM call exceeds the given timeout.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(srim.run)
        return future.result(timeout=timeout)

def residual_energy(ion: Ion, layer: Layer, depth_mm=None, precision_keV=10, timeout=60):
    """
    Calculate the residual energy of an ion after passing through a target material.

    Parameters:
    -----------
    ion : srim.Ion
        Incident ion, defined using srim.Ion.
    layer : srim.Layer
        Target layer, defined using srim.Layer.
    depth_mm : float or array-like
        Depth(s) in millimeters at which to compute residual energy. If None, use full layer thickness.
    precision_keV : int, optional
        Energy rounding precision in keV to avoid SRIM hanging at problematic energies.
    timeout : int, optional
        Timeout in seconds for the SRIM run. Default is 60.

    Returns:
    --------
    residual_energy : float or ndarray
        Residual ion energy after passing through the given depth(s) [MeV].
        Returns a float if input depth is scalar; otherwise, a NumPy array.
    """
    # Use the layer width as the default depth if none is given
    depth = depth_mm if depth_mm is not None else layer.width / 1e7  # Convert Å → mm

    # Prepare a list of energy perturbations to try if SRIM fails (e.g. ±10, ±20 ... keV)
    base_energy_MeV = ion.energy / 1e6
    perturbations = [0] + [i * precision_keV / 1000 for i in range(1, 6)]
    adjustments = [delta for pair in zip(perturbations[1:], [-x for x in perturbations[1:]]) for delta in pair]
    adjustments.insert(0, 0.0)  # Ensure original energy is tried first

    # Try running SRIM with adjusted energies until successful or out of retries
    for delta in adjustments:
        try_energy = base_energy_MeV + delta
        energy_safe = safe_ion_energy(try_energy, precision_keV)
        safe_ion = Ion(ion.symbol, energy=energy_safe * 1e6)  # Convert MeV → eV

        srim = SR(layer, safe_ion, output_type=5)  # Output type 5 = MeV·cm²/mg
        try:
            # print(f"Trying SRIM with {energy_safe:.6f} MeV (Δ = {delta:+.4f} MeV)")
            results = run_srim_with_timeout(srim, timeout=timeout)
            break  # Exit loop on successful SRIM run
        except Exception as e:
            print(f"⚠️  SRIM failed or timed out at {energy_safe:.6f} MeV (Δ = {delta:+.4f} MeV): {str(e)}")
    else:
        # If all energy variants fail, raise an error
        raise RuntimeError(f"❌ SRIM failed for all energy perturbations near {base_energy_MeV:.6f} MeV.")

    # Extract energy and stopping power from SRIM output
    E = np.array(results.data[0]) * 1e-3  # Convert energy from keV to MeV
    S = np.array(results.data[1])         # Stopping power: MeV·cm²/mg

    # Convert stopping power to MeV/mm using material density
    density_mg_cm3 = layer.density * 1000        # g/cm³ → mg/cm³
    S_mm = S * density_mg_cm3 / 10               # MeV/mm

    # Ensure energy array is in descending order for correct integration
    E = E[::-1]
    S_mm = S_mm[::-1]

    # Integrate CSDA range: compute depth as a function of energy
    depth_cumulative = -cumulative_trapezoid(1 / S_mm, E, initial=0)  # [mm]

    # Interpolate energy as a function of depth
    E_vs_depth = interp1d(depth_cumulative, E, kind='linear', bounds_error=False)

    # Evaluate residual energy at input depth(s)
    depth_input = np.atleast_1d(depth)  # Always treat input as array for safety
    energy_out = E_vs_depth(depth_input)

    # Return scalar if input was scalar, else full array
    return float(energy_out[0]) if np.isscalar(depth) else energy_out

def residual_energy_through_stack(ion, layer_stack, verbose=True):
    """
    Calculate the residual energy of an ion after passing through a stack of layers.

    Parameters:
    -----------
    ion : srim.Ion
        Incident ion (initial state), defined using srim.Ion.
    layer_stack : list of srim.Layer
        List of Layer objects representing the material stack.
    verbose : bool, optional
        If True, print energy after each layer. Default is True.

    Returns:
    --------
    final_energy : float
        Ion energy (in MeV) after traversing all layers in the stack.

    energy_profile : list of float
        List of residual energies (in MeV) after each layer.
    """
    current_energy = ion.energy / 1e6  # Convert from eV to MeV
    energy_profile = []

    for i, layer in enumerate(layer_stack):
        # Extract physical thickness of the layer in mm
        layer_thickness_mm = layer.width / 1e7  # width is in Angstroms

        # Check for zero-thickness layers and skip them
        if layer_thickness_mm <= 0:
            if verbose:
                print(f"Warning: Skipping layer {i+1} ({layer.name}) with zero or negative width.")
            energy_profile.append(current_energy)
            continue

        # Define ion at current energy
        ion_segment = Ion(ion.symbol, energy=current_energy * 1e6)  # energy in eV

        # Compute residual energy
        energy_after_layer = residual_energy(
            ion_segment,
            layer
        )

        # Optional logging
        if verbose:
            print(f"Energy after layer {i+1} ({layer.name}): {energy_after_layer} MeV")

        # Stop early if ion is fully stopped
        if energy_after_layer <= 0:
            if verbose:
                print(f"Ion stopped in layer {i+1} ({layer.name}).")
            energy_profile.append(0.0)
            return 0.0, energy_profile

        current_energy = energy_after_layer
        energy_profile.append(current_energy)

    return current_energy, energy_profile

def compute_tid(ion, layer_stack, fluence, verbose=True, timeout=60, precision_keV=10):
    """
    Compute Total Ionizing Dose (TID) in Silicon after ion passes through a material stack.

    Parameters:
    -----------
    ion : srim.Ion
        Incident ion object (e.g., Ion('H', energy=60e6)).
    layer_stack : list of srim.Layer
        Material stack (list of layers) the ion passes through.
    fluence : float
        Ion fluence [ions/cm^2].
    verbose : bool
        If True, print intermediate results.
    timeout : int, optional
        Timeout in seconds for the SRIM run. Default is 60 seconds.
    precision_keV : int, optional
        Energy quantization used for rounding. Default is 10 keV.

    Returns:
    --------
    residual_energy : float
        Energy of the ion after passing through the shielding [MeV].

    tid_rads : float
        Total Ionizing Dose in rads.
    """
    k = 1.602e-5  # Conversion constant: rads / (MeV * mg / cm^2)

    # Step 1: Propagate through the stack to get residual energy
    residual_E, _ = residual_energy_through_stack(
        ion,
        layer_stack,
        verbose=verbose
    )

    if residual_E <= 0:
        if verbose:
            print("Warning: Ion fully stopped before reaching silicon. TID = 0.")
        return 0.0, 0.0

    # Step 2: Compute LET(E) in Silicon using SRIM
    si_layer = Layer(
        elements={'Si': {'stoich': 1.0, 'E_d': 15.0, 'lattice': 0.0, 'surface': 4.7}},
        density=2.33,  # g/cm³ for Si
        width=1e5,     # 0.01 mm in Ångströms
        name="Si"
    )

    try:
        si_ion = Ion(ion.symbol, energy=safe_ion_energy(residual_E, precision_keV) * 1e6)  # MeV → eV
        srim = SR(si_layer, si_ion, output_type=5)
        results = run_srim_with_timeout(srim, timeout=timeout)
    except Exception as e:
        print(f"⚠️ SRIM failed when calculating LET in Si at {residual_E:.6f} MeV: {e}")
        return residual_E, np.nan

    # Interpolate LET from SRIM table
    energies = np.array(results.data[0]) * 1e-3  # keV → MeV
    LETs = np.array(results.data[1])             # MeV·cm²/mg

    LET_interp = np.interp(residual_E, energies, LETs)

    # Step 3: Compute TID
    tid = k * fluence * LET_interp  # rads

    if verbose:
        print(f"Residual energy: {residual_E:.2f} MeV")
        print(f"Interpolated LET in Si: {LET_interp:.4f} MeV·cm²/mg")
        print(f"TID: {tid:.2f} rad")

    return residual_E, tid

def sweep_al_shield_dose(
    al_thicknesses_mm,
    fluence,
    beam_energy_MeV=45.0,
    max_layer_thickness_mm=2.0,
    verbose=False
):
    """
    Calculate residual energy and dose for a sweep of total aluminum shielding thicknesses.

    Parameters:
    -----------
    al_thicknesses_mm : array-like
        Array of total aluminum thicknesses to simulate [mm].
    fluence : float
        Ion fluence in ions/cm².
    beam_energy_MeV : float, optional
        Initial energy of the ion beam [MeV]. Default is 45 MeV.
    max_layer_thickness_mm : float, optional
        Maximum individual aluminum layer thickness [mm] for SRIM stability. Default is 2 mm.
    verbose : bool, optional
        Whether to print detailed output from SRIM calls.

    Returns:
    --------
    residual_E : ndarray
        Array of residual energies at the target [MeV].
    doses : ndarray
        Array of corresponding TID values [rad].
    """
    al_thicknesses_mm = np.asarray(al_thicknesses_mm)
    residual_E = np.full(len(al_thicknesses_mm), np.nan)
    doses = np.full(len(al_thicknesses_mm), np.nan)

    try:
        for w_idx, total_thickness_mm in enumerate(tqdm(al_thicknesses_mm, desc="Sweeping Al thickness")):
            ion = Ion('H', energy=beam_energy_MeV * 1e6)

            # Build stack of aluminum layers with max thickness limit
            num_full_layers = int(total_thickness_mm // max_layer_thickness_mm)
            remainder_thickness = total_thickness_mm % max_layer_thickness_mm

            layer_stack = []

            if num_full_layers > 0:
                full_layer = Layer(
                    elements={'Al': {'stoich': 1.0, 'E_d': 25.0, 'lattice': 0.0, 'surface': 3.36}},
                    density=2.702,
                    width=max_layer_thickness_mm * 1e7,
                    name="Al"
                )
                layer_stack.extend([full_layer] * num_full_layers)

            if remainder_thickness > 0:
                partial_layer = Layer(
                    elements={'Al': {'stoich': 1.0, 'E_d': 25.0, 'lattice': 0.0, 'surface': 3.36}},
                    density=2.702,
                    width=remainder_thickness * 1e7,
                    name="Al"
                )
                layer_stack.append(partial_layer)

            try:
                res_E, dose = compute_tid(
                    ion,
                    layer_stack,
                    fluence,
                    verbose=verbose
                )
            except Exception as e:
                print(f"⚠️ Error at total thickness {total_thickness_mm:.2f} mm: {e}")
                res_E, dose = np.nan, np.nan

            residual_E[w_idx] = res_E
            doses[w_idx] = dose

    except KeyboardInterrupt:
        print("\n⏹️ Sweep interrupted by user. Returning partial results.")

    return residual_E, doses