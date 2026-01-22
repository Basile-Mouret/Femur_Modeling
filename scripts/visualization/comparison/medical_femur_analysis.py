#!/usr/bin/env python3
"""
Medical Femur Analysis Tool

Analyzes a patient's femur shape compared to the population mean (atlas),
showing:
1. Geodesic (Fréchet) distance - overall shape deviation
2. Per-vertex deviation heatmap
3. Tangent PCA component breakdown - which shape variations are present
4. Clinical interpretation of deformations

Usage:
    python medical_femur_analysis.py <patient_femur.obj> [--output report/]

Author: Femur Modeling Project
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import argparse
import json
import numpy as np
import pyvista as pv
import trimesh
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from lddmm import TangentPCA


# =============================================================================
# Anatomical Component Definitions
# =============================================================================

# Based on Tangent PCA analysis - see ANALYSIS_NOTES.md for details
# Each component includes direction-specific findings for sentence generation
COMPONENT_ANATOMY = {
    1: {
        "name": "Overall Size",
        "short": "Size",
        "parens": "",  # No clarification needed
        "description": "Global scaling of the femur",
        "clinical": "Correlates with patient stature, general implant sizing",
        "finding_pos": "Larger than average femur size",
        "finding_neg": "Smaller than average femur size",
        "significance": "high",
    },
    2: {
        "name": "Bone Proportions",
        "short": "Proportions",
        "parens": "allometric",
        "description": "Allometric scaling - length vs thickness ratio",
        "clinical": "Important for implant sizing, fracture risk assessment",
        "finding_pos": "Shorter and thicker bone proportions",
        "finding_neg": "Longer and thinner bone proportions",
        "significance": "high",
    },
    3: {
        "name": "Global Femoral Torsion + Width",
        "short": "Torsion",
        "parens": "mixed",
        "description": "Rotational alignment (anteversion) on both proximal and distal ends, combined with condylar width",
        "clinical": "Hip arthroplasty planning, gait mechanics, patellofemoral tracking",
        "finding_pos": "Increased femoral anteversion with wider condyles",
        "finding_neg": "Decreased femoral anteversion (retroversion) with narrower condyles",
        "finding_magnitude": "Atypical femoral torsion angle",  # When direction less important
        "significance": "high",
    },
    4: {
        "name": "Extremity Shape",
        "short": "Extremities",
        "parens": "mixed/noisy",
        "description": "Mixed variation in proximal and distal extremity proportions",
        "clinical": "May affect implant fit at bone ends",
        "finding_magnitude": "Atypical extremity morphology",  # Direction not meaningful
        "significance": "low",  # Noisy component
    },
    5: {
        "name": "Global Shape and Size",
        "short": "Global Shape and Size",
        "parens": "mixed/noisy",
        "description": "Mixed variation in and overall size and geoemtry",
        "clinical": "Non significant clinically",
        "finding_magnitude": "Atypical global geometry and size",  # Magnitude matters
        "significance": "low",
    },
    6: {
        "name": "Proximal Femoral Version ",
        "short": "Proximal Femoral Version",
        "parens": "",
        "description": "Angulation of the femoral head (hip articulation surface)",
        "clinical": "Hip alignment, hip arthroplasty planning, gait mechanics",
        "finding_pos": "Proximal Femoral Anteversion",
        "finding_neg": "Proximal Femoral Retroversion",
        "significance": "medium",
    },
    7: {
        "name": "Femoral Neck Length",
        "short": "Femoral Neck Length",
        "parens": "",
        "description": "Variation in femoral neck length and head offset",
        "clinical": "Lever arm for the Hip, Risk of Impingement",
        "finding_pos": "Shorter Femoral Neck Length",
        "finding_neg": "Longer Femoral Neck Length",
        "finding_magnitude": "Atypical femoral neck length",  # When direction less important
        "significance": "medium",  # Likely noise at 0.7% variance
    },
}


# Deviation severity levels based on statistical significance (standard deviations)
# Using standard normal distribution percentiles:
# |σ| < 0.5: ~38% of population (very common)
# |σ| < 1.0: ~68% of population (normal range)
# |σ| < 1.5: ~87% of population (mild deviation)
# |σ| < 2.0: ~95% of population (moderate deviation)
# |σ| < 2.5: ~99% of population (significant)
# |σ| >= 2.5: <1% of population (extreme)

def get_deviation_level(sigma: float) -> Tuple[str, str, str]:
    """
    Get deviation level based on standard deviations from mean.
    
    Returns:
        (level_name, color_code, emoji)
    """
    magnitude = abs(sigma)
    if magnitude < 0.5:
        return ("Normal", "#2ecc71", "✓")  # Green
    elif magnitude < 1.0:
        return ("Typical", "#3498db", "○")  # Blue
    elif magnitude < 1.5:
        return ("Mild", "#f39c12", "△")  # Orange
    elif magnitude < 2.0:
        return ("Moderate", "#e67e22", "▲")  # Dark orange
    elif magnitude < 2.5:
        return ("Significant", "#e74c3c", "◆")  # Red
    else:
        return ("Extreme", "#8e44ad", "★")  # Purple


def get_direction_text(sigma: float) -> str:
    """Get human-readable direction of deviation."""
    if abs(sigma) < 0.3:
        return "average"
    elif sigma > 0:
        return "increased"
    else:
        return "decreased"


def get_finding_sentence(pc_num: int, sigma: float) -> str:
    """
    Generate a complete sentence describing the finding for a PC.
    
    Takes into account whether direction matters or only magnitude.
    """
    if pc_num not in COMPONENT_ANATOMY:
        return f"Atypical variation in component {pc_num}"
    
    anat = COMPONENT_ANATOMY[pc_num]
    
    # Get severity adjective
    if abs(sigma) >= 2.5:
        severity = "Extremely"
    elif abs(sigma) >= 2.0:
        severity = "Significantly"
    elif abs(sigma) >= 1.5:
        severity = "Moderately"
    elif abs(sigma) >= 1.0:
        severity = "Mildly"
    else:
        severity = "Slightly"
    
    # Check if this component has direction-specific findings
    if "finding_pos" in anat and "finding_neg" in anat:
        if sigma > 0:
            base = anat["finding_pos"]
        else:
            base = anat["finding_neg"]
        # Combine severity with finding
        return f"{severity} {base[0].lower()}{base[1:]}"
    else:
        # Use magnitude-only finding
        base = anat.get("finding_magnitude", f"Atypical {anat['name'].lower()}")
        return f"{severity} {base[0].lower()}{base[1:]}"


@dataclass
class FemurAnalysisResult:
    """Results from femur shape analysis."""
    # Basic info
    patient_file: str
    n_vertices: int
    
    # Distance metrics
    l2_distance: float
    l2_rmse: float
    l2_max: float
    frechet_distance: float
    
    # PCA decomposition
    pca_coefficients: np.ndarray
    pca_coefficients_std: np.ndarray  # In units of standard deviation
    explained_variance_ratio: np.ndarray
    
    # Per-vertex data
    per_vertex_distances: np.ndarray
    vertices: np.ndarray
    atlas_vertices: np.ndarray


def load_obj_mesh(obj_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from OBJ file."""
    mesh = trimesh.load(obj_path, process=False, force='mesh')
    return np.array(mesh.vertices, dtype=np.float32), np.array(mesh.faces)  # type: ignore


def compute_frechet_distance_simple(
    source: np.ndarray,
    target: np.ndarray,
    sigma: float = 10.0
) -> float:
    """
    Compute approximate Fréchet distance using kernel regression.
    
    This is a simplified computation suitable for analysis purposes.
    """
    import torch
    
    # Subsample for efficiency
    n_points = len(source)
    subsample = max(1, n_points // 2000)
    if subsample > 1:
        source = source[::subsample]
        target = target[::subsample]
    
    device = torch.device("cpu")
    source_t = torch.tensor(source, dtype=torch.float64, device=device)
    target_t = torch.tensor(target, dtype=torch.float64, device=device)
    
    # Gaussian kernel
    def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        N, M = x.shape[0], y.shape[0]
        if N * M < 1e7:
            diff = x.unsqueeze(1) - y.unsqueeze(0)
            sq_dist = (diff ** 2).sum(dim=2)
            return torch.exp(-sq_dist / (2 * sigma ** 2))
        # Chunked for large matrices
        chunk_size = max(1, int(1e7 / M))
        K = torch.zeros(N, M, device=x.device, dtype=x.dtype)
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            diff = x[i:end_i].unsqueeze(1) - y.unsqueeze(0)
            sq_dist = (diff ** 2).sum(dim=2)
            K[i:end_i] = torch.exp(-sq_dist / (2 * sigma ** 2))
        return K
    
    # Compute kernel and solve
    K = gaussian_kernel(source_t, source_t, sigma)
    K = K + 1e-6 * torch.eye(K.shape[0], device=device, dtype=torch.float64)
    
    displacement = target_t - source_t
    KtK = K.T @ K
    reg_matrix = KtK + 1.0 * torch.eye(K.shape[0], device=device, dtype=torch.float64)
    
    momentum = torch.zeros_like(source_t)
    for d in range(3):
        rhs = K.T @ displacement[:, d]
        momentum[:, d] = torch.linalg.solve(reg_matrix, rhs)
    
    # RKHS norm
    norm_sq = 0.0
    for d in range(3):
        p_d = momentum[:, d]
        norm_sq = norm_sq + (p_d @ K @ p_d).item()
    
    return float(np.sqrt(norm_sq))


def analyze_femur(
    patient_path: str,
    tangent_pca_model: TangentPCA,
    sigma: float = 10.0
) -> FemurAnalysisResult:
    """
    Perform comprehensive analysis of a patient's femur shape.
    
    Args:
        patient_path: Path to patient's femur OBJ file
        tangent_pca_model: Fitted TangentPCA model with atlas
        sigma: Kernel bandwidth for Fréchet distance
        
    Returns:
        FemurAnalysisResult with all analysis data
    """
    # Load patient mesh
    patient_vertices, _ = load_obj_mesh(patient_path)
    
    # Get atlas (population mean)
    atlas = tangent_pca_model.atlas
    if atlas is None:
        raise RuntimeError("TangentPCA model not fitted")
    
    # Verify correspondence
    if patient_vertices.shape != atlas.shape:
        raise ValueError(f"Shape mismatch: patient {patient_vertices.shape} vs atlas {atlas.shape}")
    
    # Compute L2 metrics
    per_vertex_distances = np.linalg.norm(patient_vertices - atlas, axis=1)
    l2_total = np.sqrt(np.sum(per_vertex_distances ** 2))
    l2_rmse = np.sqrt(np.mean(per_vertex_distances ** 2))
    l2_max = np.max(per_vertex_distances)
    
    # Compute Fréchet distance
    frechet_dist = compute_frechet_distance_simple(atlas, patient_vertices, sigma=sigma)
    
    # Project onto Tangent PCA
    coefficients = tangent_pca_model.project(patient_vertices)
    
    # Convert to standard deviation units
    eigenvalues = tangent_pca_model.eigenvalues
    if eigenvalues is not None:
        stds = np.sqrt(eigenvalues)
        coefficients_std = coefficients / stds
    else:
        coefficients_std = coefficients
    
    # Get explained variance
    explained_var = tangent_pca_model.explained_variance_ratio
    if explained_var is None:
        explained_var = np.zeros_like(coefficients)
    
    return FemurAnalysisResult(
        patient_file=Path(patient_path).name,
        n_vertices=len(patient_vertices),
        l2_distance=float(l2_total),
        l2_rmse=float(l2_rmse),
        l2_max=float(l2_max),
        frechet_distance=frechet_dist,
        pca_coefficients=coefficients,
        pca_coefficients_std=coefficients_std,
        explained_variance_ratio=explained_var,
        per_vertex_distances=per_vertex_distances,
        vertices=patient_vertices,
        atlas_vertices=atlas,
    )


def interpret_pca_components(result: FemurAnalysisResult, n_components: int = 5) -> List[str]:
    """
    Generate summary of dominant PCA components by magnitude.
    
    Returns the top components sorted by absolute deviation from mean,
    using sentence-based findings.
    """
    interpretations = []
    
    # Sort by absolute coefficient magnitude
    indices = np.argsort(np.abs(result.pca_coefficients_std))[::-1]
    
    for idx in indices[:n_components]:
        coeff_std = float(result.pca_coefficients_std[idx])
        pc_num = idx + 1
        
        # Skip low-significance components unless extreme
        significance = COMPONENT_ANATOMY.get(pc_num, {}).get('significance', 'high')
        if significance == 'low' and abs(coeff_std) < 2.0:
            continue
        
        # Generate sentence-based finding
        sentence = get_finding_sentence(pc_num, coeff_std)
        level, _, emoji = get_deviation_level(coeff_std)
        
        interpretation = f"{emoji} {sentence} (PC{pc_num}: {coeff_std:+.2f}σ, {level})"
        interpretations.append(interpretation)
    
    return interpretations


def get_clinical_findings(result: FemurAnalysisResult) -> List[Dict]:
    """
    Generate clinical findings based on significant deviations.
    
    Returns list of findings with anatomical context.
    """
    findings = []
    
    for i, coeff in enumerate(result.pca_coefficients_std[:7]):  # First 7 components
        pc_num = i + 1
        coeff_val = float(coeff)  # Convert numpy scalar to float
        level, color, emoji = get_deviation_level(coeff_val)
        
        if abs(coeff_val) >= 1.0:  # Only report if outside normal range
            if pc_num in COMPONENT_ANATOMY:
                anatomy = COMPONENT_ANATOMY[pc_num]
                finding = {
                    "pc": pc_num,
                    "name": anatomy["name"],
                    "sigma": coeff_val,
                    "level": level,
                    "color": color,
                    "emoji": emoji,
                    "direction": get_direction_text(coeff),
                    "description": anatomy["description"],
                    "clinical": anatomy["clinical"],
                }
                findings.append(finding)
    
    # Sort by severity (absolute sigma)
    findings.sort(key=lambda x: abs(x["sigma"]), reverse=True)
    return findings


def print_analysis_report(result: FemurAnalysisResult) -> None:
    """Print a formatted analysis report to console."""
    print("\n" + "=" * 70)
    print("FEMUR SHAPE ANALYSIS REPORT")
    print("=" * 70)
    
    print(f"\nPatient file: {result.patient_file}")
    print(f"Mesh vertices: {result.n_vertices:,}")
    
    print("\n--- Distance Metrics ---")
    print(f"  Fréchet (geodesic) distance: {result.frechet_distance:.2f}")
    print(f"  Euclidean L2 total:          {result.l2_distance:.2f}")
    print(f"  Per-vertex RMSE:             {result.l2_rmse:.2f} mm")
    print(f"  Maximum deviation:           {result.l2_max:.2f} mm")
    
    # Deviation severity
    if result.l2_rmse < 2:
        severity = "NORMAL - within typical variation"
    elif result.l2_rmse < 5:
        severity = "MILD - slight deviation from mean"
    elif result.l2_rmse < 10:
        severity = "MODERATE - noticeable shape difference"
    else:
        severity = "SIGNIFICANT - substantial shape deviation"
    
    print(f"\n  Overall assessment: {severity}")
    
    print("\n--- Anatomical Component Analysis ---")
    print(f"  {'PC':<6} {'Component':<24} {'σ':<8} {'Level':<12}")
    print("  " + "-" * 55)
    
    for i in range(min(7, len(result.pca_coefficients_std))):
        coeff = result.pca_coefficients_std[i]
        pc_num = i + 1
        level, _, emoji = get_deviation_level(coeff)
        
        # Get component name with parenthetical
        if pc_num in COMPONENT_ANATOMY:
            anat = COMPONENT_ANATOMY[pc_num]
            name_part = anat["short"]
            if anat.get("parens"):
                name_part += f" ({anat['parens']})"
        else:
            name_part = f"Mode {pc_num}"
        
        print(f"  PC{pc_num:<3} {name_part:<24} {coeff:>+6.2f}  {emoji} {level:<10}")
    
    # Clinical findings with complete sentences
    findings = get_clinical_findings(result)
    
    # Filter out low-significance components unless they have extreme values
    significant_findings = [f for f in findings 
                          if COMPONENT_ANATOMY.get(f['pc'], {}).get('significance', 'high') != 'low'
                          or abs(f['sigma']) >= 2.0]
    
    if significant_findings:
        print("\n--- Clinical Findings ---")
        for f in significant_findings:
            sentence = get_finding_sentence(f['pc'], f['sigma'])
            print(f"\n  {f['emoji']} {sentence}")
            print(f"     PC{f['pc']}: {f['sigma']:+.2f}σ")
            print(f"     → {f['clinical']}")
    else:
        print("\n--- Clinical Findings ---")
        print("  ✓ No clinically significant deviations detected.")
        print("    Shape falls within normal population variation.")
    
    print("\n--- Deviation Level Legend ---")
    print("  ✓ Normal (<0.5σ)  ○ Typical (<1σ)  △ Mild (<1.5σ)")
    print("  ▲ Moderate (<2σ)  ◆ Significant (<2.5σ)  ★ Extreme (≥2.5σ)")
    
    print("\n" + "=" * 70)


def visualize_analysis(
    result: FemurAnalysisResult,
    template_faces: np.ndarray,
    show_components: bool = True
) -> None:
    """
    Create interactive visualization of the analysis.
    
    Layout: 2 meshes on top row, findings panel on bottom row.
    """
    # Use 2x2 grid: top row for meshes, bottom row spans full width for findings
    plotter = pv.Plotter(shape=(2, 2), window_size=[1400, 900])
    
    def make_mesh(verts: np.ndarray, scalars: Optional[np.ndarray] = None, name: str = "scalar"):
        pv_faces = np.hstack([[3] + list(f) for f in template_faces])
        mesh = pv.PolyData(verts, pv_faces)
        if scalars is not None:
            mesh[name] = scalars
        return mesh
    
    # === TOP ROW: Two femur views ===
    
    # Plot patient first to establish camera
    plotter.subplot(0, 1)
    plotter.add_text(
        f"Patient: {result.patient_file}",
        font_size=16
    )
    patient_mesh = make_mesh(result.vertices, result.per_vertex_distances, "Deviation (mm)")
    plotter.add_mesh(
        patient_mesh,
        scalars="Deviation (mm)",
        cmap="coolwarm",
        smooth_shading=True,
        scalar_bar_args={"title": "Deviation (mm)", "n_labels": 5, "font_family": "arial", "title_font_size": 14, "label_font_size": 12}
    )
    plotter.reset_camera()  # type: ignore[no-untyped-call]
    plotter.camera.zoom(0.85)
    camera_position = plotter.camera_position
    
    # Atlas (mean shape) - same camera
    plotter.subplot(0, 0)
    plotter.add_text("Population Mean (Atlas)", font_size=16)
    atlas_mesh = make_mesh(result.atlas_vertices)
    plotter.add_mesh(atlas_mesh, color="lightgray", smooth_shading=True)
    plotter.camera_position = camera_position
    
    # === BOTTOM ROW: Findings panel (spans both columns conceptually) ===
    
    if show_components:
        # =====================================================================
        # LEFT PANEL: Component Analysis
        # =====================================================================
        plotter.subplot(1, 0)
        
        y_pos = 0.97
        
        # Title - LARGE
        plotter.add_text("COMPONENT ANALYSIS", position=(0.03, y_pos), font_size=24, viewport=True, color="white")
        y_pos -= 0.10
        
        # Severity scale - formatted nicely with level names
        plotter.add_text("SEVERITY SCALE:", position=(0.03, y_pos), font_size=16, viewport=True, color="#cccccc")
        y_pos -= 0.055
        # First row of scale
        plotter.add_text(
            "✓ Normal (<0.5σ)    ○ Typical (<1σ)    △ Mild (<1.5σ)",
            position=(0.03, y_pos), font_size=14, viewport=True, color="#999999"
        )
        y_pos -= 0.045
        # Second row of scale
        plotter.add_text(
            "▲ Moderate (<2σ)    ◆ Significant (<2.5σ)    ★ Extreme (≥2.5σ)",
            position=(0.03, y_pos), font_size=14, viewport=True, color="#999999"
        )
        y_pos -= 0.065
        
        # Component data - each component on its own line, LARGE font
        for i in range(min(7, len(result.pca_coefficients_std))):
            coeff = result.pca_coefficients_std[i]
            pc_num = i + 1
            level, _, emoji = get_deviation_level(coeff)  # color not needed here
            
            # Get component name with parenthetical
            if pc_num in COMPONENT_ANATOMY:
                anat = COMPONENT_ANATOMY[pc_num]
                name_part = anat["short"]
                if anat.get("parens"):
                    name_part += f" ({anat['parens']})"
            else:
                name_part = f"Mode {pc_num}"
            
            # Format each row clearly - LARGE FONT
            line = f"PC{pc_num}   {name_part:<26}  {coeff:>+5.2f}σ   {emoji} {level}"
            plotter.add_text(line, position=(0.03, y_pos), font_size=18, viewport=True)
            y_pos -= 0.072
        
        # =====================================================================
        # RIGHT PANEL: Clinical Findings  
        # =====================================================================
        plotter.subplot(1, 1)
        
        y_pos = 0.97
        
        # Title - LARGE
        plotter.add_text("CLINICAL FINDINGS", position=(0.03, y_pos), font_size=24, viewport=True, color="white")
        y_pos -= 0.10
        
        # Summary metrics - larger
        plotter.add_text(
            f"RMSE: {result.l2_rmse:.1f} mm    Fréchet: {result.frechet_distance:.1f}",
            position=(0.03, y_pos), font_size=16, viewport=True, color="#cccccc"
        )
        y_pos -= 0.08
        
        findings = get_clinical_findings(result)
        
        if findings:
            # Only show components with significance >= "low" threshold
            significant_findings = [f for f in findings 
                                   if COMPONENT_ANATOMY.get(f['pc'], {}).get('significance', 'high') != 'low'
                                   or abs(f['sigma']) >= 2.0]
            
            if significant_findings:
                for f in significant_findings[:3]:  # Top 3 findings (more space per finding)
                    # Get finding sentence
                    sentence = get_finding_sentence(f['pc'], f['sigma'])
                    
                    # Main finding - LARGE
                    plotter.add_text(
                        f"{f['emoji']} {sentence}",
                        position=(0.03, y_pos), font_size=20, viewport=True
                    )
                    y_pos -= 0.07
                    
                    # PC info
                    plotter.add_text(
                        f"     PC{f['pc']}: {f['sigma']:+.2f}σ",
                        position=(0.03, y_pos), font_size=15, viewport=True, color="#aaaaaa"
                    )
                    y_pos -= 0.055
                    
                    # Clinical relevance - full text
                    plotter.add_text(
                        f"     → {f['clinical']}",
                        position=(0.03, y_pos), font_size=14, viewport=True, color="#888888"
                    )
                    y_pos -= 0.085
            else:
                plotter.add_text(
                    "✓ No clinically significant deviations",
                    position=(0.03, y_pos), font_size=20, viewport=True, color="#2ecc71"
                )
        else:
            plotter.add_text(
                "✓ Shape within normal population range",
                position=(0.03, y_pos), font_size=20, viewport=True, color="#2ecc71"
            )
            y_pos -= 0.12
            plotter.add_text(
                "All anatomical parameters fall within typical variation.",
                position=(0.03, y_pos), font_size=16, viewport=True, color="#aaaaaa"
            )
    
    # Link only the top two views (the mesh views)
    plotter.link_views()
    plotter.show()


def create_component_bar_chart(result: FemurAnalysisResult, output_path: Optional[str] = None) -> None:
    """Create a matplotlib bar chart of PCA component contributions with anatomical labels."""
    n_components = min(7, len(result.pca_coefficients_std))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Component coefficients with anatomical labels
    coeffs = result.pca_coefficients_std[:n_components]
    
    # Color by severity level
    colors = []
    for c in coeffs:
        level, color, _ = get_deviation_level(c)
        colors.append(color)
    
    # Anatomical labels
    labels = []
    for i in range(n_components):
        pc_num = i + 1
        if pc_num in COMPONENT_ANATOMY:
            labels.append(f"PC{pc_num}\n{COMPONENT_ANATOMY[pc_num]['short']}")
        else:
            labels.append(f"PC{pc_num}")
    
    bars = ax1.barh(range(n_components), coeffs, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axvline(x=0, color='black', linewidth=0.5)
    
    # Add severity zone shading
    ax1.axvspan(-0.5, 0.5, alpha=0.1, color='green', label='Normal')
    ax1.axvspan(-1.0, -0.5, alpha=0.1, color='blue')
    ax1.axvspan(0.5, 1.0, alpha=0.1, color='blue', label='Typical')
    ax1.axvspan(-1.5, -1.0, alpha=0.1, color='orange')
    ax1.axvspan(1.0, 1.5, alpha=0.1, color='orange', label='Mild')
    ax1.axvspan(-2.0, -1.5, alpha=0.1, color='#e67e22')
    ax1.axvspan(1.5, 2.0, alpha=0.1, color='#e67e22', label='Moderate')
    ax1.axvspan(-2.5, -2.0, alpha=0.1, color='red')
    ax1.axvspan(2.0, 2.5, alpha=0.1, color='red', label='Significant')
    
    ax1.set_yticks(range(n_components))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Deviation from Mean (σ)')
    ax1.set_title('Anatomical Shape Analysis')
    ax1.invert_yaxis()
    ax1.set_xlim(-3.5, 3.5)
    
    # Add value labels on bars
    for i, (bar, coeff) in enumerate(zip(bars, coeffs)):
        level, _, emoji = get_deviation_level(coeff)
        width = bar.get_width()
        x_pos = width + 0.1 if width >= 0 else width - 0.1
        ha = 'left' if width >= 0 else 'right'
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f'{emoji} {level}', ha=ha, va='center', fontsize=8)
    
    # Clinical findings panel
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    findings = get_clinical_findings(result)
    
    # Filter low-significance components (consistent with other outputs)
    significant_findings = [f for f in findings 
                          if COMPONENT_ANATOMY.get(f['pc'], {}).get('significance', 'high') != 'low'
                          or abs(f['sigma']) >= 2.0]
    
    # Title
    ax2.text(0.5, 0.95, 'Clinical Findings', fontsize=14, fontweight='bold',
             ha='center', transform=ax2.transAxes)
    
    y_pos = 0.85
    if significant_findings:
        for f in significant_findings:
            # Use sentence-based finding
            sentence = get_finding_sentence(f['pc'], f['sigma'])
            ax2.text(0.05, y_pos, f"{f['emoji']} {sentence}", fontsize=10, fontweight='bold',
                    transform=ax2.transAxes)
            y_pos -= 0.06
            
            # PC info
            ax2.text(0.08, y_pos, f"PC{f['pc']}: {f['sigma']:+.2f}σ",
                    fontsize=9, transform=ax2.transAxes, color='gray')
            y_pos -= 0.05
            
            # Clinical relevance
            ax2.text(0.08, y_pos, f"→ {f['clinical']}", fontsize=8, transform=ax2.transAxes,
                    style='italic', color='gray')
            y_pos -= 0.08
    else:
        ax2.text(0.5, 0.5, '✓ All parameters within\nnormal population range',
                fontsize=12, ha='center', va='center', transform=ax2.transAxes,
                color='green')
    
    # Legend at bottom
    legend_text = "Severity: ✓Normal(<0.5σ) ○Typical(<1σ) △Mild(<1.5σ) ▲Moderate(<2σ) ◆Significant(<2.5σ) ★Extreme(≥2.5σ)"
    ax2.text(0.5, 0.02, legend_text, fontsize=7, ha='center', transform=ax2.transAxes,
            color='gray')
    
    plt.suptitle(f'Femur Shape Analysis: {result.patient_file}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved component chart to {output_path}")
    else:
        plt.show()


def save_analysis_json(result: FemurAnalysisResult, output_path: str) -> None:
    """Save analysis results to JSON file."""
    data = {
        "patient_file": result.patient_file,
        "n_vertices": result.n_vertices,
        "metrics": {
            "frechet_distance": result.frechet_distance,
            "l2_distance": result.l2_distance,
            "l2_rmse": result.l2_rmse,
            "l2_max": result.l2_max,
        },
        "pca_components": {
            f"PC{i+1}": {
                "coefficient_sigma": float(result.pca_coefficients_std[i]),
                "coefficient_raw": float(result.pca_coefficients[i]),
                "variance_explained": float(result.explained_variance_ratio[i]),
            }
            for i in range(len(result.pca_coefficients_std))
        },
        "interpretations": interpret_pca_components(result),
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved analysis to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Medical femur shape analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis with visualization
    python medical_femur_analysis.py data/training/L_Femur_13_DECIM.obj.FINAL.obj
    
    # Save report to file
    python medical_femur_analysis.py patient.obj --output report/
    
    # Quick analysis without visualization
    python medical_femur_analysis.py patient.obj --no-visualize
"""
    )
    parser.add_argument("femur", help="Path to patient femur OBJ file")
    parser.add_argument("--model", type=str, default="scripts/pca/model/tangent_pca",
                        help="Path to Tangent PCA model directory")
    parser.add_argument("--template", type=str, 
                        default="data/training/L_Femur_11_DECIM.obj.FINAL.obj",
                        help="Template mesh for faces")
    parser.add_argument("--sigma", type=float, default=10.0,
                        help="Kernel bandwidth for Fréchet distance")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for reports")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Skip 3D visualization")
    parser.add_argument("--no-chart", action="store_true",
                        help="Skip matplotlib chart")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Load Tangent PCA model
    model_path = args.model if Path(args.model).is_absolute() else str(project_root / args.model)
    print(f"Loading Tangent PCA model from {model_path}...")
    tangent_pca = TangentPCA.load(model_path)
    
    # Load template for faces
    template_path = args.template if Path(args.template).is_absolute() else str(project_root / args.template)
    _, template_faces = load_obj_mesh(template_path)
    
    # Analyze femur
    femur_path = args.femur if Path(args.femur).is_absolute() else str(project_root / args.femur)
    print(f"Analyzing {femur_path}...")
    
    result = analyze_femur(femur_path, tangent_pca, sigma=args.sigma)
    
    # Print report
    print_analysis_report(result)
    
    # Save outputs if requested
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        save_analysis_json(result, str(output_dir / f"{result.patient_file}_analysis.json"))
        
        # Save chart
        if not args.no_chart:
            create_component_bar_chart(result, str(output_dir / f"{result.patient_file}_components.png"))
    
    # Visualizations
    if not args.no_chart and not args.output:
        create_component_bar_chart(result)
    
    if not args.no_visualize:
        visualize_analysis(result, template_faces)
    
    return result


if __name__ == "__main__":
    main()
