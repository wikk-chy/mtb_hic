#!/usr/bin/env python3
"""R4C5 polymer_openmm - OpenMM ring-polymer simulation of the Mtb chromosome.

Model
-----
Circular bead-spring polymer, N = 441 beads (2 x 5 kb bins per bead,
10 kb/bead; H37Rv 4,411,532 bp -> 883 Hi-C bins). All energies in kJ/mol,
written as multiples of kT = 2.494 kJ/mol (T = 300 K); lengths in units of
the bead diameter sigma = 1 (physically ~ 50 nm if 5 kb ~ 50 nm of compacted
nucleoid; only the confinement aspect ratio and volume fraction depend on
this mapping). Bead 0 = ori (genomic coordinate 0), bead 203 = ter
(2.03 Mb / 10 kb).

Force field (hand-written OpenMM forces; polychrom is not available for
py3.9):
  bonds       HarmonicBondForce, ring-closed, r0 = 1, k = 100 kT/sigma^2
  bending     HarmonicAngleForce on all triplets (i-1, i, i+1), theta0 = pi,
              k_ang = scenario-dependent (0 at baseline)
  excluded    WCA core: 4*eps*[(sigma/r)^12 - (sigma/r)^6] + eps for
              r < 2^(1/6), eps = 1 kT; zero beyond (step-function clamp)
  attraction  (S1, S3) Gaussian tail for r > 2^(1/6):
              U_att = -epsA * [exp(-((r - 1.122)/w)^2) - exp(-((rc - 1.122)/w)^2)],
              w = 0.6, clamped to zero at rc = 2.0 (energy-continuous)
  confinement cylinder (rod-shaped Mtb cell, ~3 um x 0.5 um -> aspect 6:1;
              Le et al./Barbieri et al. bacterial nucleoid geometry):
              harmonic walls kw = 10 kT/sigma^2, radius R = 5*scale,
              half-length L2 = 30*scale (scale fitted to WT P(s))
  tether      (S2, S3app) HarmonicBondForce between bead 0 and bead 203,
              rest length 1.0, stiffness k_t = strength2 (S3app) or
              strength (S2)
  integrator  LangevinMiddleIntegrator (BAOAB-like), T = 300 K,
              gamma = 1/ps, dt = 0.01 ps after a warmup ramp
              (0.002 -> 0.005 -> 0.01 ps); stable, no NaN over >4e6 steps

Scenarios (--scenario):
  baseline  confined flexible ring (fit: confinement scale vs WT P(s))
  S1        baseline + global non-specific attraction epsA = strength
            (poor-solvent collapse; no ori-ter-specific term)
  S2        baseline + ori-ter harmonic tether, k_t = strength
  S3        baseline + toroid folding dial lambda = strength:
            k_ang = 4*lambda kT/rad^2 AND epsA = 0.3*lambda kT
            (bending + weak attraction = classic toroid route; the two are
            coupled into a single dial to keep the sweep one-dimensional)
  S3app     S3 at fixed lambda = strength, plus ori-ter apposition tether
            k_app = strength2

CLI: python 02_simulate.py --scenario S1 --strength 0.5 [--strength2 10]
     [--scale 1.0] [--seed 42] [--equil 300000] [--nconf 1000]
     [--sample-every 3000] [--threads 2]

Output: runs/<scenario>_str<strength>_str2<strength2>_scale<scale>_seed<seed>.npz
  positions (nconf x N x 3 float32), production traces (energy, Rg,
  ori-ter distance, FISH pair distances), burn-in traces, parameter dict.
"""
import argparse
import json
import os
import time
import numpy as np
from openmm import (
    System, HarmonicBondForce, HarmonicAngleForce, CustomNonbondedForce,
    CustomExternalForce, CustomBondForce, LangevinMiddleIntegrator,
    Context, Platform, LocalEnergyMinimizer,
)
from openmm.unit import kilojoule_per_mole

KT = 2.494                    # kJ/mol at 300 K
N = 441                       # beads (10 kb/bead)
TER_BEAD = 203                # 2.03 Mb / 10 kb
RC_WCA = 2.0 ** (1.0 / 6.0)   # 1.122462
W_ATT = 0.6                   # Gaussian attraction tail width
RC_ATT = 2.0                  # attraction cutoff (tail ~0.12 epsA here,
                              # clamped to zero -> energy-continuous)
KBOND = 100.0                 # kT/sigma^2
KWALL = 10.0                  # kT/sigma^2
BASE_R, BASE_L2 = 5.0, 30.0   # confinement at scale = 1 (aspect 6:1)
KANG_PER_LAM = 4.0            # S3: k_ang = 4*lambda kT/rad^2
EPSA_PER_LAM = 0.3            # S3: epsA = 0.3*lambda kT

# FISH-proposal pairs (bead indices): ori-ter, mirror-arm pairs at
# 0.5/1.0/1.5 Mb (i, N-i), same-arm controls at the same distances
FISH_PAIRS = {
    "ori_ter": (0, TER_BEAD),
    "mirror_0.5Mb": (25, N - 25),
    "mirror_1.0Mb": (50, N - 50),
    "mirror_1.5Mb": (75, N - 75),
    "ctrl_0.5Mb": (25, 75),
    "ctrl_1.0Mb": (25, 125),
    "ctrl_1.5Mb": (25, 175),
}

ATT_TAIL = (  # attractive Gaussian tail, continuous, zero beyond cutoff
    "step(rc-r)*(-epsA*(exp(-((r-1.122462)/w)^2)"
    " - exp(-((rc-1.122462)/w)^2)))")


def build_system(scenario, strength, strength2, scale):
    sys_ = System()
    for _ in range(N):
        sys_.addParticle(1.0)

    bond = HarmonicBondForce()
    for i in range(N):
        bond.addBond(i, (i + 1) % N, 1.0, KBOND * KT)
    sys_.addForce(bond)

    eps_a = 0.0
    if scenario == "S1":
        eps_a = strength
    elif scenario in ("S3", "S3app"):
        eps_a = EPSA_PER_LAM * strength

    # WCA core with its own short-range neighbour list (fast)
    nb = CustomNonbondedForce("4*eps*((sigma/r)^12-(sigma/r)^6)+eps")
    nb.addGlobalParameter("eps", KT)
    nb.addGlobalParameter("sigma", 1.0)
    nb.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
    nb.setCutoffDistance(RC_WCA)
    for _ in range(N):
        nb.addParticle([])
    sys_.addForce(nb)

    # attraction tail as a separate force with its own neighbour list
    # (split because a single rc = 3 sigma force is ~10x slower)
    if eps_a > 0:
        att = CustomNonbondedForce(ATT_TAIL)
        att.addGlobalParameter("epsA", eps_a * KT)
        att.addGlobalParameter("w", W_ATT)
        att.addGlobalParameter("rc", RC_ATT)
        att.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
        att.setCutoffDistance(RC_ATT)
        for _ in range(N):
            att.addParticle([])
        sys_.addForce(att)

    if scenario in ("S3", "S3app"):
        ang = HarmonicAngleForce()
        k_ang = KANG_PER_LAM * strength * KT
        for i in range(N):
            ang.addAngle((i - 1) % N, i, (i + 1) % N, np.pi, k_ang)
        sys_.addForce(ang)

    tether_k = 0.0
    if scenario == "S2":
        tether_k = strength
    elif scenario == "S3app":
        tether_k = strength2
    if tether_k > 0:
        # CustomBondForce with a global kt_scale that is ramped 0 -> 1
        # over burn-in: applying a short-rest-length tether at full
        # strength to the stretched initial chain blows up (NaN).
        teth = CustomBondForce("kt_scale*0.5*kt*(r-r0)^2")
        teth.addGlobalParameter("kt_scale", 0.0)
        teth.addPerBondParameter("kt")
        teth.addPerBondParameter("r0")
        teth.addBond(0, TER_BEAD, [tether_k * KT, 1.0])
        sys_.addForce(teth)

    conf = CustomExternalForce(
        "kw*( step(rho-R)*(rho-R)^2 + step(az-L2)*(az-L2)^2 );"
        " rho=sqrt(x^2+y^2); az=abs(z)")
    conf.addGlobalParameter("kw", KWALL * KT)
    conf.addGlobalParameter("R", BASE_R * scale)
    conf.addGlobalParameter("L2", BASE_L2 * scale)
    for i in range(N):
        conf.addParticle(i, [])
    sys_.addForce(conf)
    return sys_


def init_helix(scale):
    """20-turn helix inside the cylinder; no overlaps, bond lengths ~1."""
    th = np.linspace(0, 2 * np.pi * 20, N)
    x = np.stack([0.7 * BASE_R * scale * np.cos(th),
                  0.7 * BASE_R * scale * np.sin(th),
                  np.linspace(-(BASE_L2 * scale - 2), BASE_L2 * scale - 2, N)],
                 axis=1)
    # rescale so mean bond length ~ 1
    d = np.linalg.norm(np.roll(x, -1, axis=0) - x, axis=1)
    return x / d.mean()


def add_rephasing_restraints(sys_, scale):
    """Soft harmonic pulls on the ori bead (to z = -(L/2-3)) and the ter
    bead (to z = +(L/2-3)), applied only during the rephasing phase
    (kr_reph set to 0 afterwards). The ring-in-tube rephasing mode has a
    spontaneous relaxation time of 5e5+ steps; pulling the markers to the
    poles during initialisation removes this slow mode from burn-in."""
    zt = BASE_L2 * scale - 3
    f = CustomExternalForce("kr_reph*step(s*(z-zt))*(z-zt)^2")
    f.addGlobalParameter("kr_reph", 3.0 * KT)
    f.addPerParticleParameter("s")
    f.addPerParticleParameter("zt")
    f.addParticle(0, [1.0, -zt])
    f.addParticle(TER_BEAD, [-1.0, zt])
    sys_.addForce(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True,
                    choices=["baseline", "S1", "S2", "S3", "S3app"])
    ap.add_argument("--strength", type=float, default=0.0)
    ap.add_argument("--strength2", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--equil", type=int, default=250_000)
    ap.add_argument("--nconf", type=int, default=700)
    ap.add_argument("--sample-every", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--init", default="longitudinal",
                    choices=["longitudinal", "helix"])
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tag = (f"{args.scenario}_str{args.strength:g}_str2{args.strength2:g}"
           f"_scale{args.scale:g}_seed{args.seed}"
           + ("_helixinit" if args.init == "helix" else ""))
    out_path = f"{args.outdir}/{tag}.npz"

    sys_ = build_system(args.scenario, args.strength, args.strength2,
                        args.scale)
    if args.init == "longitudinal":
        add_rephasing_restraints(sys_, args.scale)
    integ = LangevinMiddleIntegrator(300.0, 1.0, 0.01)
    ctx = Context(sys_, integ, Platform.getPlatformByName("CPU"),
                  {"Threads": str(args.threads)})
    ctx.setPositions(init_helix(args.scale))
    ctx.setVelocitiesToTemperature(300.0, args.seed)
    integ.setRandomNumberSeed(args.seed)

    t0 = time.time()
    LocalEnergyMinimizer.minimize(ctx, tolerance=100.0, maxIterations=2000)
    integ.setStepSize(0.002)
    integ.step(30_000)
    integ.setStepSize(0.005)
    integ.step(20_000)
    integ.setStepSize(0.01)
    if args.init == "longitudinal":
        integ.step(150_000)          # restrained rephasing to the poles
        ctx.setParameter("kr_reph", 0.0)
        x = np.asarray(ctx.getState(getPositions=True)
                       .getPositions(asNumpy=True))
        print(f"{tag}: after rephasing z_ori={x[0,2]:.1f} "
              f"z_ter={x[TER_BEAD,2]:.1f} "
              f"d_ot={np.linalg.norm(x[0]-x[TER_BEAD]):.1f}", flush=True)

    fish_names = list(FISH_PAIRS)
    fish_idx = np.array([FISH_PAIRS[k] for k in fish_names])

    def observables(x, e):
        rg = np.sqrt(((x - x.mean(0)) ** 2).sum(1).mean())
        fp = np.linalg.norm(x[fish_idx[:, 0]] - x[fish_idx[:, 1]], axis=1)
        return e, rg, fp

    # ---- burn-in with diagnostics every 2000 steps ----
    burn_e, burn_rg, burn_ot = [], [], []
    n_burn_rec = args.equil // 2000
    has_tether = (args.scenario == "S2" and args.strength > 0) or \
        (args.scenario == "S3app" and args.strength2 > 0)
    for _ib in range(n_burn_rec):
        if has_tether:      # adiabatic tether ramp across burn-in
            ctx.setParameter("kt_scale", min(1.0, (_ib + 1) / n_burn_rec))
        integ.step(2000)
        st = ctx.getState(getPositions=True, getEnergy=True)
        x = np.asarray(st.getPositions(asNumpy=True))
        e, rg, fp = observables(x, st.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
        burn_e.append(e)
        burn_rg.append(rg)
        burn_ot.append(fp[0])

    # ---- production ----
    positions = np.empty((args.nconf, N, 3), dtype=np.float32)
    prod_e = np.empty(args.nconf)
    prod_rg = np.empty(args.nconf)
    prod_fish = np.empty((args.nconf, len(fish_names)))
    for c in range(args.nconf):
        integ.step(args.sample_every)
        st = ctx.getState(getPositions=True, getEnergy=True)
        x = np.asarray(st.getPositions(asNumpy=True))
        positions[c] = x
        e, rg, fp = observables(x, st.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
        prod_e[c], prod_rg[c], prod_fish[c] = e, rg, fp
        if (c + 1) % 100 == 0:
            print(f"{tag}: {c + 1}/{args.nconf} confs "
                  f"({time.time() - t0:.0f}s)", flush=True)

    params = dict(scenario=args.scenario, init=args.init, strength=args.strength,
                  strength2=args.strength2, scale=args.scale,
                  seed=args.seed, equil=args.equil, nconf=args.nconf,
                  sample_every=args.sample_every, N=N, KT=KT,
                  kbond=KBOND, kwall=KWALL, base_R=BASE_R, base_L2=BASE_L2,
                  eps_a=(args.strength if args.scenario == "S1"
                         else EPSA_PER_LAM * args.strength
                         if args.scenario in ("S3", "S3app") else 0.0),
                  k_ang=(KANG_PER_LAM * args.strength
                         if args.scenario in ("S3", "S3app") else 0.0),
                  k_tether=(args.strength if args.scenario == "S2"
                            else args.strength2
                            if args.scenario == "S3app" else 0.0),
                  runtime_s=time.time() - t0)
    np.savez_compressed(out_path, positions=positions,
                        prod_e=prod_e, prod_rg=prod_rg, prod_fish=prod_fish,
                        burn_e=np.array(burn_e), burn_rg=np.array(burn_rg),
                        burn_ot=np.array(burn_ot),
                        fish_names=np.array(fish_names),
                        params=json.dumps(params))
    print(f"{tag}: done in {params['runtime_s']:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
