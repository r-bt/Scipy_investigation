import numpy as np
import sympy as sym
import os

import pickle

import src.sol.fluids as fluids
import src.mms.MMS as MS

from src.msh.mesh import Mesh2D
from src.msh.mk import struct2D
import src.datawrite.dataWriter as dw
import logging
logging.disable(logging.CRITICAL)

# suppress stupid deprecation warnings
import warnings
warnings.simplefilter('ignore')

#Create MS

nu = 0.05
x, y, t = sym.symbols('x y t')
π = sym.pi
k = π

# analytical pressure
p = - 1/4 * (sym.cos(2*π*x) + sym.cos(2*π*y))*sym.exp(-4*nu*π**2*t)

Δt = 5e-8
#T_final = 2.5e-6
steps = 5
T_final = steps * Δt

# analytical velocity
u = - sym.cos(π*x)*sym.sin(π*y)*sym.exp(-2*nu*π**2*t)
v =  sym.sin(π*x)*sym.cos(π*y)*sym.exp(-2*nu*π**2*t)
vel = [u, v]

ins_ms = MS.INS_MS(vel, p, nu)
fn_ms = ins_ms.lambdified_MS

p_vec = ins_ms.lambdified_MS['p']
u_vec = ins_ms.lambdified_MS['v'][0]
v_vec = ins_ms.lambdified_MS['v'][1]

p_order = 3

def clean_mat():
    for f in os.listdir():
        if f.endswith(".mat"):
            os.remove(f)

def INS_test_run(mesh, u_LSO, dp_LSO, out_file, p, Δt, T_final):
    #  init Setup
    setup = fluids.Setup()
    setup.output_dir = '.'

    # add the mesh
    setup.mesh = mesh

    # Step 2: Boundary 0 = left; 1, right; 2, bottom; 3, top
    TOL = 1e-6
    boundary_ids = [lambda x: abs(x[:, 0] - (0.0)) < TOL,
                    lambda x: abs(x[:, 0] - (1.0)) < TOL,
                    lambda x: abs(x[:, 1] - (0.0)) < TOL,
                    lambda x: abs(x[:, 1] - (1.0)) < TOL]
    setup.set_meshid(boundary_ids)

    # set number of fields, (V,P,T)
    setup.num_fields = (1, 1, 0)

    # set Dirichlet boundaries on the predictor velocity
    # zero Neumann boundaries on the pressure correction
    bctype_t, bctype_v, bctype_dp = setup.get_bcid2type_arrays()
    bctype_dp[:] = 1
    setup.set_bctype(bctype_t, bctype_v, bctype_dp)

    # set bc value functions obeying MS
    u_bcval_fn = lambda x, t, bcid: MS.eval_at_dgnodes(u_vec, x, t)
    v_bcval_fn = lambda x, t, bcid: MS.eval_at_dgnodes(v_vec, x, t)
    dp_bcval_fn = lambda x, t, bcid: 0 * x[:, 0]  # set this to zero everywhere

    setup.set_bc_value_functions(vector_bcs=[[u_bcval_fn, v_bcval_fn]], p_bcs=[dp_bcval_fn])
    setup.set_bc_value_functions(vector_bcs=[[u_bcval_fn, v_bcval_fn]], p_bcs=[dp_bcval_fn])

    # null forcing function
    force_ex = lambda sol, t: [np.zeros_like(arr) for arr in sol.trace]

    p_IC = [lambda x: MS.eval_at_dgnodes(p_vec, x, t=0)]
    velocity_IC = [[lambda x: MS.eval_at_dgnodes(u_vec, x, t=0),
                    lambda x: MS.eval_at_dgnodes(v_vec, x, t=0)]]
    setup.set_initial_value_functions(vector_init=velocity_IC, p_init=p_IC)

    # set timestepping options
    setup.set_time_options('IMEX', time_start=0, time_end=T_final, dt=Δt, params=[2, 2, 2])
    time_steps = int(T_final / Δt)

    # set diffusion parameter
    setup.set_diffusion_params(nu=nu)

    # stability parameters
    ℓ = 1  # problem length scale
    τ = 1000
    # τ_p = 1/(1.* Δt * τ)
    aii = 0.4358665215
    τ_p = 1. / (aii * τ * Δt)
    setup.set_hdg_stability(tau_d=τ, tau_p=τ_p)

    # set solver options
    setup.set_solver_options('NAVIER-STOKES',
                             'TIME_DEPENDENT',
                             #                         'NODAL_TVD',
                             'TIME_DEPENDENT_BCS')

    # nodal limiter
    # setup.sf = 1
    # setup.alpha = 8
    # setup.alphatop = 4
    # setup.forder = 2

    setup.linearSolverOptions['P'] = dp_LSO
    setup.linearSolverOptions['vecs'] = u_LSO

    # explicitly shut off PPE
    setup.ppe_bool = False
    setup.validate()
    setup.make_sol(degree=p)
    setup.restart = False

    solver = fluids.Solver(setup)
    solver._time_debug = True
    ncdw = dw.ncDataWriter_MSEAS3DHDG(solver.sol, outFile=out_file, outputFreq=steps)
    solver.dataWriters = [ncdw]
    solver.do_rotcor = True
    solver.solve()
    return solver

def solveMatrix(Nx):

    print(Nx)
    ## run with GMRES
    dp_LSO = fluids.LinearSolverOption(solverType='LGMRES', preconditioning='ILU')
    u_LSO = fluids.LinearSolverOption(solverType='LGMRES')
    outfile = 'INS_GMRES_p{}_h1r{}.nc'.format(p_order, Nx)

    # setup
    mask = 1 * np.ones((Nx, Nx))
    T, P = struct2D(mask, [0, 1], [0, 1])
    mesh = Mesh2D(elm=T, vert=P)

    # INS run + cleanup
    GMRES_solver = INS_test_run(mesh, u_LSO, dp_LSO, outfile, p_order, Δt, T_final)
    clean_mat()

    return(GMRES_solver)

if __name__ == "__main__":

    solver = solveMatrix(10)

    A = solver.project.hdgker[0].it_A

    b = solver.project.hdgker[0].it_b

    f = open("10x10", "ab")
    pickle.dump(A, f)
    f.close()

    f = open("10x10_solution", "ab")
    np.save(f, b)
    f.close()