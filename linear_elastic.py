#!/usr/bin/env python
from optparse import OptionParser
import numpy as nm

import sys
from matplotlib.sphinxext.mathmpl import options_spec
import matplotlib.pyplot as plt
sys.path.append('.')

from sfepy.base.base import assert_, output, ordered_iteritems, IndexedStruct
from sfepy.base.base import IndexedStruct
from sfepy.discrete import (FieldVariable, Material, Integral, Function,
                            Equation, Equations, Problem)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect,ScipyIterative,PyAMGSolver,PETScKrylovSolver
from sfepy.solvers.nls import Newton
#from sfepy.postprocess.viewer import Viewer
from sfepy.mesh import mesh_generators
from sfepy.terms.terms_elastic import CauchyStressTerm
from sfepy.mechanics.matcoefs import stiffness_from_lame,stiffness_from_youngpoisson
from sfepy.mechanics.tensors import get_von_mises_stress
from sfepy.base.base import Struct
from sfepy.discrete.probes import LineProbe
from its2D_2 import stress_strain
from its2D_3 import nodal_stress
from sfepy.discrete.projections import project_by_component
import uuid
import os
from pyamg import smoothed_aggregation_solver
from scipy.sparse.linalg import bicgstab, spilu

def gen_lines(problem):
    """
    Define two line probes.

    Additional probes can be added by appending to `ps0` (start points) and
    `ps1` (end points) lists.
    """
    ps0 = [[-0.4,  -0.5, 0]]
    ps1 = [[0.4, -0.5, 0]]

    # Use enough points for higher order approximations.
    n_point = 30

    labels = ['%s -> %s' % (p0, p1) for p0, p1 in zip(ps0, ps1)]
    probes = []
    for ip in xrange(len(ps0)):
        p0, p1 = ps0[ip], ps1[ip]
        probes.append(LineProbe(p0, p1, n_point))

    return probes, labels

def probe_results(u, strain, stress, probe, label):
    """
    Probe the results using the given probe and plot the probed values.
    """
    results = {}

    pars, vals = probe(u)
    results['u'] = (pars, vals)
    pars, vals = probe(strain)
    results['cauchy_strain'] = (pars, vals)
    pars, vals = probe(stress)
    results['cauchy_stress'] = (pars, vals)

    fig = plt.figure()
    plt.clf()
    fig.subplots_adjust(hspace=0.4)
    plt.subplot(311)
    pars, vals = results['u']
    for ic in range(vals.shape[1]):
        plt.plot(pars, vals[:,ic], label=r'$u_{%d}$' % (ic + 1),
                 lw=1, ls='-', marker='+', ms=3)
    plt.ylabel('displacements')
    plt.xlabel('probe %s' % label, fontsize=8)
    plt.legend(loc='best', fontsize=10)

    sym_indices = ['H', 'Shear','V']

    plt.subplot(312)
    pars, vals = results['cauchy_strain']
#r_stress = [0,4,8]
    for ic in range(vals.shape[1]):
        plt.plot(pars, vals[:,ic], label=r'$e_{%s}$' % sym_indices[ic],
                 lw=1, ls='-', marker='+', ms=3)
    plt.ylabel('Cauchy strain')
    plt.xlabel('probe %s' % label, fontsize=8)
    plt.legend(loc='best', fontsize=10)

    plt.subplot(313)
    pars, vals = results['cauchy_stress']
    for ic in range(vals.shape[1]):
        plt.plot(pars, vals[:,ic], label=r'$\sigma_{%s}$' % sym_indices[ic],
                 lw=1, ls='-', marker='+', ms=3)
    plt.ylabel('Cauchy stress')
    plt.xlabel('probe %s' % label, fontsize=8)
    plt.legend(loc='best', fontsize=10)

    return fig, results

def probe_results2(u, strain, stress, probe, label):
    """
        Probe the results using the given probe and plot the probed values.
        """
    results = {}
    
    pars, vals = probe(u)
    results['u'] = (pars, vals)
    pars, vals = probe(strain)
    results['cauchy_strain'] = (pars, vals)
    pars, vals = probe(stress)
    results['cauchy_stress'] = (pars, vals)
    
    fig = plt.figure()
    plt.clf()
    

    sym_indices = ['Shear', 'H','V']
    
    plt.subplot(111)
    
    pars, vals = results['cauchy_stress']
    for ic in range(vals.shape[1]):
        plt.plot(pars, vals[:,ic], label=r'$\sigma_{%s}$' % sym_indices[ic],
                 lw=1, ls='-', marker='+', ms=3)
    plt.ylabel('Cauchy stress')
    plt.xlabel('probe %s' % label, fontsize=8)
    plt.legend(loc='best', fontsize=10)
    
    return fig, results


def shift_u_fun(ts, coors, bc=None, problem=None, shift=0.0):
    """
    Define a displacement depending on the y coordinate.
    """
    val = 0.000001 #shift * coors[:,1]**2

    return val

usage = """%prog [options]"""
help = {
    'show' : 'show the results figure',
}

def get_mat(coors, mode, pb):
    if mode == 'qp':
        cnf = pb.conf
        # get material coefficients
        
            # given values
        E_f, nu_f, E_m, nu_m  = 160.e9, 0.2, 5.e9, 0.45

        nqp = coors.shape[0]
        nel = pb.domain.mesh.n_el
        nqpe = nqp / nel
        out = nm.zeros((nqp, 6, 6), dtype=nm.float64)

        # set values - matrix
        D_m = stiffness_from_youngpoisson(3, E_m, nu_m)
        Ym = pb.domain.regions['Ym'].get_cells()
        idx0 = (nm.arange(nqpe)[:,nm.newaxis] * nm.ones((1, Ym.shape[0]),
                    dtype=nm.int32)).T.flatten()
        idxs = (Ym[:,nm.newaxis] * nm.ones((1, nqpe),
                    dtype=nm.int32)).flatten() * nqpe
        out[idxs + idx0,...] = D_m

        # set values - fiber
        D_f = stiffness_from_youngpoisson(3, E_f, nu_f)
        Yf = pb.domain.regions['Yf'].get_cells()
        idx0 = (nm.arange(nqpe)[:,nm.newaxis] * nm.ones((1, Yf.shape[0]),
                    dtype=nm.int32)).T.flatten()
        idxs = (Yf[:,nm.newaxis] * nm.ones((1, nqpe),
                    dtype=nm.int32)).flatten() * nqpe
        out[idxs + idx0,...] = D_f

        return {'D': out}

def get_mat1(ts,coors, mode=None,**kwargs):
    if mode == 'qp':
        #cnf = pb.conf
        # get material coefficients
        
            # given values
            #E_f, nu_f, E_m, nu_m  = 160.e9, 0.2, 5.e9, 0.4

        nqp = coors.shape[0]
        #nel = pb.domain.mesh.n_el
        #nqpe = nqp / nel
        out = nm.zeros((nqp, 6, 6), dtype=nm.float64)
        D_m = stiffness_from_youngpoisson(3, E_m, nu_m)
        D_f = stiffness_from_youngpoisson(3, E_f, nu_f)
        out[0:nqp, ...] = D_f
        for i in nm.arange(nqp):
            if coors[i,2]>-0.1 and coors[i,2]<0.1:
                out[i,...] = D_m
        return {'D': out}
    
def get_mat2(ts,coors, mode=None,**kwargs):
    #if mode == 'qp':
        #cnf = pb.conf
        # get material coefficients
        
            # given values
    E_f, nu_f, E_m, nu_m  = 160.e9, 0.28, 5.e9, 0.45

    #nqp = coors.shape[0]
        #nel = pb.domain.mesh.n_el
        #nqpe = nqp / nel
    out = nm.zeros((6, 6), dtype=nm.float64)

        # set values - matrix
    D_m = stiffness_from_youngpoisson(3, E_m, nu_m)
        

        # set values - fiber
    D_f = stiffness_from_youngpoisson(3, E_f, nu_f)
         
    out[...] = D_m
    
    out[0:nqp/2,...] = D_f
        #out[nqp/2:nqp,...] = D_m


    return {'D': out}



dims = nm.array((1.0,1.0,1.0))
shape = nm.array((20,20,20))
NX=shape[0]
NY=shape[1]
NZ=shape[2]
centre = nm.array((0.0,0.0,0.0))

E_f, nu_f, E_m, nu_m  = 160.e9, 0.2, 5.e9, 0.3
#E_f, nu_f, E_m, nu_m  = 160.e9, 0.2, 160.e9, 0.2


def main():
    from sfepy import data_dir

    parser = OptionParser(usage=usage, version='%prog')
    parser.add_option('-s', '--show',
                      action="store_true", dest='show',
                      default=False, help=help['show'])
    options, args = parser.parse_args()
    options_probe = True
    folder = str(uuid.uuid4())
    os.mkdir(folder)
    os.chdir(folder)
    
    file = open('README.txt', 'w')
    file.write('DIMENSIONS\n')
    file.write('Lx = '+str(dims[0])+' Ly = '+str(dims[1])+' Lz = '+str(dims[2])+'\n')
    file.write('DISCRETIZATION (NX, NY, NZ)\n')
    file.write(str(NX)+'  '+str(NY)+'  '+str(NZ)+'\n')
    file.write('MATERIALS\n')
    file.write(str(E_f)+' '+str(nu_f)+' '+str(E_m)+' '+str(nu_m)+'\n')
    
    #mesh = Mesh.from_file(data_dir + '/meshes/2d/rectangle_tri.mesh')
    
    mesh = mesh_generators.gen_block_mesh(dims,shape,centre,name='block')
    domain = FEDomain('domain', mesh)

    min_x, max_x = domain.get_mesh_bounding_box()[:,0]
    min_y, max_y = domain.get_mesh_bounding_box()[:,1]
    min_z, max_z = domain.get_mesh_bounding_box()[:,2]
    eps = 1e-8 * (max_x - min_x)
    print min_x, max_x
    print min_y, max_y
    print min_z, max_z
    R1 = domain.create_region('Ym',
                                  'vertices in z < %.10f' % (max_z/2))
    R2 = domain.create_region('Yf',
                                  'vertices in z >= %.10f' % (min_z/2))
    omega = domain.create_region('Omega', 'all')
    gamma1 = domain.create_region('Left',
                                  'vertices in x < %.10f' % (min_x + eps), 
                                  'facet')
    gamma2 = domain.create_region('Right',
                                  'vertices in x > %.10f' % (max_x - eps),
                                  'facet')
    gamma3 = domain.create_region('Front',
                                  'vertices in y < %.10f' % (min_y + eps),
                                  'facet')
    gamma4 = domain.create_region('Back',
                                  'vertices in y > %.10f' % (max_y - eps),
                                  'facet')
    gamma5 = domain.create_region('Bottom',
                                  'vertices in z < %.10f' % (min_z + eps),
                                  'facet')
    gamma6 = domain.create_region('Top',
                                  'vertices in z > %.10f' % (max_z - eps),
                                  'facet')



    field = Field.from_args('fu', nm.float64, 'vector', omega, approx_order=2)

    u = FieldVariable('u', 'unknown', field)
    v = FieldVariable('v', 'test', field, primary_var_name='u')
    mu=1.1
    lam=1.0
    m = Material('m', lam=lam, mu=mu)
    f = Material('f', val=[[0.0], [0.0],[0.0]])
    #mu,lam=m.get_constants_mu_lam()
    #print mu.lam 
    D = stiffness_from_lame(3,lam, mu)    
    mat = Material('Mat', D=D)

    #D = stiffness_from_youngpoisson(2, options.young, options.poisson)
    get_mat = Function('get_mat1',get_mat1)
    #get_mat1=Function('get_mat', (lambda ts, coors, mode=None, problem=None, **kwargs:
    #                get_mat(coors, mode, problem)))
    #mat = Material('Mat', function=Function('get_mat1',get_mat1))
    #mat = Material('Mat', 'get_mat')
    integral = Integral('i', order=3)

    t1 = Term.new('dw_lin_elastic(Mat.D, v, u)',
         integral, omega, Mat=mat, v=v, u=u)
    t2 = Term.new('dw_volume_lvf(f.val, v)', integral, omega, f=f, v=v)
    eq = Equation('balance', t1 + t2)
    eqs = Equations([eq])

    fix_u = EssentialBC('fix_u', gamma1, {'u.all' : 0.0})
    left_bc  = EssentialBC('Left',  gamma1, {'u.0' : 0.0})
    right_bc = EssentialBC('Right', gamma2, {'u.0' : 0.0})
    back_bc = EssentialBC('Front', gamma3, {'u.1' : 0.0})
    front_bc = EssentialBC('Back', gamma4, {'u.1' : 0.0})
    bottom_bc = EssentialBC('Bottom', gamma5, {'u.all' : 0.0})
    top_bc = EssentialBC('Top', gamma6, {'u.2' : 0.2})

    bc=[left_bc,right_bc,back_bc,front_bc,bottom_bc,top_bc]
    #bc=[bottom_bc,top_bc]

    bc_fun = Function('shift_u_fun', shift_u_fun, extra_args={'shift' : 0.01})
    shift_u = EssentialBC('shift_u', gamma2, {'u.0' : bc_fun})
    #get_mat = Function('get_mat1',get_mat1)
    #mat = Material('Mat', function=Function('get_mat1',get_mat1))
    #ls = ScipyDirect({'method':'umfpack'})
    ##############################
    #  ##### SOLVER SECTION  #####
    ##############################
    
    # GET MATRIX FOR PRECONTITIONER #
    
    
    #ls = ScipyIterative({'method':'bicgstab','i_max':5000,'eps_r':1e-10})
    #ls = ScipyIterative({})
    
#ls = PyAMGSolver({'i_max':5000,'eps_r':1e-10})
#conf = Struct(method='cg', precond='gamg', sub_precond=None,i_max=10000, eps_a=1e-50, eps_r=1e-5, eps_d=1e4, verbose=True)
    #ls = PETScKrylovSolver({'method' : 'cg', 'precond' : 'icc', 'eps_r' : 1e-10, 'i_max' : 5000})
    conf = Struct(method='bcgsl', precond='jacobi', sub_precond=None,
                  i_max=10000, eps_a=1e-50, eps_r=1e-10, eps_d=1e4,
                  verbose=True)
                  #conf = Struct(method = 'cg', precond = 'icc', eps_r = 1e-10, i_max = 5000)
    ls = PETScKrylovSolver(conf)
#if hasattr(ls.name,'ls.scipy_iterative'):
    file.write(str(ls.name)+' '+str(ls.conf.method)+' '+str(ls.conf.precond)+' '+str(ls.conf.eps_r)+' '+str(ls.conf.i_max)+'\n' )
        #    else:
#file.write(str(ls.name)+' '+str(ls.conf.method)+'\n')



   
   
   # conf = Struct(method='bcgsl', precond='jacobi', sub_precond=None,
   #                 i_max=10000, eps_a=1e-50, eps_r=1e-8, eps_d=1e4,
#              verbose=True)
            
                 
                 
#ls = PETScKrylovSolver(conf)



#ls = ScipyIterative({'method':'bicgstab','i_max':100,'eps_r':1e-10})


    nls_status = IndexedStruct()
    nls = Newton({'i_max':1,'eps_a':1e-10}, lin_solver=ls, status=nls_status)

    pb = Problem('elasticity', equations=eqs, nls=nls, ls=ls)
    


    dd=pb.get_materials()['Mat']
    dd.set_function(get_mat1)
    
    
    pb.save_regions_as_groups('regions')

    #pb.time_update(ebcs=Conditions([fix_u, shift_u]))

    pb.time_update(ebcs=Conditions(bc))
    pb.save_regions_as_groups('regions')

#ls = ScipyIterative({'method':'bicgstab','i_max':100,'eps_r':1e-10})


#   A = pb.mtx_a
#   M = spilu(A,fill_factor = 1)
    
    #conf = Struct(solvers ='ScipyIterative',method='bcgsl', sub_precond=None,
# i_max=1000, eps_r=1e-8)
        
#pb.set_conf_solvers(conf)
    vec = pb.solve()
    print nls_status
    file.write('TIME TO SOLVE\n')
    file.write(str(nls.status.time_stats['solve'])+'\n')
    file.write('TIME TO CREATE MATRIX\n')
    file.write(str(nls.status.time_stats['matrix'])+'\n')
    #out = post_process(out, pb, state, extend=False)
    ev = pb.evaluate
    out = vec.create_output_dict()
    strain = ev('ev_cauchy_strain.3.Omega(u)', mode='el_avg')
    stress = ev('ev_cauchy_stress.3.Omega(Mat.D, u)', mode='el_avg',
                copy_materials=False)

    out['cauchy_strain'] = Struct(name='output_data', mode='cell',
                                  data=strain, dofs=None)
    out['cauchy_stress'] = Struct(name='output_data', mode='cell',
                                  data=stress, dofs=None)


    # Postprocess the solution.
    #out = vec.create_output_dict()
    #out = stress_strain(out, pb, vec,lam,mu, extend=True)
    #pb.save_state('its2D_interactive.vtk', out=out)
    #print 'aqui estoy'
    pb.save_state('strain.vtk', out=out)
    #pb.save_state('disp.vtk', out=vec)
    #print 'ahora estoy aqui'
    #out = stress_strain(out, pb, vec, extend=True)
    #pb.save_state('out.vtk', out=out)
    print nls_status
    
    order = 3
    strain_qp = ev('ev_cauchy_strain.%d.Omega(u)' % order, mode='qp')
    stress_qp = ev('ev_cauchy_stress.%d.Omega(Mat.D, u)' % order,
                       mode='qp', copy_materials=False)

    file.close()
    options_probe=False
    if options_probe:
        # Probe the solution.
        probes, labels = gen_lines(pb)
        nls_options = {'eps_a':1e-8,'i_max':1}
        ls = ScipyDirect({})
        ls2 = ScipyIterative({'method':'bicgstab','i_max':5000,'eps_r':1e-20})
        order = 5
        sfield = Field.from_args('sym_tensor', nm.float64, (3,), omega,
                                approx_order=order-1)
        stress = FieldVariable('stress', 'parameter', sfield,
                               primary_var_name='(set-to-None)')
        strain = FieldVariable('strain', 'parameter', sfield,
                               primary_var_name='(set-to-None)')

        cfield = Field.from_args('component', nm.float64, 1, omega,
                                 approx_order=order-1)
        component = FieldVariable('component', 'parameter', cfield,
                                  primary_var_name='(set-to-None)')

        ev = pb.evaluate
        order = 2*(order - 1) #2 * (2- 1)
        print "before strain_qp"
        strain_qp = ev('ev_cauchy_strain.%d.Omega(u)' % order, mode='qp')
        stress_qp = ev('ev_cauchy_stress.%d.Omega(Mat.D, u)' % order,
                       mode='qp', copy_materials=False)
        print "before projections"
        print stress
        project_by_component(strain, strain_qp, component, order,ls2,nls_options)
        #print 'strain done'
        project_by_component(stress, stress_qp, component, order,ls2,nls_options)

        print "after projections"
        
        all_results = []
        for ii, probe in enumerate(probes):
            fig, results = probe_results2(u, strain, stress, probe, labels[ii])

            fig.savefig('test_probe_%d.png' % ii)
            all_results.append(results)

        for ii, results in enumerate(all_results):
            output('probe %d:' % ii)
            output.level += 2
            for key, res in ordered_iteritems(results):
                output(key + ':')
                val = res[1]
                output('  min: %+.2e, mean: %+.2e, max: %+.2e'
                       % (val.min(), val.mean(), val.max()))
            output.level -= 2
    
    #pb.save_state('linear_elasticity_3D.vtk', vec)
if __name__ == '__main__':
    main()
