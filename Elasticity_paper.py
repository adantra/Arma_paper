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
from sfepy.terms.terms_dot import DotProductSurfaceTerm
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
#####################################################################
## Function for defining mechanical properties  #####################
#####################################################################
def get_mat1(ts,coors, mode=None,**kwargs):
    if mode == 'qp':
        nqp = coors.shape[0]
        out = nm.zeros((nqp, 6, 6), dtype=nm.float64)
        D_m = stiffness_from_youngpoisson(3, E_m, nu_m)
        D_f = stiffness_from_youngpoisson(3, E_f, nu_f)
        out[0:nqp, ...] = D_f
        top = centre[2] - dims[2]/2
        bottom = centre[2] + dims[2]/2
        thicknes = 0.2*(bottom - top)
        for i in nm.arange(nqp):
            if coors[i,2]>thicknes + top and coors[i,2]<bottom-thicknes:
                out[i,...] = D_m
        return {'D': out}

#####################################################################
##### Function for defining load and density gradient ###############
#####################################################################
def get_mat_f(ts,coors, mode=None,**kwargs):
    if mode == 'qp':
        nqp = coors.shape[0]
        out = nm.ones((nqp), dtype=nm.float64)*density
        eps = 1.0-8*(dims(2),3)
        top = centre[2] - dims[2]/2 + eps
        for i in nm.arange(nqp):
            if coors[i,2] < top:
                out[i,2] = Load
            else:
                out[i,2] = Density
        return {'val' : out}

##########################################################
######## GLOBAL VARIABLES DEFINING THE PROBLEM ###########
##########################################################

######## PROBLEM DIMENSIONS ###########
dims = nm.array((5000.0,5000.0,5000.0))
######## DISCRETIZATION ###############
shape = nm.array((20,20,100))
NX=shape[0]
NY=shape[1]
NZ=shape[2]
centre = nm.array((0.0,0.0,0.0))
##### DEPTH (ft)   #########################
Depth = 5000.0
##### Density (psi/ft) #####################
Density = 1.0
###### LOAD (psi) ##########################
Load = Depth * Density
### Young's Modulus and Poisson ratio for two materials
E_f, nu_f, E_m, nu_m  = 4000000, 0.2, 2000000, 0.3
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
    f = Material('f', val=[[0.0], [0.0],[-1.0]])
    load = Material('Load',val=[[0.0],[0.0],[-Load]])

    D = stiffness_from_lame(3,lam, mu)
    mat = Material('Mat', D=D)

    get_mat = Function('get_mat1',get_mat1)
    get_mat_f = Function('get_mat_f',get_mat1)

    integral = Integral('i', order=3)
    s_integral = Integral('is',order=2)

    t1 = Term.new('dw_lin_elastic(Mat.D, v, u)',
         integral, omega, Mat=mat, v=v, u=u)
    t2 = Term.new('dw_volume_lvf(f.val, v)', integral, omega, f=f, v=v)
    #t3 = Term.new('DotProductSurfaceTerm(Load.val, v)',s_integral,gamma5,Load=load,v=v)
    t3 = Term.new('dw_surface_ltr( Load.val, v )',s_integral,gamma6,Load=load,v=v)
    eq = Equation('balance', t1 + t2 + t3)
    eqs = Equations([eq])

    fix_u = EssentialBC('fix_u', gamma1, {'u.all' : 0.0})
    left_bc  = EssentialBC('Left',  gamma1, {'u.0' : 0.0})
    right_bc = EssentialBC('Right', gamma2, {'u.0' : 0.0})
    back_bc = EssentialBC('Front', gamma3, {'u.1' : 0.0})
    front_bc = EssentialBC('Back', gamma4, {'u.1' : 0.0})
    bottom_bc = EssentialBC('Bottom', gamma5, {'u.all' : 0.0})
    top_bc = EssentialBC('Top', gamma6, {'u.2' : 0.2})

    bc=[left_bc,right_bc,back_bc,front_bc,bottom_bc]
    #bc=[bottom_bc,top_bc]


    ##############################
    #  ##### SOLVER SECTION  #####
    ##############################

    conf = Struct(method='bcgsl', precond='jacobi', sub_precond=None,
                  i_max=10000, eps_a=1e-50, eps_r=1e-10, eps_d=1e4,
                  verbose=True)

    ls = PETScKrylovSolver(conf)

    file.write(str(ls.name)+' '+str(ls.conf.method)+' '+str(ls.conf.precond)+' '+str(ls.conf.eps_r)+' '+str(ls.conf.i_max)+'\n' )

    nls_status = IndexedStruct()
    nls = Newton({'i_max':1,'eps_a':1e-10}, lin_solver=ls, status=nls_status)

    pb = Problem('elasticity', equations=eqs, nls=nls, ls=ls)

    dd=pb.get_materials()['Mat']
    dd.set_function(get_mat1)
    #xload = pb.get_materials()['f']
    #xload.set_function(get_mat_f)

    pb.save_regions_as_groups('regions')

    pb.time_update(ebcs=Conditions(bc))

    vec = pb.solve()
    print nls_status


    file.write('TIME TO SOLVE\n')
    file.write(str(nls.status.time_stats['solve'])+'\n')
    file.write('TIME TO CREATE MATRIX\n')
    file.write(str(nls.status.time_stats['matrix'])+'\n')

    ev = pb.evaluate
    out = vec.create_output_dict()
    strain = ev('ev_cauchy_strain.3.Omega(u)', mode='el_avg')
    stress = ev('ev_cauchy_stress.3.Omega(Mat.D, u)', mode='el_avg',
                copy_materials=False)

    out['cauchy_strain'] = Struct(name='output_data', mode='cell',
                                  data=strain, dofs=None)
    out['cauchy_stress'] = Struct(name='output_data', mode='cell',
                                  data=stress, dofs=None)

    pb.save_state('strain.vtk', out=out)

    print nls_status


    file.close()

if __name__ == '__main__':
    main()
