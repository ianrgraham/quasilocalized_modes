"""
Goal of this module is to quickly calculate the hessian for a system of N particles with 
"""

import numpy as np
import scipy.spatial as spa
import scipy.linalg as sla
import scipy.special as spe
import scipy.sparse as ssp
from scipy.sparse.linalg import eigsh
import time
import copy

from numba import njit, prange
import string

np.seterr(all='raise')

"""
Below are a number of fast functions implemented with numba

The speed up from these have been trememdous, especially the tensor based algebra
"""

@njit
def PBC(arr, L):
    """
    arr - input vector of paticle displacements
    L - the approximate size of the system

    Corrects particle displacements in 2D systems with periodic BCs

    In the case of a system with LE BCs, it is to be used only in the cases of
    comparing LE boxes at similar strains. In this case the arr vector should
    be computed using by applying the boxmat before taking the difference.

    Assumes nothing has shifted by more than L/2. beyond periodic boundaries
    """
    arr[arr < -L / 2.] += L
    arr[arr > L / 2.] -= L
    return arr

@njit
def get_hoomd_bidisperse_r(pid):
    if pid == 0:
        r = 5/12
    else:
        r = 7/12
    return r

@njit
def PBC_LE(diff, box):
    sub_dim = (diff > box[0]/2).astype(np.int64)
    add_dim = (diff < -box[0]/2).astype(np.int64)
    LE = (add_dim - sub_dim)[::-1]
    LE[1:] = 0
    diff += (add_dim - sub_dim + LE*box[3])*box[0]
    return diff

@njit
def hertzian_k(r, sigma):
    # return second derivative of hertzian potential
    return 1.5/(sigma*sigma)*np.sqrt(1-r/sigma)

@njit
def hertzian_e(r, sigma):
    # return 3rd derivative of hertzian potential
    return -.75/(sigma*sigma*sigma*np.sqrt(1-r/sigma))

@njit
def harmonic_k(r, sigma):
    # return second derivative of harmonic potential
    return 1

@njit
def harmonic_e(r, sigma):
    # return 3rd derivative of harmonic potential
    return 0

@njit
def _routine_calc_hessian(e, el, ed, s, dim, hessian, func):
    for l in np.arange(len(e)):
        # get K
        k = func(el[l], s[l])*ed[l]
        # add to all places
        k_outer = np.outer(k, k)
        for i in e[l]:
            for j in e[l]:
                if i == j:
                    hessian[i*dim:(i+1)*dim,j*dim:(j+1)*dim] -= k_outer
                else:
                    hessian[i*dim:(i+1)*dim,j*dim:(j+1)*dim] += k_outer
    return hessian

#@njit
#def _vec_outer(v, vec):
#    pass

@njit
def _tensor_dot(v, p):
    shape1 = v.shape
    out = np.zeros(shape1[0])
    for i in np.arange(shape1[1]):
        for j in np.arange(shape1[2]):
            out += v[:,i,j]*p[i,j]
    return out

@njit
def _filter_mode(vec, edges, u3s, v3s, dim, N, NE):
    filt_vec = np.zeros_like(vec)

    self_outers = np.zeros((N,dim,dim))
    for idx in np.arange(N):
        u1 = vec[idx*dim:(idx+1)*dim]
        self_outers[idx] = np.outer(u1,u1) 

    for idx in np.arange(NE):
        #for e, u, v in zip(edges, u3s, v3s):
        e = edges[idx]
        u = u3s[idx]
        v = v3s[idx]
        p1 = e[0]
        p2 = e[1]

        u1 = vec[p1*dim:(p1+1)*dim]
        u2 = vec[p2*dim:(p2+1)*dim]

        v1 = self_outers[p1]
        v2 = np.outer(u1,u2)
        v3 = self_outers[p2]

        t1 = _tensor_dot(v, v1)
        t2 = _tensor_dot(v, v2)
        t3 = _tensor_dot(v, v3)

        out = u*(t1 - 2*t2 + t3)

        filt_vec[p1*dim:(p1+1)*dim] += out
        filt_vec[p2*dim:(p2+1)*dim] -= out

    return filt_vec

"""
A few class definitions to help get eigenmodes/vales from a jammed packing
"""

class bid:
    # more like a struct
    def __init__(self, pset, sigma):
        self.pset = pset
        self.sigma = sigma

class k_table:
    def __init__(self, U_dxdx, sigma, points=1000):
        # sigma will be rmax
        self.vals = np.zeros((points), dtype=np.float64)
        self.points_np = np.linspace(0, sigma, points)
        self.points = [[p] for p in self.points_np]
        for i, r in enumerate(self.points_np):
            self.vals[i] = U_dxdx(r, sigma)
        self.tree = spa.cKDTree(self.points)
    
    def get_k(self, r):
        dist, ind = self.tree.query([r], k=2)
        d1, d2 = dist.T
        v1, v2 = self.vals[ind].T
        v = (d1)/(d1 + d2)*(v2 - v1) + v1
        return v


class k_multi_table:

    #tables = []

    def __init__(self, U_dxdx, bids, points=1000):
        self.bids = bids
        self.tables = []
        for bid in bids:
            self.tables.append(k_table(U_dxdx, bid.sigma, points))
        return


    def get_bond_k_bid(self, r, bid):
        idx = self.bids.index(bid)
        return self.tables[idx].get_k(r)

    def get_bond_k_sigma(self, r, sigma):
        for idx, bid in enumerate(self.bids):
            if bid.sigma == sigma:
                break
        return self.tables[idx].get_k(r)


class mode_calculator:

    def _find_bond_list(self):
        max_r = np.max(self.rad)
        assert(self.box[0] == self.box[1])
        l = 3*max_r
        n = int(self.box[0]//l)
        l = self.box[0]/n

        #cell_pre = np.linspace(0,self.box[0], n+1)[:-1]
        #self.cells = np.meshgrid(cell_pre, cell_pre)

        # this was the most annoying bug ever....
        tmp = [[] for i in np.arange(n)]
        self.cells = [copy.deepcopy(tmp) for i in np.arange(n)]

        #print(self.cells)

        # organize particles into boxes
        for idx, p in enumerate(self.pos):
            tmp_p = p + self.box[0]/2
            tmp_p[0] -= tmp_p[1]*self.box[3]
            #print(l, p, np.floor_divide(tmp_p, l), np.mod(np.floor_divide(tmp_p, l), n).astype(np.int64))
            #time.sleep(1)
            j = np.mod(np.floor_divide(tmp_p, l), n).astype(np.int64)
            #if (j == np.array([0,0])).all():
            #print(p, tmp_p, j, idx, self.box[3])
            #print(idx)
            self.cells[j[0]][j[1]].append(idx)
            #print(self.cells[0][0])

        #print(self.cells)

        neighbors = [[0,0],[1,0],[1,-1],[0,-1],[-1,-1]]

        edges = []
        edge_len = []
        edge_dir = []
        sigmas = []

        for i in np.arange(n):
            for j in np.arange(n):
                for idx, neigh in enumerate(neighbors):
                    i2 = (i + neigh[0])%n
                    j2 = (j + neigh[1])%n
                    #print('     ',(l*i,l*j),(l*i2,l*j2))
                    #print('     ',self.cells[i2][j2])
                    for i3 in self.cells[i][j]:
                        for j3 in self.cells[i2][j2]:
                            if idx == 0 and j3 <= i3:
                                continue
                            dir_tmp = PBC_LE(self.pos[i3]-self.pos[j3], self.box)
                            er_tmp = np.linalg.norm(dir_tmp)
                            sig_tmp = self.rad[i3] + self.rad[j3]
                            if er_tmp < sig_tmp:
                                #try:
                                edges.append([i3,j3])
                                edge_len.append(er_tmp)
                                edge_dir.append(dir_tmp/er_tmp)
                                sigmas.append(sig_tmp)
                                    #print(dir_tmp, er_tmp)
                                #except:
                                #    print(i3, j3, sig_tmp, dir_tmp, er_tmp)
                            #else:
                            #print(i3, j3, sig_tmp, dir_tmp, er_tmp)
                                

        self.edges = np.array(edges)
        self.edge_len = np.array(edge_len)
        self.edge_dir = np.array(edge_dir)
        self.sigmas = np.array(sigmas)
        return
    
    
    def _init_k_table(self):
        #bids = [bid(('A','A'), 5/6), bid(('A','B'), 1), bid(('B','B'), 7/6)]
        alph = list(string.ascii_uppercase)
        rs = np.unique(self.rad)
        combs = int(spe.comb(len(rs), 2, repetition=True))
        bids = []
        for i in np.arange(combs):
            if i < len(rs):
                bids.append(bid((alph[i],alph[i]), 2*rs[i]))
            else:
                j = i%len(rs)
                k = i//len(rs)
                bids.append(bid((alph[j],alph[k]),rs[j]+rs[k]))

        self.hertz_multi_table = k_multi_table(self.k_func, bids)
        return
    

    def _calc_hessian(self, use_KDTree=False, fast=True):
        Nd = self.pos.size
        hessian = np.zeros((Nd, Nd))#ssp.lil_matrix((Nd, Nd))
        #hessian = _routine_calc_hessian(self.edges, self.edge_len, self.edge_dir, self.sigmas)
        #if use_KDTree:
        #    k_func = njit(self.hertz_multi_table.get_bond_k_sigma)
        #else:
        k_func = self.k_func
        if fast:
            hessian = _routine_calc_hessian(self.edges, self.edge_len, self.edge_dir, self.sigmas, self.dim, hessian, k_func)
        else:
            for e, el, ed, s in zip(self.edges, self.edge_len, self.edge_dir, self.sigmas):
                # get K
                k = k_func(el, s)*ed
                # add to all places
                k_outer = np.outer(k, k)
                for i in e:
                    for j in e:
                        if i == j:
                            hessian[i*self.dim:(i+1)*self.dim,j*self.dim:(j+1)*self.dim] -= k_outer
                        else:
                            hessian[i*self.dim:(i+1)*self.dim,j*self.dim:(j+1)*self.dim] += k_outer

        self.hessian = ssp.csr_matrix(hessian)

    def _calc_modes(self, k=10):
        vals, vecs = eigsh(self.hessian, k=k, sigma=0)
        self.evals = vals
        self.evecs = vecs


    def __init__(self, pos, rad, box, k_func, e_func=None, use_KDTree=False, k=50, dim=2):
        assert(dim == 2)
        self.pos=pos.reshape((len(pos)//dim, dim))
        self.rad=rad
        self.box=box
        self.dim=dim
        self.k_func = k_func
        self.e_func = e_func
        self.U3_calculated = False
        if use_KDTree:
            self._init_k_table()
        print("Creating bond list.")
        self._find_bond_list()
        print("Constructing Hessian")
        self._calc_hessian(use_KDTree=use_KDTree)
        print("Calculating",k,"smallest eigenmodes and vectors")
        try:
            self._calc_modes(k=k)
        except RuntimeError:
            print("Eigen calculation failed, try again by runing {obj}._calc_modes(k=k')")
        #self._calc_contraction_matrix

    def _compute_U3(self):
        # U3 will have to be represented
        #Nd = self.pos.size
        u3s = []
        v3s = []
        for el, ed, s in zip(self.edge_len, self.edge_dir, self.sigmas):
            u3 = self.e_func(el, s)
            v3 = np.tensordot(ed, ed, 0)
            #print(v3)
            v3 = np.tensordot(ed, v3, 0)
            #print(v3)
            u3s.append(u3)
            v3s.append(v3)
        #hessian = ssp.lil_matrix((Nd, Nd))
        # will have to scrap sparse matrix representation here, since linerly independent elements just scale linearly with
        self.u3s = np.array(u3s)
        self.v3s = np.array(v3s)
        self.U3_calculated = True

    def _filter_mode(self, vec, fast=True):
        if fast:
            filt_vec = _filter_mode(vec, self.edges, self.u3s, self.v3s, self.dim, self.pos.size//2, len(self.edges))
            return filt_vec
        else:
            # this routine is quite slow in practice, only keeping it as a check that the fast routine is correct
            filt_vec = np.zeros_like(vec)
            # iterate over bonds
            indices = np.array([[0,0,0],[0,0,1],[0,1,0],
                                [1,0,0],[1,1,1],[1,1,0],
                                [1,0,1],[0,1,1]])
            signs = np.array([1,-1,-1,-1,-1,1,1,1])
            #print(self.v3s)
            for e, u, v in zip(self.edges, self.u3s, self.v3s):
                for idx, s in zip(indices, signs):
                    p1 = e[idx[0]]
                    p2 = e[idx[1]]
                    p3 = e[idx[2]]
                    tmp = s*u*np.tensordot(v, np.outer(vec[p2*self.dim:(p2+1)*self.dim],vec[p3*self.dim:(p3+1)*self.dim]))
                    filt_vec[p1*self.dim:(p1+1)*self.dim] += tmp
            return filt_vec
        
    def recalc_modes(self, k=10):
        self._calc_modes(k=k)

    def filter_modes(self, fast=True, parallel=False):
        # if contraction doesn't exist, compute it
        assert(self.e_func is not None)
        if not self.U3_calculated:
            print("Computing U3 terms")
            self._compute_U3()

        filtered_vecs = []
        print("Filtering eigenvectors")
        for vec in self.evecs.T:
            filt_vec = self._filter_mode(vec, fast=fast)
            filtered_vecs.append(filt_vec)
        
        return filtered_vecs


"""
Below I made some nice wrappers for returning a mode_calculator object for the different systems that I have
"""

def mode_calculator_gsd(gsd_file_handle, idx, r_func=get_hoomd_bidisperse_r, k_func=hertzian_k, e_func=hertzian_e, use_KDTree=False, k=30, dim=2):
    """
    for gsd files produced from hoomd
    """
    s = gsd_file_handle[idx]
    mc = mode_calculator(s.particles.position[:,:dim].flatten(), 
                        np.vectorize(r_func)(s.particles.typeid),
                        s.configuration.box, k_func, e_func=e_func, k=k, dim=dim, use_KDTree=use_KDTree)
    return mc

def mode_calculator_nc(s, idx, k_func=hertzian_k, e_func=hertzian_e, use_KDTree=False, k=30, dim=2):
    """
    for netcdf files produced from Carl Goodrich's jsrc code
    """
    print("This should not work at the moment")
    print("I need to apply the box matrix forward to all positions")
    return None
    box = s.variables.BoxMatrix[idx,:]
    new_box = np.array([box[0],box[-1],1,box[1]/box[0]])
    mc = mode_calculator(s.variables.pos[idx,:], 
                        s.variables.rad[idx,:],
                        new_box, k_func, e_func=e_func, k=k, dim=dim, use_KDTree=use_KDTree)
    return mc
