# Based on TT-NF code from https://github.com/toshas/ttnf
# Modifications and/or extensions have been made for specific purposes in this project.

import torch
import numpy as np
import quimb.tensor as qtn
import torch.nn.functional as F
import time
import tntorch as tn

from tensorly.decomposition import matrix_product_state
from tensorly import tt_to_tensor
import tensorly as tl
tl.set_backend('pytorch') # or any other backend



def get_qtt_shape(dim_grid, dim=2):
    """ 
    dim_Grid: size side in xD grid
    dim: dimension of the grid (default 2d)
    """
    dim_grid_log2 = int(np.log2(dim_grid))
    dim_grid = int(dim_grid) # size of 3D grid
    num_factors = dim*dim_grid_log2
    shape_source = [dim_grid]*dim
    shape_target = [2**dim]*dim_grid_log2
    shape_factors = [2]*num_factors
    factor_ids = torch.arange(num_factors)

    factor_source_to_target = factor_ids.reshape(dim, dim_grid_log2).T.reshape(-1).tolist()
    factor_target_to_source = factor_ids.reshape(dim_grid_log2, dim).T.reshape(-1).tolist()

    return shape_source, shape_target, shape_factors, factor_source_to_target, factor_target_to_source

def get_tt_ranks(shape, max_rank=None, payload_position='first_core', payload_dim=0, scale_ranks_by_payload=True):
    if type(shape) not in (tuple, list) or len(shape) == 0:
        raise ValueError(f'Invalid shape: {shape}')
    ranks_left = torch.cumprod(torch.tensor(shape), dim=0).tolist()
    ranks_right = list(reversed(torch.cumprod(torch.tensor(list(reversed(shape))), dim=0).tolist()))
    if payload_position == 'first_core' and payload_dim != 0 and scale_ranks_by_payload:
        ranks_left = [payload_dim*r for r in ranks_left]
    if payload_position == 'last_core' and payload_dim != 0 and scale_ranks_by_payload:
        ranks_right = [payload_dim*r for r in ranks_right]
    ranks_tt = [min(a, b) for a, b in zip(ranks_left, ranks_right)]
    if max_rank is not None:
        ranks_tt = [min(r, max_rank) for r in ranks_tt]
    return ranks_tt


def get_qtt_TTNF(dim_grid, max_rank = 256, dim=2, payload_dim = 0, payload_position= "first_core", scale_ranks_by_payload = True, canonization = "None", compression_alg = "compress_all", sigma_init = None):
    shape_source, shape_target, shape_factors, factor_source_to_target, factor_target_to_source = get_qtt_shape(dim_grid, dim=dim)
    # remove the last element of the shape_target to get the correct number of cores
    ranks = get_tt_ranks(shape_target[:-1], max_rank=max_rank, payload_dim=payload_dim, payload_position=payload_position, scale_ranks_by_payload=scale_ranks_by_payload)
    if payload_position == "first_core":
        last = 2**dim
        if scale_ranks_by_payload:
            last *= payload_dim
        first_shape = [(payload_dim , 2**dim, last)]
    else:
        first_shape = [(2**dim, 2**dim)]
    if payload_position == "last_core":
        first = 2**dim
        if scale_ranks_by_payload:
            first *= payload_dim
        last_shape = [(first, 2**dim, payload_dim )]
    else:
        last_shape = [(2**dim, 2**dim)]
    tt_core_shapes = first_shape + [
            (ranks[i], 2**dim, ranks[i+1])
            for i in range(len(ranks)-1)
        ] + last_shape
    if sigma_init is None or sigma_init == 0:
        sigma_init = (-torch.tensor(ranks).double().log().sum() / (2. * len(ranks)+2)).exp().item()
    print("sigma_init", sigma_init)
    cores = [torch.randn(shape, dtype=torch.float) * sigma_init for shape in tt_core_shapes]
    if payload_position == "first_core":
        first_tensor = [qtn.Tensor(cores[0], inds=('payload', 'k0', 'virt0'), tags=['I0'])]
    else:
        first_tensor = [qtn.Tensor(cores[0], inds=('k0', 'virt0'), tags=['I0'])]
    if payload_position == "last_core":
        last_tensor = [qtn.Tensor(cores[-1], inds=('virt' + str(len(cores)-2), 'k' + str(len(cores)-1), 'payload'), tags=['I'+str(len(cores)-1)])]
    else:
        last_tensor = [qtn.Tensor(cores[-1], inds=('virt' + str(len(cores)-2), 'k' + str(len(cores)-1)), tags=['I'+str(len(cores)-1)])]
   
    middle_tensors = [qtn.Tensor(cores[i], inds=('virt'+str(i-1), 'k'+str(i), 'virt'+str(i)), tags=['I'+str(i)]) for i in  range(1, len(cores)-1)]
    tt = qtn.TensorNetwork(first_tensor + middle_tensors + last_tensor)

    tt = apply_compression(tt, compression_alg, max_rank)
    tt = apply_canonization(tt, canonization, len(cores), key = "I")

    return tt, shape_source, shape_target, shape_factors, factor_source_to_target, factor_target_to_source


def get_mpo_cores(dim):
    """
    Returns the cores of the MPO for the prolongation operator - Lubasch et al. 2018
    """
    interm_core = torch.zeros(size=(2,2,2,2)).float()
    interm_core[0,0,0,0] = 1
    interm_core[0,0,1,1] = 1
    interm_core[0,1,1,0] = 1
    interm_core[1,1,0,1] = 1
    # the last 
    last_core = torch.zeros(size=(2,2)).float()
    last_core[0,0] = 1
    last_core[0,1] = 0.5
    last_core[1,1] = 0.5

    first_core = interm_core[0]

    # interm_core kron product for each dimension
    interm_core_tmp = interm_core.clone()
    last_core_tmp = last_core.clone()
    first_core_tmp = first_core.clone()
    for i in range(dim - 1):
        interm_core = torch.kron(interm_core, interm_core_tmp)
        last_core = torch.kron(last_core, last_core_tmp)
        first_core = torch.kron(first_core_tmp, first_core)
    
    return interm_core, first_core, last_core

def contract_qtt(qtt, output_inds=None, expression=None, inds='k', masked_components=None):
    if output_inds is None:
        output_inds = [inds+str(i) for i in range(len(qtt.tensors))]
    if expression is not None:
        tensors = [t.data for t in qtt.tensors]
        if masked_components is not None:
            masked_products = [tensors[i] * masked_components[i] for i in range(len(tensors))]
            tensors = masked_products
            
        data = expression(*tensors)
    else:
        data = qtt.contract(output_inds=output_inds).data
    return data

def compute_output_inds(qtt, inds='k', payload_position="first_core", payload=0):
    output_inds = [inds+str(i) for i in range(len(qtt.tensors))]
    if payload > 0:
        if payload_position == "first_core":
            output_inds = ['payload'] + output_inds
        else: # last core
            output_inds = output_inds + ['payload']
    return output_inds

def qtt_to_tensor(qtt, shape_source, shape_factors, factor_target_to_source, payload, inds='k', payload_position="first_core", grayscale=False, expression=None, cores=None, masked_components=None):
    output_inds = compute_output_inds(qtt, inds=inds, payload_position=payload_position, payload=payload)
    data = contract_qtt(qtt, output_inds=output_inds, expression=expression, masked_components=masked_components)

    if grayscale and payload == 0:
        return tensor_order_from_qtt(shape_source, shape_factors, factor_target_to_source, data)
    elif payload_position == "first_core":
        data = data.reshape([payload] + shape_factors)
        factor_target_to_source = [i+1 for i in factor_target_to_source]
        data = data.permute(factor_target_to_source + [0])
        data = data.reshape(shape_source + [payload])
    else:
        raise ValueError("Only payload at first core and grayscale are supported")

    return data
        

def tensor_order_from_qtt(shape_source, shape_factors, factor_target_to_source, data):
    data = data.reshape(shape_factors)
    data = data.permute(factor_target_to_source)
    data = data.reshape(shape_source)
    return data

def tensor_order_to_qtt(shape_source, shape_factors, factor_source_to_target, data):
    data = data.reshape(shape_factors)
    data = data.permute(factor_source_to_target)
    data = data.reshape(shape_source)
    return data


def coord_tensor_to_coord_qtt3d(coords_xyz, dim_grid_log2, chunk=False, checks=False, reverse = True):
    """
    Converts a tensor of 3D coordinates to a QTT representation.
        :param coords_xyz: tensor of 3D coordinates - Nx3 tensor of integers
        :param dim_grid_log2: log2 of the grid size
        :param chunk: if True, the output is a list of QTTs, each of which is a chunk of the input
        :param checks: if True, performs some checks on the input
    :return:
        QTT representation of the coordinates
    """
    if checks:
        if not torch.is_tensor(coords_xyz) or coords_xyz.dim() != 2 or coords_xyz.shape[1] != 3:
            raise ValueError('Coordinates is not an Nx3 tensor')
        if not coords_xyz.dtype in (torch.int, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError('Coordinates are not integer')
        if torch.any(coords_xyz < 0) or torch.any(coords_xyz > 2 ** dim_grid_log2 - 1):
            raise ValueError('Coordinates out of bounds')
    bit_factors = 2 ** torch.arange(
        start=dim_grid_log2-1,
        end=-1,
        step=-1,
        device=coords_xyz.device,
        dtype=coords_xyz.dtype
    ) 
    # make random coords_xyz - Nx3 grid
    ran = torch.randint(0, 2 ** dim_grid_log2, (3,) )  

    bits_xyz = coords_xyz.unsqueeze(-1).bitwise_and(bit_factors).ne(0).byte()  # N x 3 x dim_grid_log2 
    # bitwisd returns a tensor of the same dtype as the first operand and the second operand is cast to that dtype

    bits_xyz = bits_xyz * torch.tensor([[[4], [2], [1]]], device=coords_xyz.device, dtype=torch.uint8)  # qtt octets
    core_indices = bits_xyz.sum(dim=1, dtype=torch.uint8)  # N x dim_grid_log2
    if chunk:
        core_indices = core_indices.chunk(dim_grid_log2, dim=1)  # [core_0_ind, ..., core_last_ind]
        core_indices = [c.view(-1) for c in core_indices]
    return core_indices



def coords_to_v2_tensor_ring(coords_xy):
    # Split the 2D coordinates into separate lists for x and y coordinates
    coords = coords_xy.chunk(2, dim=1)
    return [c.view(-1) for c in coords]


def coord_tensor_to_coord_qtt2d(coords_xy, dim_grid_log2, chunk=False, checks = False):
    if checks:
        if not torch.is_tensor(coords_xy) or coords_xy.dim() != 2 or coords_xy.shape[1] != 2:
            raise ValueError('Coordinates is not an Nx3 tensor')
        if not coords_xy.dtype in (torch.int, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError('Coordinates are not integer')
        if torch.any(coords_xy < 0) or torch.any(coords_xy > 2 ** dim_grid_log2 - 1):
            raise ValueError('Coordinates out of bounds')

    bit_factors = 2 ** torch.arange(
        start=dim_grid_log2-1,
        end=-1,
        step=-1,
        device=coords_xy.device,
        dtype=coords_xy.dtype
    )
    bits_xy = coords_xy.unsqueeze(-1).bitwise_and(bit_factors).ne(0).byte()  # N x 2 x dim_grid_log2
    bits_xy = bits_xy * torch.tensor([[[2], [1]]], device=coords_xy.device, dtype=torch.uint8)  # here we multiply by 2 and 1 to get the qtt octets
    core_indices = bits_xy.sum(dim=1, dtype=torch.uint8)  # N x dim_grid_log2
    if chunk:
        core_indices = core_indices.chunk(dim_grid_log2, dim=1)  # [core_0_ind, ..., core_last_ind]
        core_indices = [c.view(-1) for c in core_indices]
    return core_indices

def coord_qtt2d_to_coord_tensor(coords_qtt):
    dim_grid_log2 = coords_qtt.shape[1]
    bit_factors = 2 ** torch.arange(
        start=max(dim_grid_log2, 2) - 1,
        end=-1,
        step=-1,
        device=coords_qtt.device,
        dtype=coords_qtt.dtype
    )
    bits_qtt = coords_qtt.unsqueeze(-1).bitwise_and(bit_factors[-2:]).ne(0).byte()  # N x dim_grid_log2 x 2
    coords_xy_bits = bits_qtt.permute([0, 2, 1]) * bit_factors[-dim_grid_log2:]  # N x 2 x dim_grid_log2
    return coords_xy_bits.sum(dim=-1)

def get_pixel_value_for_coord_2d(mps, coords_xy, dim_grid_log2):
    coords = coord_tensor_to_coord_qtt2d(coords_xy, dim_grid_log2).tolist()
    value = mps.amplitude(coords[0])
    return value

def compress_all_tree(self, inplace=False, **general_compress_ocores):
        """Canonically compress this tensor network, assuming it to be a tree.
        This generates a tree spanning out from the most central tensor, then
        compresses all bonds inwards in a general_decoreh-first manner, using an infinite
        ``canonize_distance`` to shift the orthogonality center.
        """
        tn = self if inplace else self.copy()

        # order out spanning tree by general_decoreh first search
        def sorter(t, tn, distances, connectivity):
            return distances[t]

        tid0 = tn.most_central_tid()
        span = tn.get_tree_span([tid0], sorter=sorter)
        for tid1, tid2, _ in span:
            # absorb='right' shifts orthog center inwards
            tn._compress_between_tids(
                tid1, tid2, absorb='right',
                canonize_distance=float('inf'), **general_compress_ocores)

        return tn

def generate_indices(lower_ind, len_mps, physical_index = "k"):
    """
    Generates start charachters for lower and upper indices of the prolongation MPO. Also generates the last core index.

    This function dynamically creates indices for the current, upper, and last core tensor operations
    based on the provided lower index. It is designed to handle indices in a tensor network where
    the indices are a combination of a character followed by numeric digits.
    Parameters:
    - lower_ind (str): The current lower index of the tensor network. Expected to be a character followed by numeric digits.
    - len_mps (int): The length of the matrix product state (MPS), used to generate the last core index.
    - physical_index (str, optional): The prefix character for the physical index in the tensor network.
                                      Defaults to "k".
    Returns:
    - (tuple): A tuple containing three elements:
        - The current lower index (str).
        - The next upper index (str), generated by incrementing the digit part of the lower index.
        - The last core index (str), which combines the upper index with the length of the MPS.

    Example:
    >>> generate_indices("k0", 3)
    ("k", "k1", "k13") # k is the prefix for physical indices and k1 is the prefix physical indices of the resulting TT after contraction. k13 is the last core index
    >>> generate_indices("b2", 4, physical_index="b")
    ("b2", "b3", "b34")
    """
    if len(lower_ind) == 2:
        return lower_ind[0], f"{physical_index}1", f"{physical_index}1{len_mps}"
    else:
        char = lower_ind[0]
        lower_ind = lower_ind[:-1]
        second_digit = int(lower_ind[1]) + 1
        upper_index = char + str(second_digit)
        last_core_index = char + str(second_digit) + str(len_mps)
        return lower_ind, upper_index, last_core_index

            
@torch.no_grad()
def prolongate_qtt(tt, dim = 3, ranks_tt = 128, payload_position = "first_core", compression_alg = "compress_all", canonization = "None"):
    """
    Prolongs a tensor network using the Prolongation Matrix Product Operator (MPO) method as described in Lubasch et al. 2018.

    This function extends the provided TT/MPS by applying a prolongation operation. It constructs a Matrix Product Operator based on the dimensions and specified cores, and then applies it to the tensor network. This operation is useful in quantum tensor network algorithms for increasing the resolution or size of a tensor network representation.

    Parameters:
    - tt (TensorNetwork): The tensor network to be prolonged.
    - dim (int, optional): The dimension of the MPO cores. Defaults to 3.
    - ranks_tt (int, optional): The maximum bond dimension for the tensor network compression. Defaults to 128.
    - payload_position (str, optional): The position of the payload in the tensor network. Defaults to "first_core".
    - compression_alg (str, optional): The algorithm to use for tensor network compression. Options are "compress_all" or other custom algorithms. Defaults to "compress_all".
    - canonization (str, optional): The canonization method to be applied to the tensor network. Defaults to "None".

    Returns:
    - Tuple: A tuple containing the prolonged tensor network and additional shape information:
        - prolonged (TensorNetwork): The prolonged tensor network.
        - shape_source (list): The shape of the source tensor network.
        - shape_factors (list): The shape factors used in the prolongation.
        - factor_target_to_source (list): The factor mapping from the target to the source network.

    Note:
    - The function uses 'torch.no_grad()' context to disable gradient calculations, optimizing memory usage and computation time during the prolongation process.
    - It assumes that the first tensor in the network contains the necessary indices for index generation.
    """
    
    interm_core, first_core, last_core = get_mpo_cores(dim)
    len_mps = len(tt.tensors)
    tensors = tt.tensors
    device = tensors[0].data.device # get device from tt

    # Construct  list of tensors for MPO
    # First core, interm_core, ..., interm_core, MOCK last core as random tensor - will be replaced by last_core
    # Last core is never used but necessary to construct an MPO with proper indices and length
    mpo_tensors = [first_core.to(device)] + [interm_core.to(device)] * (len_mps - 1)  + [torch.rand_like(first_core).to(device)]

    lower_ind,  upper_ind, last_core_ind = generate_indices(tensors[1].inds[1], len_mps) # use first core index 

    mpo = qtn.MatrixProductOperator(
        mpo_tensors,
        lower_ind_id=lower_ind+'{}',
        upper_ind_id=upper_ind+'{}'
    )
    
    prolonged_tensors = [mpo.tensors[i] @ tensors[i] for i in range(len_mps)]
    
    last_core_tensor = qtn.Tensor(
        data= last_core.to(device), 
        inds=[prolonged_tensors[-1].inds[1]] + [last_core_ind], 
        tags=['I'+str(len_mps)]
    )
    
    prolonged = qtn.TensorNetwork(prolonged_tensors + [last_core_tensor]) # constructs a TensorNetwork from a list of quimb tensors
    prolonged.fuse_multibonds_() # fuses bonds with same name
    
    
    # compress and cannonize
    prolonged = apply_canonization(prolonged, canonization, len_mps)
    prolonged = apply_compression(prolonged, compression_alg, ranks_tt)

    # transpose necessary since we want  virtual_index, index, virtual_index instead of virtual_index, virtual_index, index
    transpose_tensors(prolonged, payload_position=payload_position)

    check_correctly_fused_cores(prolonged)

    shape_source, _, shape_factors, _, factor_target_to_source = get_qtt_shape(int( 2**(len_mps+1)), dim=dim)
    return prolonged, shape_source, shape_factors, factor_target_to_source

def apply_canonization(prolonged, canonization, len_mps, key = "I"):
    CANONIZATION_OPTIONS = {
        "None": None,
        "left":  f'{key}0',
        "right": f'{key}{len_mps}',
        "middle": f'{key}{len_mps//2}'}
    # Canonization
    canonization_value = CANONIZATION_OPTIONS.get(canonization)
    if canonization_value is not None:
        prolonged = qtn.TensorNetwork.canonize_around(prolonged, canonization_value)
    return prolonged    

def apply_compression(prolonged, compression_alg, ranks_tt):
    COMPRESSION_OPTIONS = {
        "compress_all": lambda: qtn.TensorNetwork.compress_all(prolonged, max_bond=ranks_tt, inplace=True),
        "tree_compress": lambda: compress_all_tree(prolonged, inplace=True, max_bond=ranks_tt) }
    # Compression
    compression_function = COMPRESSION_OPTIONS.get(compression_alg)
    if compression_function is not None:
        compression_function()

    return prolonged

def transpose_tensors(prolonged, payload_position="first_core"):
    for i in range(len(prolonged.tensors) - 1):
        inds = prolonged.tensors[i].inds
        if i == 0:
            if payload_position == "first_core":
                prolonged.tensors[i].transpose_(inds[2], inds[1], inds[0])
            else:
                print("prolonged.tensors[i]", prolonged.tensors[i])
                prolonged.tensors[i].transpose_(inds[1], inds[0])
        else:
            prolonged.tensors[i].transpose_(inds[0], inds[2], inds[1])


def check_correctly_fused_cores(prolonged):
    for i in range(len(prolonged.tensors)):
        ten = prolonged.tensors[i]
        len_ten = len(ten.shape)
        if len_ten != 2 and len_ten != 3:
            print("Look at the TN",prolonged)
            raise ValueError("One core has wrong number of indices")


def get_core_tensors(qtt):
    return [t.data for t in qtt.tensors]
    #return [qtt.tensors[i].data for i in range(len(qtt.tensors))]

def batched_indexed_gemv(bv, mm, m_indices, return_unordered=False, checks=False, reverse=True):
    if checks:
        if not (torch.is_tensor(bv) and torch.is_tensor(mm) and torch.is_tensor(m_indices)):
            raise ValueError('Operand is not a tensor')
        #if not (bv.dtype == mm.dtype and m_indices.dtype in (torch.int, torch.int8, torch.int16, torch.int32, torch.int64)):
        #    raise ValueError(f'Incompatible dtypes: {bv.dtype=} {mm.dtype=} {m_indices.dtype=}')
    
    m_indices_uniq_vals, m_indices_uniq_cnts = m_indices.unique(sorted=True, return_counts=True)
    if checks:
        if m_indices_uniq_vals.max() >= mm.shape[0]:
            raise ValueError('Incompatible index and matrices')
    m_indices_order_fwd = m_indices.argsort()
    m_indices_order_bwd = m_indices_order_fwd.argsort()
    bv = bv[m_indices_order_fwd]
    bv_split = bv.split(m_indices_uniq_cnts.cpu().tolist())
    bv_out = []

    for i, v_in in zip(m_indices_uniq_vals.cpu().tolist(), bv_split):
        if reverse:
            mm_vals = mm[i]
        else:
            mm_vals = mm[i].T
        v_out = F.linear(v_in, mm_vals, None)
        bv_out.append(v_out)
    bv_out = torch.cat(bv_out, dim=0)
    if return_unordered:
        return bv_out, m_indices_order_fwd, m_indices_order_bwd
    bv_out = bv_out[m_indices_order_bwd]
    return bv_out


def sample_intcoord_tt_v2(input, coords, last_core_is_payload=False, checks=False, reverse=True):
    """
    Performs sampling of QTT using integer coordinates (version 2). This version avoids model replication by
    applying applying an algorithm that (1) permutes all samples according to mode slice that will be used to propagate
    using the QTT contraction formula, then (2) applies torch.linear mode times to each of the groups, and (3) keeps
    track of the permutation to either recover the initial order or return the inverse permutation.
    :param input (List[torch.Tensor]): QTT cores
    :param coords (List[torch.Tensor]): Coordinates indexing QTT cores
    :param last_core_is_payload (bool): When True, last core is a payload and thus needs no indexing
    :param checks: Extra checks
    :return: torch.Tensor of a shape (N, dim_payload) containing payloads at the sampled points.
    """
    if checks:
        if last_core_is_payload and len(input) < 2:
            raise ValueError('Operand does not carry a payload (need at least two cores)')
        if len(coords) + int(last_core_is_payload) != len(input):
            raise ValueError('Coordinates do not cover all non-payload modes', len(coords) + int(last_core_is_payload), len(input))

    # cores are permuted after first prolongation from virtual_index index, virtual_index virtual_index virtual_index index - FIX in prolongation

    core_indices_0 = coords[-1].int()
    input_0 = input[-1].T

    bv = input_0.index_select(0, core_indices_0)

    permutation_fwd, permutation_bwd = None, None

    for ci in range(len(input)-2, -1, -1):
        mm = input[ci].permute(1, 0, 2)  # m x r_l x r_r
        core_ind = coords[ci]
        if permutation_fwd is not None:
            core_ind = core_ind[permutation_fwd]
        bv, p_fwd, p_bwd = batched_indexed_gemv(bv, mm, core_ind, return_unordered=True, checks=checks, reverse=reverse)
        permutation_fwd = p_fwd if permutation_fwd is None else permutation_fwd[p_fwd]
        permutation_bwd = p_bwd if permutation_bwd is None else p_bwd[permutation_bwd]

    if permutation_bwd is not None:
        bv = bv[permutation_bwd]

    # print("bv", bv.shape.len)
    return bv




def sample_intcoord_tt_v3(input, coords, last_core_is_payload=False, checks=False):
    """
    Performs sampling of QTT using integer coordinates (version 3). This version borrows the same algorithm
    from version 2, but also checks whether the trailing QTT cores are matricized identities (represented with buffers).
    For a group of such cores, the function saves computation by replacing matrix multiplication with indexing.
    :param input (List[torch.Tensor]): QTT cores
    :param coords (List[torch.Tensor]): Coordinates indexing QTT cores
    :param tt_core_isparam (List[bool]): Indicates which cores are parameters and which are identity matricizations.
    :param checks: Extra checks
    :return: torch.Tensor of a shape (N, dim_payload) containing payloads at the sampled points.
    """
    tt_core_isparam = [s[0] * s[1] != s[2] and s[0] != s[1] * s[2] for s in input.shape]
    if tt_core_isparam is None or all(tt_core_isparam):
        return sample_intcoord_tt_v2(input, coords, last_core_is_payload=last_core_is_payload, checks=checks)
    if checks:
        if last_core_is_payload and len(input) < 2:
            raise ValueError('Operand does not carry a payload (need at least two cores)')
        if len(input) != len(tt_core_isparam):
            raise ValueError('Incompatible operands')
        if len(coords) + int(last_core_is_payload) != len(input):
            raise ValueError('Coordinates do not cover all non-payload modes')

    bv = None
    permutation_fwd, permutation_bwd = None, None
    indices_left, indices_right = None, None

    for ci in range(0, len(input) - int(last_core_is_payload)):
        core_ind = coords[ci]
        if not tt_core_isparam[ci] and bv is None:
            # left
            if indices_left is None:
                indices_left = core_ind.long().clone()
            else:
                indices_left *= input[ci].shape[1]
                indices_left += core_ind.long()
        elif tt_core_isparam[ci]:
            # middle
            if bv is None:
                bv = input[ci][indices_left, core_ind.long(), :]
            else:
                if ci == len(input)-1:
                    mm = input[ci].squeeze(-1).T # m x r_l
                else:
                    mm = input[ci].permute(1, 0, 2)  # m x r_l x r_r
                if permutation_fwd is not None:
                    core_ind = core_ind[permutation_fwd]
                bv, p_fwd, p_bwd = batched_indexed_gemv(bv, mm, core_ind, return_unordered=True, checks=checks)
                permutation_fwd = p_fwd if permutation_fwd is None else permutation_fwd[p_fwd]
                permutation_bwd = p_bwd if permutation_bwd is None else p_bwd[permutation_bwd]
        else:
            # right
            cur_ind = core_ind.long().unsqueeze(-1) * input[ci].shape[2]
            if indices_right is None:
                indices_right = cur_ind
            else:
                indices_right += cur_ind

    if permutation_bwd is not None:
        bv = bv[permutation_bwd]

    if indices_right is not None:
        if last_core_is_payload:
            indices_right = indices_right + torch.arange(
                input[-1].shape[1], dtype=indices_right.dtype, device=indices_right.device
            ).view(1, -1)
        bv = bv.take_along_dim(indices_right, 1)

    return bv



def sample_intcoord_tt_v2(input, coords, last_core_is_payload=False, checks=False):
    """
    Performs sampling of TT using integer coordinates (version 2). This version avoids model replication by
    applying applying an algorithm that (1) permutes all samples according to mode slice that will be used to propagate
    using the TT contraction formula, then (2) applies torch.linear mode times to each of the groups, and (3) keeps
    track of the permutation to either recover the initial order or return the inverse permutation.
    :param input (List[torch.Tensor]): TT cores
    :param coords (List[torch.Tensor]): Coordinates indexing TT cores
    :param last_core_is_payload (bool): When True, last core is a payload and thus needs no indexing
    :param checks: Extra checks
    :return: torch.Tensor of a shape (N, dim_payload) containing payloads at the sampled points.
    """
    if checks:
        if last_core_is_payload and len(input) < 2:
            raise ValueError('Operand does not carry a payload (need at least two cores)')
        if len(coords) + int(last_core_is_payload) != len(input):
            raise ValueError('Coordinates do not cover all non-payload modes', len(coords) + int(last_core_is_payload), len(input))

    core_indices_0 = coords[-1].int()
    input_0 = input[-1].T
    

    bv = input_0.index_select(0, core_indices_0)
    permutation_fwd, permutation_bwd = None, None

    for ci in range(len(input)-2, -1, -1): # loop from len(input)-2 to 0 since payload is at first core 
        mm = input[ci].permute(1, 0, 2)  # m x r_l x r_r
        core_ind = coords[ci]
        if permutation_fwd is not None:
            core_ind = core_ind[permutation_fwd]
        bv, p_fwd, p_bwd = batched_indexed_gemv(bv, mm, core_ind, return_unordered=True, checks=checks)
        permutation_fwd = p_fwd if permutation_fwd is None else permutation_fwd[p_fwd]
        permutation_bwd = p_bwd if permutation_bwd is None else p_bwd[permutation_bwd]
        

    if permutation_bwd is not None:
        bv = bv[permutation_bwd]

    return bv

def sample_intcoord_tt_ring_v2(input, coords, last_core_is_payload=False, checks=False):
    """
    Performs sampling of TT using integer coordinates (version 2). This version avoids model replication by
    applying applying an algorithm that (1) permutes all samples according to mode slice that will be used to propagate
    using the TT contraction formula, then (2) applies torch.linear mode times to each of the groups, and (3) keeps
    track of the permutation to either recover the initial order or return the inverse permutation.
    :param input (List[torch.Tensor]): TT cores
    :param coords (List[torch.Tensor]): Coordinates indexing TT cores
    :param last_core_is_payload (bool): When True, last core is a payload and thus needs no indexing
    :param checks: Extra checks
    :return: torch.Tensor of a shape (N, dim_payload) containing payloads at the sampled points.
    """
    if checks:
        if last_core_is_payload and len(input) < 2:
            raise ValueError('Operand does not carry a payload (need at least two cores)')
        if len(coords) + int(last_core_is_payload) != len(input):
            raise ValueError('Coordinates do not cover all non-payload modes', len(coords) + int(last_core_is_payload), len(input))

    core_indices_0 = coords[0].int()
    input_0 = input[0]

    bv = input_0.index_select(1, core_indices_0)
    permutation_fwd, permutation_bwd = None, None
    
    bv = bv.permute(1,0,2)

    #for ci in range(len(input)-2, -1, -1): # loop from len(input)-2 to 0 since payload is at first core 
    # loo from start instead
    for ci in range(1, len(input)-1): # loop from len(input)-2 to 0 since payload is at first core
        mm = input[ci].permute(1, 0, 2)  # m x r_l x r_r
        core_ind = coords[ci]
        if permutation_fwd is not None:
            core_ind = core_ind[permutation_fwd]
        print("bv", bv.shape)
        print("mm", mm.shape)
        bv, p_fwd, p_bwd = batched_indexed_gemv(bv, mm, core_ind, return_unordered=True, checks=checks)
        permutation_fwd = p_fwd if permutation_fwd is None else permutation_fwd[p_fwd]
        permutation_bwd = p_bwd if permutation_bwd is None else p_bwd[permutation_bwd]
    
    if last_core_is_payload:
        # multiply with last core in input[-1]
        bv = torch.einsum('ijk,bik->bj', input[-1], bv)

    if permutation_bwd is not None:
        bv = bv[permutation_bwd]

    return bv


def sample_intcoord_qtt2d_v2(input, coords_xy, checks=False):
    """
    Performs sampling of QTT voxel grid using integer coordinates (version 2). See `sample_intcoord_tt_v2`.
    :param input (List[torch.Tensor]): QTT cores
    :param coords_xy (torch.Tensor): Batch of coordinates (N, 2) in the X,Y format
    :param checks: Extra checks
    :return: torch.Tensor of a shape (N, dim_payload) containing payloads at the sampled points.
    """
    coords = coord_tensor_to_coord_qtt2d(coords_xy, len(input), chunk=True, checks=checks)
    bv = sample_intcoord_tt_v2(input, coords, last_core_is_payload=False, checks=checks)
    return bv




def sample_intcoord_qtt2d_v2(input, coords_xy, checks=False):
    """
    Performs sampling of QTT voxel grid using integer coordinates (version 2). See `sample_intcoord_tt_v2`.
    :param input (List[torch.Tensor]): TT cores
    :param coords_xy (torch.Tensor): Batch of coordinates (N, 2) in the X,Y format
    :param checks: Extra checks
    :return: torch.Tensor of a shape (N, dim_payload) containing payloads at the sampled points.
    """
    coords = coord_tensor_to_coord_qtt2d(coords_xy, len(input), chunk=True, checks=checks)
    bv = sample_intcoord_tt_v2(input, coords, last_core_is_payload=False, checks=checks)
    return bv



def sample_generic_3d(
        input, # tensors as cores 
        coords_xyz,
        fn_sample_intcoord,
        outliers_handling="raise",
        sample_redundancy_handling = True,
        checks=False,
):
    
    if type(input) in (list, tuple):
        #dim_grid = 2 ** (len(input) - 1)
        dim_grid = 2 ** (len(input) )
        dim_payload = input[0].shape[0] # TODO check is correct
    elif torch.is_tensor(input):
        dim_grid = input.shape[0] 
        dim_payload = input.shape[-1]
    else:
        raise ValueError('Invalid input in sample_generic_3d')

    if checks:
        if not torch.is_tensor(coords_xyz) or coords_xyz.dim() != 2 or coords_xyz.shape[1] != 3:
            raise ValueError('Coordinates is not an Nx3 tensor')
        if not coords_xyz.dtype in (torch.float, torch.float16, torch.float32, torch.float64):
            raise ValueError('Coordinates are not floats')

    batch_size = coords_xyz.shape[0]
    mask_valid, mask_need_remap = None, None

    if outliers_handling:
        if outliers_handling == 'raise':
            mask_bad_left = coords_xyz < 0
            if torch.any(mask_bad_left):
                mask_bad_left = mask_bad_left.any(dim=-1)
                bad_coords = coords_xyz[mask_bad_left]
                raise ValueError(f'Coordinates out of bounds < 0: {bad_coords}')
            mask_bad_right = coords_xyz > dim_grid - 1
            if torch.any(mask_bad_right):
                mask_bad_right = mask_bad_right.any(dim=-1)
                bad_coords = coords_xyz[mask_bad_right]
                raise ValueError(f'Coordinates out of bounds > {dim_grid-1}: {bad_coords}')
        elif outliers_handling == 'clamp':
            coords_xyz.clamp_(min=0, max=dim_grid-1)
        elif outliers_handling == 'zeros':
            # mask_valid = torch.ones(mask_valid.shape, device=mask_valid.device, dtype=bool)
            mask_valid = torch.all(coords_xyz >= 0, dim=1) & torch.all(coords_xyz <= dim_grid - 1, dim=1)
            coords_xyz = coords_xyz[mask_valid]
            # print("Coords:",coords_xyz[1,2])
            if coords_xyz.shape[0] == 0:
                return torch.zeros(batch_size, dim_payload, dtype=coords_xyz.dtype, device=coords_xyz.device)
            mask_need_remap = coords_xyz.shape[0] < batch_size
        else:
            raise ValueError(f'Unknown outliers handling: {outliers_handling}')

    offs = torch.tensor([
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ], device=coords_xyz.device)

    coo_left_bottom_near = torch.floor(coords_xyz)
    wb = coords_xyz - coo_left_bottom_near
    wa = 1.0 - wb

    coo_left_bottom_near = coo_left_bottom_near.int()

    coo_cube = [coo_left_bottom_near] + [coo_left_bottom_near + offs[i] for i in range(7)]
    coo_cube = torch.cat(coo_cube, dim=0)
    coo_cube.clamp_max_(dim_grid - 1)

    # sample without redundancy
    if sample_redundancy_handling:
        coo_unique, coo_map_back = coo_cube.unique(dim=0, sorted=False, return_inverse=True)
        val_unique = fn_sample_intcoord(input, coo_unique, checks=checks)
        val = val_unique[coo_map_back]
    else:
        val = fn_sample_intcoord(input, coo_cube, checks=checks)

    # trilinear interpolation
    num_samples = coords_xyz.shape[0]
    val = val.view(4, 2, num_samples, dim_payload)
    val = val[:, 0, :, :] * wa[None, :, 2, None] + val[:, 1, :, :] * wb[None, :, 2, None]  # 4 x B x P
    val = val.view(2, 2, num_samples, dim_payload)
    val = val[:, 0, :, :] * wa[None, :, 1, None] + val[:, 1, :, :] * wb[None, :, 1, None]  # 2 x B x P
    val = val[0] * wa[:, 0, None] + val[1] * wb[:, 0, None]  # B x P

    ret = torch.zeros(batch_size, dim_payload, dtype=coords_xyz.dtype, device=coords_xyz.device)
    ret[mask_valid] = val
    return ret


def sample_intcoord_tensor3d(input, coords_xyz, checks=False):
    coords = coords_xyz.chunk(3, dim=1)
    coords = [c.view(-1) for c in coords]
    return sample_intcoord_tensor(input, coords, checks=checks)

# make same function but for payload first dimension
def sample_intcoord_tensor(input, coords, checks=False):
    """ Sample a tensor at integer coordinates
    input - tensor to sample from has shape (grid_size, grid_size, grid_size, payload_size)
    coords - list of 1D tensors with coordinates, has shape (num_points, )
    checks - if True, perform some checks
    returns - tensor of shape (num_points, payload_size)
    """
    if checks:
        if not torch.is_tensor(input):
            raise ValueError('Input is not a tensor')
        if type(coords) not in (tuple, list):
            raise ValueError('Coords is not a list')
        if len(coords) + 1 != input.dim():
            raise ValueError('Coordinates do not cover all non-payload modes')
        if not all(
                torch.is_tensor(coo) and coo.dim() == 1 and
                coo.dtype in (torch.int, torch.int8, torch.int16, torch.int32, torch.int64) and
                coo.numel() == coords[0].numel() and coo.dtype == coords[0].dtype and
                torch.all(coo >= 0) and torch.all(coo < input.shape[i])
                for i, coo in enumerate(coords)
        ):
            raise ValueError('Bad coordinates')
    input_flat = input.reshape(-1, input.shape[-1]) # input.shape[-1] if payload is last core
    strides = torch.tensor(input.shape[1:len(coords)]).flip(0).cumprod(0).flip(0).tolist() + [1]
    coords_flat = torch.cat([coo.view(-1, 1) for coo in coords], dim=1)
    coords_flat = (coords_flat.long() * torch.tensor(strides).to(coords_flat.device).view(1, -1)).sum(dim=1)
    out = input_flat.index_select(0, coords_flat)
    return out
        






def qtt_svd(image, rank, dim=2, payload_dim=0, payload_position="grayscale", library="tensorly"):
    target = image.clone()
    target_org = target.clone()
    dim_grid = target_org.shape[0]

    shape_source, shape_target, shape_factors, factor_source_to_target, factor_target_to_source = get_qtt_shape(dim_grid, dim=dim)

    ranks_tt = get_tt_ranks(shape_target[:-1], max_rank=rank, payload_dim=payload_dim, payload_position=payload_position)
    target = tensor_order_to_qtt(shape_target, shape_factors, factor_source_to_target, target)
    if library == "tensorly":
        ranks_tt = [1] + ranks_tt + [1]
        factors = matrix_product_state(target, rank=ranks_tt, verbose=False)
        num_elements =  [np.prod(f.shape) for f in factors]

        target_recov = tt_to_tensor(factors)
    elif library == "tntorch":
        factors = tn.Tensor(target, ranks_tt=ranks_tt)
        cores = factors.cores
        num_elements = [np.prod(c.shape) for c in cores]

        target_recov = torch.from_numpy(factors.numpy()).float()  # Convert to torch tensor and set type as float
    else:
        raise ValueError("Invalid library specified")

    num_params = sum(num_elements)
    num_params_org = target_org.numel()
    size_tn_mb = num_params * 4 / 1024 / 1024
    size_org_mb = num_params_org * 4 / 1024 / 1024

    print("Number of parameters", num_params)
    print("Size in MB", size_tn_mb)
    print("Number of params for target:", num_params_org)
    print("Size in MB for target:", size_org_mb)
    print('Compression ratio: {}/{} = {:g}'.format(num_params, num_params_org, num_params_org/num_params))
    
    target_recov = tensor_order_from_qtt(shape_source, shape_factors, factor_target_to_source, target_recov)

    return target_recov, target_org, factors


def get_qtt(im_tensor, ranks_tt, dim=2,compress = True, print_info =False):
    shape_im_tensor = im_tensor.shape[0]
    shape_source, shape_target, shape_factors, factor_source_to_target, factor_target_to_source = get_qtt_shape(int(shape_im_tensor), dim=dim)
    if print_info:
      # permute input tensor
      print("im_tensor.shape", im_tensor.shape)
      print("shape_source", shape_source)
      print("shape_target", shape_target)
      print("shape_factors", shape_factors)
      # multiply shape factors to get total number of factors
      print("multiplied shape factors", np.prod(shape_factors))
      print("factor_source_to_target", factor_source_to_target)
      print("factor_target_to_source", factor_target_to_source)

    im_tensor = tensor_order_to_qtt(shape_source, shape_factors, factor_source_to_target, im_tensor)
    # make qtt of input tensor
    qtt = qtn.MatrixProductState.from_dense(im_tensor, dims=shape_target)
    if compress:
        qtt.compress(method='svd', max_bond=ranks_tt, form = len(shape_target)//2)
    # convert to tensor network 
    qtt = qtn.TensorNetwork(qtt.tensors)

    return qtt, shape_source, shape_target, shape_factors, factor_source_to_target, factor_target_to_source


def create_identity_MPO(qtt, rank=2):
    """
    Create an identity MPO with modified ranks.

    Args:
    qtt (TensorNetwork): The tensor network to base the MPO on.
    rank (int, optional): Rank for modification. Defaults to 2.

    Returns:
    TensorNetwork: A new modified MPO.
    """
    L = len(qtt.tensors)
    mpo = qtn.MPO_identity(L, phys_dim=4)  # Assuming qtn is quimb.tensor

    new_mpo_tensors = []
    for t in mpo.tensors:
        data = t.data.astype(np.float32)
        data = np.repeat(data, rank, axis=0)
        if len(data.shape) == 4:
            data = np.repeat(data, rank, axis=1)
                
        new_mpo_tensors.append(torch.from_numpy(data))

    new_mpo = qtn.MatrixProductOperator(new_mpo_tensors, lower_ind_id=mpo.lower_ind_id, upper_ind_id=mpo.upper_ind_id)
    new_mpo.reindex_({f'k{i}': idx for i, idx in enumerate(get_physical_indices(qtt))})
    return new_mpo


def get_physical_indices(qtt, payload_position="first_core"):
    """
    Extract physical indices from each tensor in a tensor network based on payload position.

    Args:
    qtt (TensorNetwork): The tensor network from which to extract indices.
    payload_position (str, optional): Position of the payload, affects how indices are selected. Defaults to "first_core".

    Returns:
    list: A list of physical indices from each tensor.
    """
    indices = []
    for i, t in enumerate(qtt.tensors):
        if i == 0:
            if payload_position == "first_core":
                indices.append(t.inds[1])
            elif len(t.inds) == 2 and payload_position == "grayscale":
                indices.append(t.inds[0])
        elif len(t.inds) == 2 and i == len(qtt.tensors) - 1:
            indices.append(t.inds[-1])
        elif len(t.inds) == 3:
            indices.append(t.inds[1])            
    return indices



@torch.no_grad()
def increase_ranks_with_MPO(qtt, max_rank, payload_position="first_core"):
    """
    Increase the ranks of a tensor network up to a maximum rank.

    Args:
    qtt (TensorNetwork): The tensor network to modify.
    max_rank (int): The maximum rank to achieve.
    payload_position (str, optional): Position of the payload. Defaults to "first_core".

    Returns:
    TensorNetwork: The modified tensor network.
    """
    print(f"Max rank BEFORE: {qtt.max_bond()}")
    
    phys_indices_before = get_physical_indices(qtt)
    current_max_rank = qtt.max_bond()
    rank_multiplier = int(np.ceil(max_rank / current_max_rank))
    mpo = create_identity_MPO(qtt, rank=rank_multiplier)
    
    print("mpo ", mpo)

    new_tensors = [mpo_tensor @ qtt_tensor for mpo_tensor, qtt_tensor in zip(mpo.tensors, qtt.tensors)]
    qtt = qtn.TensorNetwork(new_tensors)
    qtt.reindex_({f'b{i}': phys_indices_before[i] for i in range(len(phys_indices_before))})
    qtt.fuse_multibonds_()
    
    print("qtt before compression", qtt)

    transpose_tensors(qtt, payload_position=payload_position)
    check_correctly_fused_cores(qtt)
    qtt.compress_all(max_bond=max_rank, inplace=True)
    print("qtt after compression", qtt)
    print(f"Max rank After: {qtt.max_bond()}")

    return qtt


@torch.no_grad()
def increase_ranks(qtt, max_rank, payload_position="first_core"):
    """
    Increase the ranks of a tensor network up to a maximum rank.

    Args:
    qtt (TensorNetwork): The tensor network to modify.
    max_rank (int): The maximum rank to achieve.
    payload_position (str, optional): Position of the payload. Defaults to "first_core".

    Returns:
    TensorNetwork: The modified tensor network.
    """
    max_bond = qtt.max_bond()
    diff = abs(max_rank - max_bond)
    
    tensor_list = []
    for i in range(len(qtt.tensors)):
        tensor = qtt.tensors[i].data
        if len(tensor.shape) == 3:
            if i == 0:
                tensor = torch.cat((tensor, torch.zeros(tensor.shape[0], tensor.shape[1], diff)), dim = 2)
            else:
                tensor = torch.cat((tensor, torch.zeros(tensor.shape[0], tensor.shape[1], diff)), dim = 2)
                tensor = torch.cat((tensor, torch.zeros(diff, tensor.shape[1], tensor.shape[2])), dim = 0)
        elif len(tensor.shape) == 2:
            tensor = torch.cat((tensor, torch.zeros(diff, tensor.shape[1])), dim = 0)
            
        tensor_list.append(tensor)
    
    # make into tensors with indices as before
    tensors = [ qtn.Tensor(data=tensor_list[i], inds=qtt.tensors[i].inds, tags=qtt.tensors[i].tags) for i in range(len(qtt.tensors))]    
    qtt = qtn.TensorNetwork(tensors)             
    
    # compress 
    if max_rank < max_bond:
        qtt.compress_all(max_bond=max_rank, inplace=True)
        

    return qtt