import torch
import torch.nn as nn
import torch.nn.functional as F

def extend_knots_uniform(grid, k_extend=0):
    """
    Uniformly extends each spline's knot vector by k_extend on both sides.
    grid: (n_splines, n_knots_original)
    returns (n_splines, n_knots_original + 2*k_extend)
    """
    if k_extend <= 0:
        return grid
    device = grid.device
    step = (grid[:, -1] - grid[:, 0]) / (grid.shape[1] - 1)  # shape (n_splines,)
    step = step.unsqueeze(1)  # (n_splines, 1)

    out = grid
    for _ in range(k_extend):
        left = out[:, [0]] - step
        out = torch.cat([left, out], dim=1)
        right = out[:, [-1]] + step
        out = torch.cat([out, right], dim=1)
    return out

def cox_deboor_inplace(x, t, k):
    """
    Evaluate B^k_i(x) for i=0..(n_knots - k - 2) using the classical in-place
    Cox–De Boor recursion.

    Args:
       x: (n_splines, n_samples)
       t: (n_splines, n_knots)
          Already extended if needed. (We do that outside this function)
       k: polynomial order (>= 0)

    Returns:
       B: (n_splines, n_knots - k - 1, n_samples)
         i.e. the final B^k. Each 'column' i is the i-th B-spline of order k.
    """
    device = x.device
    n_splines, n_knots = t.shape
    n_samples = x.shape[1]

    # number of final basis functions:
    n_bases = n_knots - k - 1
    if n_bases <= 0:
        # Not enough knots => return zeros
        return torch.zeros(n_splines, 1, n_samples, device=device)

    # 1) B => shape (n_splines, n_knots-1, n_samples)
    # We'll fill B with the zero-order basis (step function).
    B = torch.zeros((n_splines, n_knots - 1, n_samples), device=device)
    # For k=0, we have intervals [t_i, t_{i+1}).
    left_edges  = t[:, :-1].unsqueeze(-1)   # (n_splines, n_knots-1, 1)
    right_edges = t[:, 1:].unsqueeze(-1)    # (n_splines, n_knots-1, 1)
    x_expanded  = x.unsqueeze(1)           # (n_splines, 1, n_samples)
    mask = (x_expanded >= left_edges) & (x_expanded < right_edges)
    B[mask] = 1.0

    if k == 0:
        # Just return B^0 => but slice to n_bases if needed
        return B[:, :n_bases, :]

    # 2) Recursively do the in-place update for r=1..k
    #    for each r, we update B[:, i, :] using
    #    B^r_i(x) = alpha_i * B^{r-1}_i(x) + beta_i * B^{r-1}_{i+1}(x)
    # We'll do python-level loops over i, which is slightly less "vector-broadcasty," 
    # but ensures no shape mismatch.
    for r in range(1, k+1):
        # For each i in range(n_knots - 1 - r):
        #   alpha = (x - t_i)/(t_{i+r} - t_i)
        #   beta  = (t_{i+r+1} - x)/(t_{i+r+1} - t_{i+1})
        #   B[i] = alpha*B[i] + beta*B[i+1]
        i_stop = (n_knots - 1) - r  # last valid i index
        for i in range(i_stop):
            # shape => (n_splines, 1) for t_i
            t_i     = t[:, i  ].unsqueeze(-1)
            t_ir    = t[:, i+r].unsqueeze(-1)
            denom_a = (t_ir - t_i).clamp_min(1e-14)
            alpha   = (x - t_i) / denom_a

            t_i1    = t[:, i+1].unsqueeze(-1)
            t_irp1  = t[:, i+r+1].unsqueeze(-1)
            denom_b = (t_irp1 - t_i1).clamp_min(1e-14)
            beta    = (t_irp1 - x) / denom_b

            # B[:, i, :] => shape (n_splines, n_samples)
            B_i   = B[:, i, :]
            B_ip1 = B[:, i+1, :]

            # alpha, beta => shape (n_splines, n_samples)
            # we broadcast them:
            alpha_2d = alpha
            beta_2d  = beta

            # in-place update:
            B[:, i, :] = alpha_2d * B_i + beta_2d * B_ip1

    # at the end, B is B^k. The columns past (n_bases) are not valid.
    return B[:, :n_bases, :]

def spline_evaluate(x, grid, coef, k):
    """
    Evaluate the spline with coefficient 'coef' for each sample in x.

    x:    (n_splines, n_samples)
    grid: (n_splines, n_knots_original)
    coef: (n_splines, n_bases) 
          n_bases should be (n_knots_original - 1) + k
    k: int

    returns y: (n_splines, n_samples)
    """
    # 1) Extend the knot vector by k on both sides
    grid_ext = extend_knots_uniform(grid, k_extend=k)
    # new shape => (n_splines, n_knots_original + 2k)

    # 2) Evaluate B-splines of order k => shape (n_splines, n_bases, n_samples)
    B = cox_deboor_inplace(x, grid_ext, k)
    # B => (n_splines, n_bases, n_samples)

    # 3) multiply by coef => shape => (n_splines, n_samples)
    #   we can do an einsum or matmul across dimension #1
    #   B[s, :, n] * coef[s, :]
    # => y[s, n]
    y = torch.einsum('sbn,sb->sn', B, coef)
    return y


class SplineKAN(nn.Module):
    """
    Single-layer "Spline KAN" that maps (batch, in_dim) -> (batch, out_dim) using:
      - in_dim * out_dim separate B-splines (size = in_dim*out_dim),
      - each has (num+1) knots, order k,
      - plus a base activation scaled by scale_base,
      - plus the B-spline scaled by scale_sp.
    """
    def __init__(
        self,
        in_dim=3,
        out_dim=2,
        num=5,
        k=3,
        noise_scale=0.1,
        scale_base=1.0,
        scale_sp=1.0,
        base_fun=nn.SiLU(),
        grid_eps=0.02,
        grid_range=[-1, 1],
        sp_trainable=True,
        sb_trainable=True,
        device='cpu'
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.size = in_dim * out_dim
        self.num = num
        self.k = k
        self.device = device

        # Build uniform grid => (size, num+1)
        knots_1d = torch.linspace(grid_range[0], grid_range[1], steps=num+1, device=device)
        g2d = torch.einsum('i,j->ij', torch.ones(self.size, device=device), knots_1d)
        self.grid = nn.Parameter(g2d, requires_grad=False)

        # B-spline # of bases => (num + k), if uniform (num+1 -1) + k
        nbases = (num + k)
        init_coef = torch.randn((self.size, nbases), device=device) * noise_scale
        self.coef = nn.Parameter(init_coef)

        # scalars
        if isinstance(scale_base, float):
            sb = torch.full((self.size,), fill_value=scale_base, device=device)
        else:
            sb = torch.as_tensor(scale_base, dtype=torch.float32, device=device)
        self.scale_base = nn.Parameter(sb, requires_grad=sb_trainable)

        sp = torch.full((self.size,), scale_sp, device=device)
        self.scale_sp = nn.Parameter(sp, requires_grad=sp_trainable)

        self.base_fun = base_fun
        self.grid_eps = grid_eps

        # allow "mask" or "sharing" if you want
        self.mask = nn.Parameter(torch.ones(self.size, device=device), requires_grad=False)
        self.weight_sharing = torch.arange(self.size, device=device)  # identity
        self.lock_counter = 0
        self.lock_id = torch.zeros(self.size, dtype=torch.int, device=device)

    def forward(self, x):
        """
        x: (batch, in_dim)
        returns y: (batch, out_dim)
        """
        batch_size = x.shape[0]
        device = x.device

        # Expand => shape (size, batch)
        # The same approach as your original: each out_dim copies x => flatten => permute
        # shape => (batch, in_dim, out_dim) => (batch, in_dim*out_dim) => (batch, size) => permute => (size, batch)
        x_expanded = torch.einsum("ij,k->ikj", x, torch.ones(self.out_dim, device=device))
        x_expanded = x_expanded.reshape(batch_size, self.size).permute(1, 0)
        # => (size, batch)

        # base => (size, batch)
        base_act = self.base_fun(x_expanded)

        # Evaluate B-spline => (size, batch)
        # gather the relevant grid/coef by weight_sharing if you want (or just use them direct)
        g = self.grid[self.weight_sharing]   # (size, num+1)
        c = self.coef[self.weight_sharing]   # (size, nbases)
        spline_val = spline_evaluate(x_expanded, g, c, self.k)

        # scale & mask
        # => shape (size, batch)
        out_spline = (self.scale_base.unsqueeze(-1) * base_act +
                      self.scale_sp.unsqueeze(-1)   * spline_val)
        out_spline = out_spline * self.mask.unsqueeze(-1)

        # reshape => (out_dim, in_dim, batch) => (batch, out_dim, in_dim) => sum over in_dim
        out_3d = out_spline.reshape(self.out_dim, self.in_dim, batch_size)
        out_3d = out_3d.permute(2, 0, 1)  # => (batch, out_dim, in_dim)
        y = out_3d.sum(dim=2)            # => (batch, out_dim)
        return y

    # Additional methods if needed...

    def update_grid_from_samples(self, x):
        '''
        update grid from samples

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> model = SplineKAN(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-3.0002, -1.7882, -0.5763,  0.6357,  1.8476,  3.0002]])
        '''
        batch = x.shape[0]
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(
            1, 0)
        x_pos = torch.sort(x, dim=1)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k, device=self.device)
        num_interval = self.grid.shape[1] - 1
        ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
        grid_adaptive = x_pos[:, ids]
        margin = 0.01
        grid_uniform = torch.cat(
            [grid_adaptive[:, [0]] - margin + (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) * a for a in
             np.linspace(0, 1, num=self.grid.shape[1])], dim=1)
        self.grid.data = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k, device=self.device)

    def initialize_grid_from_parent(self, parent, x):
        '''
        update grid from a parent SplineKAN & samples

        Args:
        -----
            parent : SplineKAN
                a parent SplineKAN (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> batch = 100
        >>> parent_model = SplineKAN(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(parent_model.grid.data)
        >>> model = SplineKAN(in_dim=1, out_dim=1, num=10, k=3)
        >>> x = torch.normal(0,1,size=(batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-1.0000, -0.8000, -0.6000, -0.4000, -0.2000,  0.0000,  0.2000,  0.4000,
          0.6000,  0.8000,  1.0000]])
        '''
        batch = x.shape[0]
        # preacts: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        x_eval = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch,
                                                                                                  self.size).permute(1,
                                                                                                                     0)
        x_pos = parent.grid
        sp2 = SplineKAN(in_dim=1, out_dim=self.size, k=1, num=x_pos.shape[1] - 1, scale_base=0., device=self.device)
        sp2.coef.data = curve2coef(sp2.grid, x_pos, sp2.grid, k=1, device=self.device)
        y_eval = coef2curve(x_eval, parent.grid, parent.coef, parent.k, device=self.device)
        percentile = torch.linspace(-1, 1, self.num + 1).to(self.device)
        self.grid.data = sp2(percentile.unsqueeze(dim=1))[0].permute(1, 0)
        self.coef.data = curve2coef(x_eval, y_eval, self.grid, self.k, self.device)

    def get_subset(self, in_id, out_id):
        '''
        get a smaller SplineKAN from a larger SplineKAN (used for pruning)

        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons

        Returns:
        --------
            spb : SplineKAN

        Example
        -------
        >>> kanlayer_large = SplineKAN(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        '''
        spb = SplineKAN(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun, device=self.device)
        spb.grid.data = self.grid.reshape(self.out_dim, self.in_dim, spb.num + 1)[out_id][:, in_id].reshape(-1,
                                                                                                            spb.num + 1)
        spb.coef.data = self.coef.reshape(self.out_dim, self.in_dim, spb.coef.shape[1])[out_id][:, in_id].reshape(-1,
                                                                                                                  spb.coef.shape[
                                                                                                                      1])
        spb.scale_base.data = self.scale_base.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(-1, )
        spb.scale_sp.data = self.scale_sp.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(-1, )
        spb.mask.data = self.mask.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(-1, )

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        spb.size = spb.in_dim * spb.out_dim
        return spb

    def lock(self, ids):
        '''
        lock activation functions to share parameters based on ids

        Args:
        -----
            ids : list
                list of ids of activation functions

        Returns:
        --------
            None

        Example
        -------
        >>> model = SplineKAN(in_dim=3, out_dim=3, num=5, k=3)
        >>> print(model.weight_sharing.reshape(3,3))
        >>> model.lock([[0,0],[1,2],[2,1]]) # set (0,0),(1,2),(2,1) functions to be the same
        >>> print(model.weight_sharing.reshape(3,3))
        tensor([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
        tensor([[0, 1, 2],
                [3, 4, 0],
                [6, 0, 8]])
        '''
        self.lock_counter += 1
        # ids: [[i1,j1],[i2,j2],[i3,j3],...]
        for i in range(len(ids)):
            if i != 0:
                self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = ids[0][1] * self.in_dim + ids[0][0]
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = self.lock_counter

    def unlock(self, ids):
        '''
        unlock activation functions

        Args:
        -----
            ids : list
                list of ids of activation functions

        Returns:
        --------
            None

        Example
        -------
        >>> model = SplineKAN(in_dim=3, out_dim=3, num=5, k=3)
        >>> model.lock([[0,0],[1,2],[2,1]]) # set (0,0),(1,2),(2,1) functions to be the same
        >>> print(model.weight_sharing.reshape(3,3))
        >>> model.unlock([[0,0],[1,2],[2,1]]) # unlock the locked functions
        >>> print(model.weight_sharing.reshape(3,3))
        tensor([[0, 1, 2],
                [3, 4, 0],
                [6, 0, 8]])
        tensor([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
        '''
        # check ids are locked
        num = len(ids)
        locked = True
        for i in range(num):
            locked *= (self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] == self.weight_sharing[
                ids[0][1] * self.in_dim + ids[0][0]])
        if locked == False:
            print("they are not locked. unlock failed.")
            return 0
        for i in range(len(ids)):
            self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = ids[i][1] * self.in_dim + ids[i][0]
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = 0
        self.lock_counter -= 1


class WaveKANLayer(nn.Module):
    '''This is a sample code for the simulations of the paper:
    Bozorgasl, Zavareh and Chen, Hao, Wav-KAN: Wavelet Kolmogorov-Arnold Networks (May, 2024)

    https://arxiv.org/abs/2405.12832
    and also available at:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4835325
    We used efficient KAN notation and some part of the code:+

    '''

    def __init__(self, in_features, out_features, wavelet_type='mexican_hat', with_bn=True, device="cuda"):
        super(WaveKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        self.with_bn = with_bn

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        # self.weight1 is not used; you may use it for weighting base activation and adding it like Spl-KAN paper
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # Base activation function #not used for this experiment
        self.base_activation = nn.SiLU()

        # Batch normalization
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        # Implementation of different wavelet types
        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2) - 1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)

        elif self.wavelet_type == 'dog':
            # Implementing Derivative of Gaussian Wavelet
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'meyer':
            # Implement Meyer Wavelet here
            # Constants for the Meyer wavelet transition boundaries
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1 / 2, torch.ones_like(v),
                                   torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

            # Meyer wavelet calculation using the auxiliary function
            wavelet = torch.sin(pi * v) * meyer_aux(v)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'shannon':
            # Windowing the sinc function to limit its support
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)  # sinc(x) = sin(pi*x) / (pi*x)

            # Applying a Hamming window to limit the infinite support of the sinc function
            window = torch.hamming_window(x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype,
                                          device=x_scaled.device)
            # Shannon wavelet is the product of the sinc function and the window
            wavelet = sinc * window
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
            # You can try many more wavelet types ...
        else:
            raise ValueError("Unsupported wavelet type")

        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        # You may like test the cases like Spl-KAN
        # wav_output = F.linear(wavelet_output, self.weight)
        # base_output = F.linear(self.base_activation(x), self.weight1)

        base_output = F.linear(x, self.weight1)
        combined_output = wavelet_output  # + base_output

        # Apply batch normalization
        if self.with_bn:
            return self.bn(combined_output)
        else:
            return combined_output


class NaiveFourierKANLayer(nn.Module):
    """
    https://github.com/Jinfeng-Xu/FKAN-GCF/blob/main/models/common/kanlayer.py
    https://github.com/GistNoesis/FourierKAN/blob/main/fftKAN.py
    """

    def __init__(self, inputdim, outdim, gridsize=300):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                          (np.sqrt(inputdim) * np.sqrt(self.gridsize)))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        # Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        # This should be fused to avoid materializing memory
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], dim=0), self.fouriercoeffs)

        y = y.view(outshape)
        return y


# This is inspired by Kolmogorov-Arnold Networks but using Jacobian polynomials instead of splines coefficients
class JacobiKANLayer(nn.Module):
    """
    https://github.com/SpaceLearner/JacobiKAN/blob/main/JacobiKANLayer.py
    """

    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super(JacobiKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))

        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Since Jacobian polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Jacobian polynomial tensors
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:  ## degree = 0: jacobi[:, :, 0] = 1 (already initialized) ; degree = 1: jacobi[:, :, 1] = x ; d
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2
        for i in range(2, self.degree + 1):
            theta_k = (2 * i + self.a + self.b) * (2 * i + self.a + self.b - 1) / (2 * i * (i + self.a + self.b))
            theta_k1 = (2 * i + self.a + self.b - 1) * (self.a * self.a - self.b * self.b) / (
                    2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            theta_k2 = (i + self.a - 1) * (i + self.b - 1) * (2 * i + self.a + self.b) / (
                    i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            jacobi[:, :, i] = (theta_k * x + theta_k1) * jacobi[:, :, i - 1].clone() - theta_k2 * jacobi[:, :,
                                                                                                  i - 2].clone()  # 2 * x * jacobi[:, :, i - 1].clone() - jacobi[:, :, i - 2].clone()
        # Compute the Jacobian interpolation
        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y


class ChebyKANLayer(nn.Module):
    """
    https://github.com/SynodicMonth/ChebyKAN/blob/main/ChebyKANLayer.py
    """

    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y


class TaylorKANLayer(nn.Module):
    """
    https://github.com/Muyuzhierchengse/TaylorKAN/
    """

    def __init__(self, input_dim, out_dim, order, addbias=True):
        super(TaylorKANLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.order = order
        self.addbias = addbias

        self.coeffs = nn.Parameter(torch.randn(out_dim, input_dim, order) * 0.01)
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x):
        shape = x.shape
        outshape = shape[0:-1] + (self.out_dim,)
        x = torch.reshape(x, (-1, self.input_dim))
        x_expanded = x.unsqueeze(1).expand(-1, self.out_dim, -1)

        y = torch.zeros((x.shape[0], self.out_dim), device=x.device)

        for i in range(self.order):
            term = (x_expanded ** i) * self.coeffs[:, :, i]
            y += term.sum(dim=-1)

        if self.addbias:
            y += self.bias

        y = torch.reshape(y, outshape)
        return y


class RBFKANLayer(nn.Module):
    """
    https://github.com/Sid2690/RBF-KAN/blob/main/RBF_KAN.py
    """

    def __init__(self, input_dim, output_dim, num_centers, alpha=1.0):
        super(RBFKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.alpha = alpha

        self.centers = nn.Parameter(torch.empty(num_centers, input_dim))
        nn.init.xavier_uniform_(self.centers)

        self.weights = nn.Parameter(torch.empty(num_centers, output_dim))
        nn.init.xavier_uniform_(self.weights)

    def gaussian_rbf(self, distances):
        return torch.exp(-self.alpha * distances ** 2)

    def forward(self, x):
        distances = torch.cdist(x, self.centers)
        basis_values = self.gaussian_rbf(distances)
        output = torch.sum(basis_values.unsqueeze(2) * self.weights.unsqueeze(0), dim=1)
        return output


class KANInterface(nn.Module):
    def __init__(self, in_features, out_features, layer_type, n_grid=None, degree=None, order=None, n_center=None):
        super(KANInterface, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if layer_type == "WavKAN":
            self.transform = WaveKANLayer(in_features, out_features)
        elif layer_type == "KAN":
            self.transform = SplineKAN(in_features, out_features, num=n_grid)
        elif layer_type == "FourierKAN":
            self.transform = NaiveFourierKANLayer(in_features, out_features, gridsize=n_grid)
        elif layer_type == "JacobiKAN":
            self.transform = JacobiKANLayer(in_features, out_features, degree=degree)
        elif layer_type == "ChebyKAN":
            self.transform = ChebyKANLayer(in_features, out_features, degree=degree)
        elif layer_type == "TaylorKAN":
            self.transform = TaylorKANLayer(in_features, out_features, order=order)
        elif layer_type == "RBFKAN":
            self.transform = RBFKANLayer(in_features, out_features, num_centers=n_center)
        elif layer_type == "Linear":
            self.transform = nn.Linear(in_features, out_features, bias=True)
        else:
            raise NotImplementedError(f"Layer type {layer_type} not implemented")

    def forward(self, x):
        B, N, L = x.shape
        x = x.reshape(B * N, L)
        return self.transform(x).reshape(B, N, self.out_features)


class KANInterfaceV2(nn.Module):
    def __init__(self, in_features, out_features, layer_type, hyperparam):
        super(KANInterfaceV2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if layer_type == "WavKAN":
            self.transform = WaveKANLayer(in_features, out_features, with_bn=False)
        elif layer_type == "KAN":
            self.transform = SplineKAN(in_features, out_features, num=hyperparam)
        elif layer_type == "FourierKAN":
            self.transform = NaiveFourierKANLayer(in_features, out_features, gridsize=hyperparam)
        elif layer_type == "JacobiKAN":
            self.transform = JacobiKANLayer(in_features, out_features, degree=hyperparam)
        elif layer_type == "ChebyKAN":
            self.transform = ChebyKANLayer(in_features, out_features, degree=hyperparam)
        elif layer_type == "TaylorKAN":
            self.transform = TaylorKANLayer(in_features, out_features, order=hyperparam)
        # elif layer_type == "TaylorKAN2":
        #     self.transform = TaylorKANLayer2(in_features, out_features, order=hyperparam)
        else:
            raise NotImplementedError(f"Layer type {layer_type} not implemented")

    def forward(self, x):
        x = self.transform(x)
        return x


class MoKLayer(nn.Module):
    def __init__(self, in_features, out_features, experts_type="A", gate_type="Linear"):
        super(MoKLayer, self).__init__()
        if experts_type == "A":
            self.experts = nn.ModuleList([
                TaylorKANLayer(in_features, out_features, order=3, addbias=True),
                TaylorKANLayer(in_features, out_features, order=3, addbias=True),
                WaveKANLayer(in_features, out_features, wavelet_type="mexican_hat", device="cuda"),
                WaveKANLayer(in_features, out_features, wavelet_type="mexican_hat", device="cuda")
            ])
        elif experts_type == "B":
            self.experts = nn.ModuleList([
                TaylorKANLayer(in_features, out_features, order=3, addbias=True),
                TaylorKANLayer(in_features, out_features, order=3, addbias=True),
                JacobiKANLayer(in_features, out_features, degree=6),
                JacobiKANLayer(in_features, out_features, degree=6),
            ])
        elif experts_type == "C":
            self.experts = nn.ModuleList([
                TaylorKANLayer(in_features, out_features, order=3, addbias=True),
                TaylorKANLayer(in_features, out_features, order=4, addbias=True),
                JacobiKANLayer(in_features, out_features, degree=5),
                JacobiKANLayer(in_features, out_features, degree=6),
            ])
        elif experts_type == "L":
            self.experts = nn.ModuleList([
                nn.Linear(in_features, out_features),
                nn.Linear(in_features, out_features),
                nn.Linear(in_features, out_features),
                nn.Linear(in_features, out_features),
            ])
        elif experts_type == "V":
            self.experts = nn.ModuleList([
                TaylorKANLayer(in_features, out_features, order=3, addbias=True),
                JacobiKANLayer(in_features, out_features, degree=6),
                WaveKANLayer(in_features, out_features, wavelet_type="mexican_hat", device="cuda"),
                nn.Linear(in_features, out_features),
            ])
        else:
            raise NotImplemented

        self.n_expert = len(self.experts)
        self.softmax = nn.Softmax(dim=-1)

        if gate_type == "Linear":
            self.gate = nn.Linear(in_features, self.n_expert)
        elif gate_type == "KAN":
            self.gate = JacobiKANLayer(in_features, self.n_expert, degree=5)
        else:
            raise NotImplemented

    def forward(self, x):
        B, N, L = x.shape
        x = x.reshape(B * N, L)
        score = F.softmax(self.gate(x), dim=-1)  # (BxN, E)
        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.n_expert)], dim=-1)  # (BxN, Lo, E)
        return torch.einsum("BLE,BE->BL", expert_outputs, score).reshape(B, N, -1).contiguous()
