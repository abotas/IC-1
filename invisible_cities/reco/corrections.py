import numpy as np
import scipy as sc


from ..core               import fit_functions as fitf
from ..core.exceptions    import ParameterNotSet
from .. evm.ic_containers import Measurement


class Correction:
    """
    Interface for accessing any kind of corrections.

    Parameters
    ----------
    xs : np.ndarray
        Array of coordinates corresponding to each correction.
    fs : np.ndarray
        Array of corrections or the values used for computing them.
    us : np.ndarray
        Array of uncertainties or the values used for computing them.
    norm_strategy : False or string
        Flag to set the normalization option. Accepted values:
        - False:    Do not normalize.
        - "max":    Normalize to maximum energy encountered.
        - "index":  Normalize to the energy placed to index (i,j).
    interp_strategy : string
        Flag to set the interpolation option. Accepted values:
        - "nearest"  : Take correction from the closest node
        - "bivariate": Cubic spline interpolation in 2d.
    default_f, default_u : floats
        Default correction and uncertainty for missing values (where fs = 0).
    """

    def __init__(self,
                 xs, fs, us,
                   norm_strategy = False,
                   norm_opts     = None,
                 interp_strategy = "nearest",
                 default_f       = 0,
                 default_u       = 0):

        self._xs = [np.array( x, dtype=float) for x in xs]
        self._fs =  np.array(fs, dtype=float)
        self._us =  np.array(us, dtype=float)

        self._interp_strategy = interp_strategy

        self._default_f = default_f
        self._default_u = default_u

        self._normalize(norm_strategy, norm_opts)
        self._get_correction = self._define_interpolation(interp_strategy)

    def __call__(self, *x):
        """
        Compute the correction factor.

        Parameters
        ----------
        *x: Sequence of nd.arrays
             Each array is one coordinate. The number of coordinates must match
             that of the `xs` array in the init method.
        """
        return Measurement(*self._get_correction(*x))

    def _define_interpolation(self, opt):
        if   opt == "nearest"  : corr = self._nearest_neighbor
        elif opt == "bivariate": corr = self._bivariate()
        else: raise ValueError("Interpolation option not recognized: {}".format(opt))
        return corr

    def _normalize(self, strategy, opts):
        if not strategy            : return

        elif   strategy == "const" :
            if "value" not in opts:
                raise ParameterNotSet(("Normalization stratery 'const' requires"
                                       "the normalization option 'value'"))
            f_ref = opts["value"]
            u_ref = 0

        elif   strategy == "max"   :
            flat_index = np.argmax(self._fs)
            mult_index = np.unravel_index(flat_index, self._fs.shape)
            f_ref = self._fs[mult_index]
            u_ref = self._us[mult_index]

        elif   strategy == "center":
            index = tuple(i // 2 for i in self._fs.shape)
            f_ref = self._fs[index]
            u_ref = self._us[index]

        elif   strategy == "index" :
            if "index" not in opts:
                raise ParameterNotSet(("Normalization stratery 'index' requires"
                                       "the normalization option 'index'"))
            index = opts["index"]
            f_ref = self._fs[index]
            u_ref = self._us[index]

        else:
            raise ValueError("Normalization option not recognized: {}".format(strategy))

        assert f_ref > 0, "Invalid reference value."

        valid    = (self._fs > 0) & (self._us > 0)
        valid_fs = self._fs[valid].copy()
        valid_us = self._us[valid].copy()

        # Redefine and propagate uncertainties as:
        # u(F) = F sqrt(u(F)**2/F**2 + u(Fref)**2/Fref**2)
        self._fs[ valid]  = f_ref / valid_fs
        self._us[ valid]  = np.sqrt((valid_us / valid_fs)**2 +
                                    (   u_ref / f_ref   )**2 )
        self._us[ valid] *= self._fs[valid]

        # Set invalid to defaults
        self._fs[~valid]  = self._default_f
        self._us[~valid]  = self._default_u

    def _find_closest_indices(self, x, y):
        # Find the index of the closest value in y for each value in x.
        return np.argmin(abs(x-y[:, np.newaxis]), axis=0)

    def _nearest_neighbor(self, *x):
        # Find the index of the closest value for each axis
        x_closest = tuple(map(self._find_closest_indices, x, self._xs))
        return self._fs[x_closest], self._us[x_closest]

    def _bivariate(self):
        f_interp = sc.interpolate.RectBivariateSpline(*self._xs, self._fs)
        u_interp = sc.interpolate.RectBivariateSpline(*self._xs, self._us)
        return lambda x, y: (f_interp(x, y), u_interp(x, y))

    def __eq__(self, other):
        for i, x in enumerate(self._xs):
            if not np.allclose(x, other._xs[i]):
                return False

        if not np.allclose(self._fs, other._fs):
            return False

        if not np.allclose(self._us, other._us):
            return False

        return True


class Fcorrection:
    def __init__(self, f, u_f, pars):
        self._f   = lambda *x:   f(*x, *pars)
        self._u_f = lambda *x: u_f(*x, *pars)

    def __call__(self, *x):
        return Measurement(self._f(*x), self._u_f(*x))


def LifetimeCorrection(LT, u_LT):
    fun   = lambda z, LT, u_LT=0: fitf.expo(z, 1, LT)
    u_fun = lambda z, LT, u_LT  : z * u_LT / LT**2 * fun(z, LT)
    return Fcorrection(fun, u_fun, (LT, u_LT))


def LifetimeXYCorrection(pars, u_pars, xs, ys, **kwargs):
    LTs = Correction((xs, ys), pars, u_pars, **kwargs)
    return (lambda z, x, y: LifetimeCorrection(*LTs(x, y))(z))


def LifetimeRCorrection(pars, u_pars):
    def LTfun(z, r, a, b, c, u_a, u_b, u_c):
        LT = a - b * r * np.exp(r / c)
        return fitf.expo(z, 1, LT)

    def u_LTfun(z, r, a, b, c, u_a, u_b, u_c):
        LT   = a - b * r * np.exp(r / c)
        u_LT = (u_a**2 + u_b**2 * np.exp(2 * r / c) +
                u_c**2 *   b**2 * r**2 * np.exp(2 * r / c) / c**4)**0.5
        return z * u_LT / LT**2 * LTfun(z, r, a, b, c, u_a, u_b, u_c)

    return Fcorrection(LTfun, u_LTfun, np.concatenate([pars, u_pars]))
