class FixedParameterSettings(object):
    def __init__(self, fix_inducing_inputs=False, fix_variance=False,
                 fix_lengthscale=False, fix_gaussian_noise_variance=False):
        self.fix_gaussian_noise_variance = fix_gaussian_noise_variance
        self.fix_lengthscale = fix_lengthscale
        self.fix_variance = fix_variance
        self.fix_inducing_inputs = fix_inducing_inputs


class InitParameters(object):
    def __init__(self, Z=None, n_inducing=None, variance=None, lengthscale=None, gaussian_noise_variance=None):
        self.gaussian_noise_variance = gaussian_noise_variance
        self.lengthscale = lengthscale
        self.variance = variance
        self.n_inducing = n_inducing
        self.Z = Z