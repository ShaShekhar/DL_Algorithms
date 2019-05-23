def batchnorm_forward(x,gamma,beta,bn_param):
    """
    Forward pass for batch normalization (Use on FC layers).
    Input:
    -x: Data of shape (N,D)
    -gamma,beta: Scale/Shift parameter of shape (D,)
    -bn_param: Dictionary with the following keys: (mode,eps,momentum,r_mean/var)
    Return a tuple of : (out,cache)
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps',1e-5)
    momentum = bn_param.get('momentum',0.9)
    N,D = x.shape
    running_mean = bn_param.get('running_mean',np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var',np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        # Forward pass
        # Step 1: Calculate mean
        mu = 1.0/float(N) * np.sum(x,axis=0)
        # Step 2: Subtract the mean of every training sample
        x_m_mu = x - mu
        # Step 3: Calculate denominator
        xmmu_square = x_m_mu**2
        # Step 4: Calculate variance
        var = 1.0/float(N)*np.sum(xmmu_square,axis=0)
        # Step 5: Add eps for numerical stability then get square root
        sqrt_var = np.sqrt(var + eps)
        # Step 6: Invert square root
        inv_var = 1.0/sqrt_var
        # Step 7: Calculate normalization
        va2 = x_m_mu * inv_var
        # Step 8: Add scale and shift
        va3 = gamma*va2
        out = va3 + beta

        # Calculate the running mean and variances to be used on prediction
        running_mean = momentum * running_mean + (1.0 - momentum)*mu
        running_var = momentum * running_var + (1.0 - momentum)*var
        # Store values
        cache = (mu,x_m_mu,xmmu_square,var,sqrt_var,inv_var,va2,va3,gamma,beta,x,bn_param)

    elif mode == 'test':
        # On prediction get the running mean/variance
        running_mean = bn_param['running_mean']
        running_var = bn_param['running_var']
        xbar = (x - running_mean)/np.sqrt(running_var+eps)
        out = gamma*xbar + beta
        cache = (x,xbar,gamma,beta,eps)
    else:
        raise ValueError('Invalid forward batchnorm mode {}'.format(mode))
    # Save updated running mean/variance
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

def batchnorm_backward(dout,cache):
    """
    Backward pass for batch normalization (Use on FC layers).
    Use computaion graph to guide the backward propagation!
    Inputs:
    - dout: Upstream derivative of shape(N,D)
    - cache: Variable of intermediates from batchnorm_forward.
    Returns a tuple of : (dx(N,D), dgamma(D), dbeta(D))
    """
    dx, dgamma, dbeta = None, None, None
    mu, x_m_mu, xmmu_square, var, sqrt_var, inv_var, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps',1e-5)
    N, D = dout.shape
    #Backprop Step 9
    dva3 = dout
    dbeta = np.sum(dout,axis=0)
    #Backprop Step 8
    dva2 = gamma*dva3
    dgamma = np.sum(va2*dva3,axis=0)
    #Backprop Step 7
    dxmu = inv_var * dva2
    dinvvar = np.sum(x_m_mu*dva2,axis=0)
    # Backprop step 6
    dsqrtvar = -1/(sqrt_var**2) * dinvvar
    #Backprop Step 5
    dvar = 0.5 * (var + eps)**(-0.5)*dsqrtvar
    #Backprop Step 4
    dxmmu_square = 1.0/float(N) * np.ones((xmmu_square.shape))*dvar
    #Backprop Step 3
    dx_m_mu += 2 * x_m_mu * dxmmu_square
    #Backprop Step 2
    dx = dx_m_mu
    dmu = -np.sum(dxmu,axis=0)
    #Backprop Step 1
    dx += 1.0/float(N) * np.ones((dx_m_mu.shape)) * dmu

    return dx,dgamma,dbeta
