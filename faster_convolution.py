def conv_forward_naive(x,w,b,conv_param):
    P = conv_param['pad']
    stride = conv_param['stride']
    N,C,H,W = x.shape
    F,C,K_H,K_W = w.shape
    # calculate output size
    H_prime = (H + 2*P - K_H)//stride + 1
    W_prime = (W + 2*P - K_W)//stride + 1
    # Initialize output volume
    out = np.zeros((N,F,H_prime,W_prime))
    # im2col
    for im_num in range(N):
        im = x[im_num,:,:,:]
        im_pad = np.pad(im,((0,0),(P,P),(P,P)),'constant',constant_values=0)
        im_col = im2col(im_pad, K_H, K_W, stride, H_prime, W_prime) # im_col has shape (H_prime*W_prime, C*K_H*K_W)
        filter_col = np.reshape(W,(F,-1))                           # filter_col.T has shape (C*K_H*H_W,F)
        mul = np.dot(im_col,filter_col.T) + b                      # mul has shape (H_prime*W_prime,F)
        out[im_num,:,:,:] = col2im(mul,H_prime,W_prime,C)
    cache = (x,w,b,conv_param)
    return out,cache

def im2col(x,K_H,K_W,stride,H_prime,W_prime):
    C,H,W = x.shape #padded x
    col = np.zeros((H_prime*W_prime, C*K_H*K_W))
    for i in range(H_prime):
        for j in range(W_prime):
            patch = x[:, i*stride:i*stride+K_H, j*stride:j*stride+K_W]
            col = [i*H_prime + j,:] = np.reshape(patch,-1) # calling get item in rows all values goes to column
    return col

def col2im(mul,H_prime,W_prime,C):
    """
    Args:
    mul : (H_prime*W_prime,F) matrix each col should be reshaped to
    c*H_prime*W_prime when c > 0 or H_prime*W_prime when C = 0
    H_prime : reshaped filter height
    W_prime : reshaped filter width
    C : reshaped filter channel, if 0 , rehsape the filter to 2D,
    otherwise reshape it to 3D.
    Returns:
    if C == 1: (F,H_prime,W_prime) matrix
    otherwise : (F,C,H_prime,W_prime) matrix
    """
    F = mul.shape[1]
    if C == 1:
        out = np.zeros((F,H_prime,W_prime))
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(H_prime,W_prime))
    else:
        out = np.zeros((F,C,H_prime,W_prime))
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(C,H_prime,W_prime))
    return out

def conv_backward_naive(dout,cache):
    """
    Args:
    - dout : Upstream derivatives
    - cache : A tuple of (x,w,b,conv_param) as in conv_forward_naive
    Returns:
    dx,dw,db
    """
    x,w,b,conv_param = cache
    P = conv_param['pad']
    stride = conv_param['stride']
    N,C,H,W = x.shape
    F,C,K_H,K_W = w.shape
    H_prime = (H + 2*P - K_H)/stride + 1
    W_prime = (W + 2*P - K_W)/stride + 1
    dw = np.zeros(w.shape)
    dx = np.zeros(x.shape)
    db = np.zeros(b.shape)

    for i in range(N):
        im = x[i,:,:,:]
        im_pad = np.pad(im,((0,0),(P,P),(P,P)),'constant',constant_values=0)
        im_col = im2col(im_pad,K_H,K_W,stride,H_prime,W_prime) # shape (H'*W', C*K_H*K_W)
        filter_col = np.reshape(W,(F,-1)).T # shape = (C*K_W*K_H, F)
        dout_i = dout[i,:,:,:] # shape = (F,H',W')
        dbias_sum = np.reshape(dout_i,(F,-1)) # (F,H'*W')
        dbias_sum = dbias_sum.T # (H'*W',F)
        db += np.sum(dbias_sum,axis=0) #(F,)
        dmul = dbias_sum # (H'*W',F)
        # mul = im_col.T*dmul
        dfilter_col = np.dot(im_col.T,dmul) # (C*K_H*K_W, H'*W')*(H'*W', F) = (C*K_H*K_W, F)
        dw += np.reshape(dfilter_col.T,(F,C,K_H,K_W))

        dim_col = np.dot(dmul,filter_col.T) # shape (H'*W', C*K_H*K_W)
        dx_padded = col2im_back(dim_col,H_prime,W_prime,stride,K_H,K_W,C)
        dx[i,:,:,:] = dx_padded[:,P:H+P,P:W+P]
    return dx,dw,db

def col2_im_back(dim_col,H_prime,W_prime,stride,K_H,K_W,H,W):
    dx = np.zeros((C,H,W))
    for i in range(H_prime*W_prime):
        row = dim_col[i,:]
        h_s = (i/W_prime)*stride # h_start
        w_s = (i % W_prime)*stride
        dx[:, h_s:h_s+K_H, w_s:w_s+K_W] += np.reshape(row,(C,K_H,K_W))
    return dx
                
