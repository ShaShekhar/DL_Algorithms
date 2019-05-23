def conv_forward_naive(x,w,b,conv_param):
    """
    Computes the forward pass for the convolution layer
    Input:
    - x : Input data of shape (N,C,H,W)
    - w : Filter weight of shape (F,C,K_H,K_W)
    - b : Biases of shape (F,)
    - conv_param : A dictionary with the following keys:
     - 'Stride' : How much pixel the sliding window will travel
     - 'Pad' : The number of pixel that will be used to zero-pad the input.
    Return a tuple of:
    - out : Output data of shape (N,F,HH,WW)
      HH = (H + 2*Pad - K_H)/S + 1
      WW = (W + 2*Pad - K_W)/S + 1
    - cache : (x,w,b,conv_param)
    """
    N,C,H,W = x.shape
    F,C,K_H,K_W = w.shape
    # get parameter
    P = conv_param['pad']
    S = conv_param['stride']
    # Calculate output size and initialize output volume
    HH = (H + 2*P - K_H)/S + 1
    WW = (W + 2*P - K_W)/S + 1
    out = np.zeros((N,F,HH,WW))
    # pad the images with zeros on the border(Use to keep the spatial information)
    x_pad = np.pad(x,((0,0),(0,0),(P,P),(P,P)),'constant',constant_values=0)
    # Apply the convolution
    for n in range(N):
        for filters in range(F):
            for down in range(0,H,S):
                for right in range(0,W,S):
                    out[n,:,down/S,right/S] = np.sum(x_pad[n,:,down:down+S,right:right+S]*w[filters,:,:,:]) + b[filters]
    cache = (x,w,b,conv_param)
    return out,cache

def conv_backward_niave(dout,cache):
    """
    Compute the backward pass for the Convolution layer.
    Inputs:
    - dout : Upstream derivatives.
    - cache : A tuple of (x,w,b,conv_param) as in conv_forward_naive.
    Return a tuple of (dw,dx,db) gradients.
    """
    x,w,b,conv_param = cache
    N,F,HH,WW = dout.shape
    N,C,H,W = x.shape
    F,C,K_H,K_W = w.shape
    P = conv_param['pad']
    S = conv_param['stride']
    # Do zero padding on x
    x_pad = np.zeros(x,((0,0),(0,0),(P,P),(P,P)),'constant',constant_values=0)
    # Initilaize outputs
    dx = np.zeros(x_pad.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    # Calculate dx with 2 extra cols/rows that will be deleted
    for n in range(N):
        for filters in range(F):
            for down in range(0,H,S):
                for right in range(0,W,S):
                    dx[n,:,down:down+K_H,right:right+K_W] += dout[n,filters,down/S,right/S]*w[filters,:,:,:]
    # deleting padded rows to match real dx
    delete_rows = range(P) + range(H+P,h+2*P,1)
    delete_columns = range(P) + range(W+P,W+2*P,1)
    dx = np.delete(dx,delete_rows,axis=2)
    dx = np.delete(dx,delete_columns,axis=3)

    # Calculate dw
    for n in range(N):
        for filters in range(F):
            for down in range(HH):
                for right in range(WW):
                    dw[filters,:,:,:] += dout[n, filters, down, right]*x_pad[n, :, down*S:down*S+K_H, right*S:right*S+K_W]
    # Calculate db,1 scalar bais per filter,so it's just a matter of summing all elements of dout per filter
    for filters in range(F):
        db[depth] = np.sum(dout[:,filters,:,:])
    return dx,dw,db    
