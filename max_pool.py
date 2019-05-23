def max_pool_forward_naive(x,pool_param):
    """
    Compute the forward max pooling
    Inputs:
    - x: 4d Input tensor of shape (N,C,H,W)
    - pool_param: dictionary with following keys:
      - 'pool_height/width': Sliding window height/width
      - 'stride': Sliding moving distance
    Return a tuple of : (out,cache)
    """
    #Get input tensor and parameter data
    N,C,H,W = x.shape
    S = pool_param['stride']
    # Consider H_F and W_F as the sliding window height and width
    K_H = pool_param['pool_height']
    K_W = pool_param['pool_width']
    # Calculate output size
    HH = (H - K_H)/S + 1
    WW = (W - K_W)/S + 1
    out = np.zeros((N,C,HH,WW))
    # Calculate output
    for n in range(N): # For each image in the batch
        for channel in range(C): # For each channel in image
            for vns in range(0,H,S): # Slide vertically taking stride into account
                for hns in range(0,W,S): # Slide horizontally taking stride into account
                    out[n,channel,r/S,c/S] = np.max(x[n,channel,vns:vns+K_H,hns:hns+K_W])
    cache = (x,pool_param)
    return out,cache

def max_pool_backward_naive(dout,cache):
    """
    Compute the backward propagation of max pooling
    Inputs:
    - dout: Upstream derivative,same size as cahed x
    - cache: A tuple of (x,pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    # Get data back from cache
    x, pool_param = cache
    S = pool_param['stride']
    K_H = pool_param['pool_height']
    K_W = pool_param['pool_width']
    N,C,HH,WW = dout.shape
    #Initialize dx
    dx = np.zeros(x.shape)
    # Caclculate dx (mask*dout)
    for n in range(N):
        for channel in range(C): # for each channel
            for vn in range(HH): # vertical node
                for hn in range(WW): # horizontal node
                    # Construct the original image from which pooling occur and calculate the mask
                    x_pool = x[n, channel, vn*S:vn*S+K_H, hn*S:hn*S+K_W]
                    mask = (x_pool == np.max(x_pool))
                    # Calculate mask*dout
                    dx[n,channel,vn*S:vn*S+H_F,hn*S:hn*S+W_F] = mask*dout[n,channel,vn,hn]
    return dx
