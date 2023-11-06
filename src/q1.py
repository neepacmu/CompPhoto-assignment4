import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter

def plot_mosiac(lf):

    size = lf.shape[0]

    cnt = 1

    for i in range(size):
        for j in range(size):
            
            plt.subplot(size, size, cnt)

            plt.imshow(lf[i,j])
            plt.axis('off')
            cnt +=1

    plt.subplots_adjust(left=0,
                    bottom=0,
                    right=1,
                    top=1,
                    wspace=0,
                    hspace=0)

    plt.savefig("output/1_2.jpg", bbox_inches='tight', dpi=2000)


def plot_mosiac_afi(lf):

    size = lf.shape[0]

    grid_x = lf.shape[0]
    grid_y = lf.shape[-1]

    cnt = 1

    for i in range(grid_x):
        for j in range(grid_y):
            
            plt.subplot(grid_x, grid_y, cnt)

            #plt.imshow(lf[i,y,x,:,j].reshape((1,1,3)))
            #print(lf[i,:,:,:,j].max(), lf[i,:,:,:,j].min())
            plt.imshow(lf[i,:,:,:,j])
            plt.axis('off')
            cnt +=1

    plt.subplots_adjust(left=0,
                    bottom=0,
                    right=1,
                    top=1,
                    wspace=0,
                    hspace=0)

    plt.savefig("output/1_5.png", bbox_inches='tight', dpi=2000)


def form_lightfield(image, size):

    height, width, channels = image.shape

    lf = np.zeros((size, size, height//size, width//size, 3))
    
    lf = np.zeros((size, size, height//size, width//size, 3))

    for j in range(0, height, size):
        for i in range(0,width, size):  
            
            j_idx = j//size
            i_idx = i//size

            lf[:,:,j_idx,i_idx,:] = image[j:j+size,i:i+size,:]

    return lf


def generate_uv_coordinates():

    lensletSize = 16
    maxUV = (lensletSize - 1) / 2
    u = np.arange(lensletSize) - maxUV
    v = np.arange(lensletSize) - maxUV

    return u,v 


def integrate(lf, depth, mask_u, mask_v):

    _, _, h, w, c = lf.shape

    u_coords,v_coords = generate_uv_coordinates()

    s = np.arange(h)
    t = np.arange(w)

    img_integrated = np.zeros((h,w,c,len(depth)))

    for channel in range(c):
        for i in range(16):
            for j in range(16):
                if mask_v[i] and mask_u[i]:

                    func = interp2d(t, s, lf[i,j,:,:,channel])

                    for id, d in enumerate(depth):

                        img_integrated[:, :, channel, id] += func(t + v_coords[j]*d, s - u_coords[i]*d)


    temp_deno = np.sum(mask_u)*np.sum(mask_v)

    img_integrated = (img_integrated/temp_deno).astype(np.uint8)

    return img_integrated


def focus_image(imgs, s1, s2, depth):

    h, w, c, d = imgs.shape

    I_l = np.zeros((h, w, d))

    I_low = np.zeros_like(I_l)
    I_high = np.zeros_like(I_l)
    
    weights = np.zeros((h, w, d))

    for d_iter in range(d):
        I_l[:,:,d_iter] = cv2.cvtColor(imgs[:, :, :, d_iter], cv2.COLOR_RGB2XYZ)[:,:,1]
        
        I_low[:,:,d_iter] = gaussian_filter(I_l[:,:,d_iter], s1)
        I_high[:,:,d_iter] = I_l[:,:,d_iter] - I_low[:,:,d_iter]

        weights[:,:,d_iter] = gaussian_filter(I_high[:,:,d_iter]**2, s2)


    infocus_imgs = np.zeros((h,w,c))

    print(infocus_imgs.shape)

    for channel in range(3):

        temp_n = np.sum(imgs[:,:,channel,:]*weights, axis = -1)
        temp_d = np.sum(weights, axis = -1)

        infocus_imgs[:,:,channel] = (temp_n/temp_d)

    infocus_imgs = infocus_imgs.astype(np.uint8)

    depth_image = np.sum(weights*np.array(depth), axis = -1)/np.sum(weights, axis = -1)
    min_d = np.min(depth_image)
    depth_image = depth_image - min_d

    max_d = np.max(depth_image)
    depth_image = depth_image/ max_d
    depth_image = (depth_image*255).astype(np.uint8)

    return infocus_imgs,depth_image



def confocal_stereo_depth_estimation(imgs, depth, aperture):

    afi = []

    for ap in aperture:

        mask_u = np.ones(16)
        mask_v = np.ones(16)

        cancel = (16 - ap)//2
        
        mask_u[:cancel] = 0
        mask_u[-cancel:] = 0

        mask_v[:cancel] = 0
        mask_v[-cancel:] = 0
        print(imgs.shape, depth, mask_u.shape, mask_v.shape)
        curr_lf = integrate(imgs, depth, mask_u, mask_v)

        afi.append(curr_lf)

    afi = np.array(afi)

    plot_mosiac_afi(afi)
    
    afi_var = np.var(afi, axis = 0)

    depth = np.argmax(np.sum(afi_var, axis = 2), axis = -1)
    
    depth = depth/np.max(depth)

    return depth*255


if __name__ == "__main__":


    im = cv2.imread('data/chessboard_lightfield.png')
    block_size = 16

    lf = form_lightfield(im, size = block_size).astype(np.uint8)

    plot_mosiac(lf)

    mask_u = np.ones(16)
    mask_v = np.ones(16)

    depth_vals = [-1.2, -1, -0.7, -0.4, -0.2, 0]
    aperture = [2, 6, 10, 12, 14]

    s1 = [0.2, 0.5, 0.7]
    s2 = [5, 25, 50]

    print(lf.shape, depth_vals, mask_u.shape, mask_v.shape)
    out_imgs = integrate(lf, depth_vals, mask_u, mask_v)

    for i in range(out_imgs.shape[3]):
        cv2.imwrite(f'output/1_3_d-{i}.png', out_imgs[:,:,:,i])

    for s_1 in s1:
        for s_2 in s2:
            focused_image, depth = focus_image(out_imgs, s_1, s_2, depth_vals)
            print(s_1, s_2)
            cv2.imwrite(f'output/1_4_focus_{s_1}_{s_2}.png', focused_image)
            cv2.imwrite(f'output/1_4_depth_{s_1}_{s_2}.png', depth)

    temp_lf = form_lightfield(im, size = block_size).astype(np.uint8)

    conf_depth = confocal_stereo_depth_estimation(lf, depth_vals, aperture)

    cv2.imwrite(f'output/1_5_depth.png', conf_depth)
