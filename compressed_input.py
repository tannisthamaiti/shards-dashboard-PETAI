import numpy as np
import blz
from sys import getsizeof
import dlisio
dlisio.set_encodings(['latin1'])

#read DLIS file
def read_file(filepath):
    image_array = None
    with dlisio.load(filepath) as file:
        for d in file:
            image_channels = d.match('FMI_DYN')
            for channel in image_channels:
                image_array = channel.curves()
    return image_array


if __name__ == "__main__":
    
    filepath = 'data/University_of_Utah_MU_ESW1_FMI_HD_7440_7550ft_Run2.dlis'
    output_dir = 'data/blz_input'
    arr = read_file(filepath)
    print(f'Original file size: {getsizeof(filepath)} MB')
    print(f'Saving numpy array to data/np_input.npz')
    np.savez('data/np_input.npz', data=arr)
    print(f"npz compressed size: {getsizeof('np_input.npz')} MB")
    b_arr = blz.barray(arr, rootdir=output_dir)
    print(f'Saving blz array to {output_dir}...')
    print(f'blz compressed size: {round(b_arr.cbytes / 1024 / 1024,2)} MB')

    # read npz array
    np_arr = np.load('data/np_input.npz')['data']
    # read blz saved files
    b_arr = blz.open('data/blz_input', mode='r')
    print('Are both numpy and blz array same?', np.array_equal(np.array(b_arr), np_arr))
    # convert blz array to numpy
    # np_b_arr = np.array(b_arr)
