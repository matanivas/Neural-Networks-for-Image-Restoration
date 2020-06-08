import imageio
import numpy as np
from scipy import signal
from keras.layers import Input, Activation, Conv2D, Add
from keras.models import Model
from skimage.color import rgb2gray
from keras.optimizers import Adam
import sol5_utils
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve



GREY = 1


"""
this function gets as parameters image path and its representation and read the image at the specified representation
"""


def read_image(filename, representation):
    im_float = imageio.imread(filename)
    type = im_float.dtype
    if type == int or type == np.uint8:
        im_float = im_float.astype(np.float64) / 255
    if representation == GREY:
        return rgb2gray(im_float)
    return im_float


def samp_patch(im, shape0, shape1, crop_size):
    return im[shape0: shape0 + crop_size[0], shape1: shape1 + crop_size[1]]

#
def load_dataset(filenames, batch_size, corruption_func, crop_size):
    cached_ims = dict()
    while True:
        source = np.zeros((batch_size, crop_size[0], crop_size[1], 1))
        target = np.zeros((batch_size, crop_size[0], crop_size[1], 1))
        images_paths = np.random.choice(filenames, size=batch_size)
        count = 0
        for image_path in images_paths:
            if not image_path in cached_ims:
                image = read_image(image_path, GREY)
                cached_ims[image_path] = image
            image = cached_ims[image_path]
            large_shape0_patch = np.random.randint(image.shape[0] - (3 * crop_size[0]))
            large_shape1_patch = np.random.randint(image.shape[1] - (3 * crop_size[1]))
            shape0_patch = np.random.randint(crop_size[0] * 2)
            shape1_patch = np.random.randint(crop_size[1] * 2)
            patch = samp_patch(image, shape0_patch + large_shape0_patch, shape1_patch + large_shape1_patch, crop_size).reshape(crop_size[0],  crop_size[1], 1)
            large_crop_size = (3 * crop_size[0], 3 * crop_size[1])
            corrupted_large_patch = corruption_func(samp_patch(image, large_shape0_patch, large_shape1_patch, large_crop_size))
            corrupted_patch = samp_patch(corrupted_large_patch, shape0_patch, shape1_patch, crop_size).reshape(crop_size[0],  crop_size[1], 1)
            target[count] += (patch - 0.5)
            source[count] += (corrupted_patch - 0.5)
            count += 1
        yield (source, target)


def resblock(input_tensor, num_channels):
    a = Conv2D(num_channels, (3, 3), padding='same', activation='relu')(input_tensor)
    a = Conv2D(num_channels, (3, 3), padding='same')(a)
    a = Add()([input_tensor, a])
    return Activation('relu')(a)


def build_nn_model(height, width, num_channels, num_res_blocks):
    input = Input(shape=(height, width, 1))
    a = Conv2D(num_channels, (3, 3), padding='same', activation='relu')(input)
    for i in range(num_res_blocks):
        a = resblock(a, num_channels)
    a = Conv2D(1, (3, 3), padding='same')(a)
    a = Add()([input, a])
    return Model(input, a)


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    crop_size = model.input_shape[1:3]
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    partition = int((len(images)*4)/5)
    training_set = load_dataset(images[0: partition], batch_size, corruption_func, crop_size)
    validation_set = load_dataset(images[partition:], batch_size, corruption_func, crop_size)
    model.fit_generator(generator=training_set, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                        validation_data=validation_set, validation_steps=(num_valid_samples/batch_size))


def restore_image(corrupted_image, base_model):
    a = Input(shape=(corrupted_image.shape[0], corrupted_image.shape[1], 1))
    b = base_model(a)
    extended_model = Model(a, b)
    prediction = extended_model.predict((corrupted_image - 0.5).reshape(1, corrupted_image.shape[0],
                                                                        corrupted_image.shape[1], 1))
    prediction = prediction.reshape(corrupted_image.shape)
    prediction += 0.5
    prediction = np.clip(prediction, 0, 1)
    return prediction.astype(np.float64)




def add_gaussian_noise(image, min_sigma, max_sigma):
    variance = np.random.uniform(min_sigma, max_sigma)
    noise = np.random.normal(0, variance, image.shape)
    noised = np.round(255*(image + noise)) / 255
    noised = np.clip(noised, 0, 1)
    return noised


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    model = build_nn_model(24, 24, 48, num_res_blocks)
    if quick_mode:
        batch_size = 10
        steps_per_epoch = 3
        num_of_epochs = 2
        valid_samples = 30
    else:
        batch_size = 100
        steps_per_epoch = 100
        num_of_epochs = 5
        valid_samples = 1000
    train_model(model, sol5_utils.images_for_denoising(), lambda image: add_gaussian_noise(image, 0, 0.2), batch_size,
                steps_per_epoch, num_of_epochs, valid_samples)
    return model


def add_motion_blur(image, kernel_size, angle):
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return convolve(image, kernel)


def random_motion_blur(image, list_of_kernel_sizes):
    angle = np.random.uniform(0, np.pi)
    kernel_size = np.random.choice(list_of_kernel_sizes)
    blurred = add_motion_blur(image, kernel_size, angle)
    return blurred


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    model = build_nn_model(16, 16, 32, num_res_blocks)
    if quick_mode:
        batch_size = 10
        steps_per_epoch = 3
        num_of_epochs = 2
        valid_samples = 30
    else:
        batch_size = 100
        steps_per_epoch = 100
        num_of_epochs = 10
        valid_samples = 1000
    train_model(model, sol5_utils.images_for_deblurring(), lambda image: random_motion_blur(image, [7]), batch_size,
                steps_per_epoch, num_of_epochs, valid_samples)
    return model


def depth_analysis(quick_mode=False):
    mse_denoising = get_mse(quick_mode, is_denoising=True)
    mse_deblurring = get_mse(quick_mode, is_denoising=False)
    plot_mse(mse_denoising, "denoising")
    plot_mse(mse_deblurring, "deblurring")


def get_mse(quick_mode, is_denoising):
    mse = []
    if is_denoising:
        for i in range(1, 6):
            model_noise = learn_denoising_model(i, quick_mode)
            cur_mse = model_noise.history.history['val_loss'][-1]
            mse.append(cur_mse)
    else:
        for i in range(1, 6):
            model_blurr = learn_deblurring_model(i, quick_mode)
            cur_mse = model_blurr.history.history['val_loss'][-1]
            mse.append(cur_mse)
    return mse


def plot_mse(mse, model_name):
    plt.figure()
    plt.title("MSE as function of number of resnet blocks- " + model_name + " model")
    plt.xlabel("num of resnet blocks")
    plt.ylabel("MSE")
    plt.xticks(range(0, 5), range(1, 6))
    plt.plot(mse, marker='.')
    plt.show()

def plot_q2():
    mse_noise = get_mse(False, True)
    mse_blurr = get_mse(False, False)
    plot_mse(mse_noise, "Denoising")
    plot_mse(mse_blurr, "Deblurring")
