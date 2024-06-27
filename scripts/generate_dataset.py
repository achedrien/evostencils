import argparse
import typing
import cv2
import numpy as np
import os


dataset_root = "/home/hadrien/Applications/mg_pytorch/evostencils/scripts/data/"
shape = "curve"
image_size = 128
start_index = 1
num_instances = 1000

def __gen_random_curve(c_x: float,
                       c_y: float,
                       r: float,
                       o_min: typing.Optional[float] = 0.8,
                       o_max: typing.Optional[float] = 1.2
                       ) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Generate a circle-based random curve.
    The base circle is centerd at (c_x, c_y) of radius r.
    The radius of this random curve will oscilate within the range [r * o_min, r * o_max].
    Refer to https://scicomp.stackexchange.com/questions/35742/generate-random-smooth-2d-closed-curves
    :param c_x:
    :param c_y:
    :param r:
    :param o_min:
    :param o_max:
    :return:
    """
    # noinspection PyPep8Naming
    H: int = 10
    A: np.ndarray = np.random.rand(H) * np.logspace(-0.5, -2.5, H)
    phi: np.ndarray = np.random.rand(H) * 2 * np.pi

    theta: np.ndarray = np.linspace(0, 2 * np.pi, 100)
    rho: np.ndarray = np.ones_like(theta)

    for i in range(H):
        rho += A[i] * np.sin(i * theta + phi[i])

    if o_min is not None:
        rho[rho < o_min] = o_min

    if o_max is not None:
        rho[o_max < rho] = o_max

    rho *= r

    # return np.min(rho), np.max(rho)

    x: np.ndarray = rho * np.cos(theta) + c_x
    y: np.ndarray = rho * np.sin(theta) + c_y

    return x, y

def __gen_random_bc_mask(image_size: int,
                         c_x: float,
                         c_y: float,
                         r: float,
                         o_min: typing.Optional[float] = 0.8,
                         o_max: typing.Optional[float] = 1.2
                         ) -> np.ndarray:
    """
    Generate a bc_mask composed of a circle-like random curve.
    The base circle is centerd at (c_x, c_y) of radius r.
    The radius of this random curve will oscilate within range [r * o_min, r * o_max].
    :param image_size:
    :param c_x:
    :param c_y:
    :param r:
    :param o_min:
    :param o_max:
    :return: Boolean bc_mask (image_size, image_size) of np.float32
    """
    img: np.ndarray = np.ones((image_size, image_size), dtype=np.uint8)
    x, y = __gen_random_curve(c_x, c_y, r, o_min, o_max)
    pts = np.array(list(zip(x, y)), dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], 0)
    return img.astype(np.float32)

def gen_punched_random_curve_region(image_size: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    A torus-like shape with smooth oscilating outer boundary and perfect circular inner boundary.
    :param image_size:
    :return: Boundary condition value (image_size, image_size) of np.float32,
             boolean boundary condition mask (image_size, image_size) of np.float32
    """
    # bc_mask
    outer: np.ndarray = __gen_random_bc_mask(image_size,
                                             image_size // 2,
                                             image_size // 2,
                                             image_size // 2 * 0.8,
                                             0.8,
                                             1.15)

    center = np.array([np.random.uniform(-0.16 * image_size / np.sqrt(2), 0.16 * image_size / np.sqrt(2)),
                       np.random.uniform(-0.16 * image_size / np.sqrt(2), 0.16 * image_size / np.sqrt(2))],
                      dtype=int) + image_size // 2
    radius = int(np.random.uniform(0.32 * image_size / 4, 0.32 * image_size / 2.5))
    inner: np.ndarray = cv2.circle(np.zeros_like(outer), center, radius, 1, cv2.FILLED)

    bc_mask: np.ndarray = inner + outer
    assert np.all(np.logical_or(bc_mask == 0, bc_mask == 1))

    # bc_value
    v = np.array([np.random.uniform(0.00, 0.25),
                  np.random.uniform(0.25, 0.50),
                  np.random.uniform(0.50, 0.75),
                  np.random.uniform(0.75, 1.00)])
    np.random.shuffle(v)

    half = (image_size + 1) // 2
    outer[:half, :half] *= v[0]
    outer[:half, half:] *= v[1]
    outer[half:, :half] *= v[2]
    outer[half:, half:] *= v[3]

    inner *= np.random.uniform(0, 1)

    bc_value = outer + inner

    return bc_value, bc_mask

def generate_punched_random_curve_region(dataset_root, shape, image_size, start_index, num_instances) -> None:
    os.makedirs(dataset_root, exist_ok=True)

    for i in range(start_index, start_index + num_instances):
        bc_value: np.ndarray
        bc_mask: np.ndarray

        bc_value, bc_mask = gen_punched_random_curve_region(image_size)

        data: np.ndarray = np.stack([bc_value, bc_mask])
        filename = os.path.join(dataset_root, 'curve_{:06d}'.format(i))
        np.save(filename, data)

def gen_random_square_region(image_size: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    bc_mask: np.ndarray = np.zeros((image_size, image_size), dtype=np.float32)

    def b1():
        bc_mask[:, :np.random.randint(image_size // 20, image_size // 3)] = 1

    def b2():
        bc_mask[:, -np.random.randint(image_size // 20, image_size // 3):] = 1

    def b3():
        bc_mask[:np.random.randint(image_size // 20, image_size // 3), :] = 1

    def b4():
        bc_mask[-np.random.randint(image_size // 20, image_size // 3):, :] = 1

    b = [b1, b2, b3, b4]
    np.random.shuffle(b)

    for i in range(np.random.randint(1, 5)):
        b[i]()

    # bc_value
    bc_value: np.ndarray = np.ones_like(bc_mask)

    v = np.array([np.random.uniform(0.00, 0.25),
                  np.random.uniform(0.25, 0.50),
                  np.random.uniform(0.50, 0.75),
                  np.random.uniform(0.75, 1.00)])
    np.random.shuffle(v)

    half = (image_size + 1) // 2
    bc_value[:half, :half] *= v[0]
    bc_value[:half, half:] *= v[1]
    bc_value[half:, :half] *= v[2]
    bc_value[half:, half:] *= v[3]

    bc_value *= bc_mask

    return bc_value, bc_mask

def generate_random_square_region(dataset_root, shape, image_size, start_index, num_instances) -> None:
    os.makedirs(dataset_root, exist_ok=True)

    for i in range(start_index, start_index + num_instances):
        bc_value: np.ndarray
        bc_mask: np.ndarray

        bc_value, bc_mask = gen_random_square_region(image_size)

        data: np.ndarray = np.stack([bc_value, bc_mask])
        filename = os.path.join(dataset_root, 'square_{:06d}'.format(i))
        np.save(filename, data)


def main(dataset_root, shape, image_size, start_index, num_instances) -> None:
    if shape == 'curve':
        generate_punched_random_curve_region(dataset_root, shape, image_size, start_index, num_instances)
    elif shape == 'square':
        generate_random_square_region(dataset_root, shape, image_size, start_index, num_instances)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main(dataset_root, shape, image_size, start_index, num_instances)
