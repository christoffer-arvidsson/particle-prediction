import deeptrack as dt
import numpy as np
from itertools import count
import tensorflow as tf

def create_particle_dataset(image_size=64, sequence_length=8, batch_size=8):
    MIN_SIZE=.5e-6
    MAX_SIZE=1.5e-6
    MAX_VEL=10 # Maximum velocity. The higher the trickier!
    MAX_PARTICLES=3 # Max number of particles in each sequence. The higher the trickier!

    # Defining properties of the particles
    particle=dt.Sphere(intensity=lambda: 10+10*np.random.rand(),
                       radius=lambda: MIN_SIZE+np.random.rand()*(MAX_SIZE-MIN_SIZE),
                       position=lambda: image_size*np.random.rand(2),vel=lambda: MAX_VEL*np.random.rand(2),
                       position_unit="pixel")

    # Defining an update rule for the particle position
    def get_position(previous_value,vel):
        newv=previous_value+vel
        for i in range(2):
            if newv[i]>63:
                newv[i]=63-np.abs(newv[i]-63)
                vel[i]=-vel[i]
            elif newv[i]<0:
                newv[i]=np.abs(newv[i])
                vel[i]=-vel[i]
        return newv

    particle=dt.Sequential(particle,position=get_position)

    # Defining properties of the microscope
    optics=dt.Fluorescence(NA=1,output_region= (0, 0,image_size, image_size),
                           magnification=10,
                           resolution=(1e-6, 1e-6),
                           wavelength=633e-9)

    # Combining everything into a dataset. Note that the sequences are flipped in
    # different directions, so that each unique sequence defines in fact 8 sequences
    # flipped in different directions, to speed up data generation
    dataset=dt.FlipUD(dt.FlipDiagonal(dt.FlipLR(dt.Sequence(optics(particle**(lambda: 1+np.random.randint(MAX_PARTICLES))),sequence_length=sequence_length))))

    return dataset

def create_autoencoder_generator(image_size=64, batch_size=8):
    dataset = create_particle_dataset(image_size, 1, batch_size)
    generator = dt.generators.Generator()
    gen = generator.generate(
        dataset,
        lambda x: x,
        batch_size=batch_size,
    )
    return gen

def create_sequence_generator(image_size=64, batch_size=8, prior_length=8, truth_length=1):
    dataset = create_particle_dataset(image_size, prior_length + truth_length, batch_size)
    generator = dt.generators.Generator()
    gen = generator.generate(
        dataset,
        batch_function = lambda x: x[:prior_length],
        label_function = lambda x: x[prior_length:],
        batch_size=batch_size,
        ndim=5,
    )

    return gen
