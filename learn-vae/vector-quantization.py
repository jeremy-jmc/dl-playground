# -----------------------------------------------------------------------------
# Vector Quantization
# -----------------------------------------------------------------------------
import numpy as np
from scipy.cluster.vq import vq, kmeans
np.random.seed(0)

"""
Vector quantization (VQ) is a data compression technique similar to k-means algorithm which can model any data distribution.

For VQ process, we require a codebook which includes a number of codewords. 
Applying VQ on a data point (gray dots) means to map it to the closest codeword (blue dots), 
i.e. replace the value of data point with the closest codeword value.

See voronoi diagram for more details.

https://github.com/MHVali/Additive-Residual-Product-Vector-Quantization-Methods/tree/main
https://github.com/lucidrains/vector-quantize-pytorch/tree/master
"""

def vector_quantization_scipy():
    # Generate some random data
    data = np.random.rand(50, 2)
    # Print the data
    print('data:', data, '\n')

    # Perform vector quantization
    codebook, _ = kmeans(data, 5)
    quantized, _ = vq(data, codebook)

    # Print the codebook
    print('codebook:', codebook, '\n')
    # Print the quantized data
    print('quantized:', quantized.reshape(-1, 1), '\n')


def vector_quantization_pytorch():
    import torch
    import matplotlib.pyplot as plt
    from vector_quantize_pytorch import VectorQuantize, ResidualVQ, GroupedResidualVQ, RandomProjectionQuantizer, FSQ, ResidualFSQ, LFQ, ResidualLFQ

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dim = 256
    cluster_number = 512
    n_samples = 1024
    x = torch.randn(1, n_samples, dim)
    print('x:', x)

    vq = VectorQuantize(
        dim=dim,
        codebook_size=cluster_number,
        decay=0.8,             # the exponential moving average decay, lower means the dictionary will change faster
        commitment_weight=1.,   # the weight on the commitment loss

        kmeans_init=True,
        learnable_codebook=True,
        in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.SGD(
            *args, **kwargs, lr=1.0e-3, momentum=0.9 # weight_decay=1.0e-6
        ),
        # affine_param=True,
        # affine_param_batch_decay=0.99,
        # affine_param_codebook_decay=0.9,
        ema_update=False,
        )

    # * target: minimize the commitment loss
    quantized, indices, commit_loss = vq(x) # (1, 1024, dim), (1, 1024), (1)
    print('indices shape:', indices.shape)
    print('quantized shape:', quantized.shape)
    print('commit_loss:', commit_loss)
    print('codebook shape:', vq.codebook.shape)


def learn_vector_quantization():
    # TODO: implement learn VQ algorithm
    pass

if __name__ == '__main__':
    # vector_quantization_scipy()
    vector_quantization_pytorch()
    # learn_vector_quantization()

"""
https://github.com/search?type=code&q=%22from+vector_quantize_pytorch+import+VectorQuantize%22+language%3APython
https://github.com/lucidrains/vector-quantize-pytorch/tree/3af411025c58822129e867dc6523d6a24743031f
https://github.com/lucidrains/DALLE2-pytorch/blob/680dfc4d93b70f9ab23c814a22ca18017a738ef6/dalle2_pytorch/vqgan_vae.py#L544
https://github.com/wilson1yan/VideoGPT/blob/733d2b1541e166d7f29df514c7d196c768011dd2/videogpt/vqvae.py#L14
https://github.com/Project-MONAI/GenerativeModels/blob/bdca1bd1ea8bda4960156285f805082fff7e8686/generative/networks/nets/vqvae.py#L274
https://docs.scipy.org/doc/scipy/reference/cluster.vq.html
https://github.com/naver/PoseGPT/blob/bb2045ca1cc2645d28426f134ff7ed11597a6f8e/models/base.py#L32
https://github.com/qiqiApink/MotionGPT/blob/a1c939b34b8f4e73ba25326e5f934b46f25896e1/train_vqvae.py#L9
https://github.com/lucidrains/vector-quantize-pytorch/tree/master
https://github.com/MHVali/Noise-Substitution-in-Vector-Quantization/blob/main/NSVQ.py

https://towardsdatascience.com/optimizing-vector-quantization-methods-by-machine-learning-algorithms-77c436d0749d
https://speechprocessingbook.aalto.fi/Modelling/Vector_quantization_VQ.html
https://medium.com/@udbhavkush4/demystifying-learning-vector-quantization-a-step-by-step-guide-with-code-implementation-from-ea3c4ab5330e
https://machinelearningmastery.com/implement-learning-vector-quantization-scratch-python/
https://www.turing.com/kb/application-of-learning-vector-quantization

VQ-VAE
https://github.com/sohananisetty/motion_vqvae/blob/bfbd4b3bf3eadf08c8f8323ca06053f9a8099590/core/models/conv_vqvae.py#L83
https://github.com/nadavbh12/VQ-VAE/tree/master
https://github.com/zalandoresearch/pytorch-vq-vae/tree/master
"""