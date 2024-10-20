

# WEEK 1:
- Problemas modernos de la AI: Generalization, Robustness. Expensive, Non-Generalizable, Non-Robust.
- What is learning? Info suficiente, Toma de decisiones, Reconocer patrones, Responder a estimulo, Comprimir informacion, Inferir a partir de info previa. Comprimir informacion => Learning es cualquier cosa q se pueda optimizar.
- Diferencia entre ML y DL: ML es algoritmos, DL es bioinspirado. ML es heuristica, DL es optimizacion. DL es subconjunto de ML. Uso de neuonas artificiales en DL y composicion de building blocks. DL es "brain inspired"
- Interpretability, Complexity, Feature extraction.
- Half-Wave Rectification
- Important Characteristics of a "Learning Machine"
    - Architecture
    - Dataset
    - Parameters
    - Optimization/Learning Rule
    - Input/Output
    - Inductive Bias
    - Computations
- Why is the Architecture important?
    - Architectures *can* play some reole in the out-of-distribution regime (and thus aid in greater generalization)
    - Different Architectures can enable better expresiveness and approximation power
    - Some architectures could provide better stability in learning (Residual Networks).
- The representation will affect Generalization Capacity. The output or task will affect the representation.
- The Hubel&Wisel experiment (Nobel Prize 1981) showed that the visual cortex has a hierarchical structure.
- The Three Levels of Marr:
    - Computational
    - Algorithmic
    - Implementational

# WEEK 2 (CNNs Part 1):
- Humans aren't conscious (or know) how they do inference for object recognition. They just do it.
- But we can observe and try to decompose how this pipeline is done, to implement and adaptaed version for a computer.
- Prior de la escena: Human vision scientists, reconocer los objetos antes (chicken and egg)
- Como harian para clasificar escenarios bosque, playa, sierra sin DL: Histograma de colores.
- Que feature space extraerias de una imagen sin imagenes. No operar sobre las imagenes en bruto sino generar una transformacion previa. Promediar sobre los patches de la imagen.
- Paradoja de Morabe: Dificiles para el humano, faciles para la computadora y al reves. Como el ajedrez
- Percepcion: una cosa es sensing (ojo/camara) y otra cosa es procesamiento/percepcion.
- Challenges in Computer Vision:
    - Object detection under different: pose, color, illumination, occlusion, shape and instance.
- AI ethics
- Feature inversion: Invertir la representacion de la imagen
- Descriptor: Embedding/Function. Classical: ecuacion fumada. Nuevo: descriptor optimizable (kernel)
- De q espacio a q espacio esta convirtiendo el estimulo a la imagen.
- Por que normalizar: Achicar los rangos de la imagen para que los kernesl del descriptor no tengan un espacio rango muy grande de valores.
- Diferencia entre hiperparametro y parametro: estatico acoplado a la arquitecctura y sistema optimizable vs. elemento optimizable
- Solo por quee freezeas los pesos un parametro no se vuelve hiperparametro. Hiperparametro esta externo al sistema.
- Probabilidad bayesiana para calcular la clase de la imagen.
- Solo por que para el humano sea facil, para la computadora no necesariamente tiene q serlo.
- Antes: Feature extractor + Classifier
- Ahora: Feature extractor + High Level + Low level. Jointly optimizable.
- Validacion empirica de q las redes neuronales tienen capacidad. Neurocognitro
- Ilya hizo sus redes neuronales usando CUDA. Primero el concepto, luego pivotear en el Hardware
- Visualizing Deep Networks
- Sparse coding
- Q tiene q pasar para q un filtro tenga X forma.
- Cuales deberian ser los filtros si es q estamos en el mundo de Minecraft
- Los animales tienen otros sistemas visuales optimos como el conejo que mira mas abajo y ancho. Mientras que el humano es mas cuadriculado
- Visualizaciones de feature inversion
- Maximizar la respuesta del estimulo. Image Synthesis de maximum response.
- Imagen adversaria: Imagen preparada para enga;ar al sistema
- Invarianza y equivarianza.
- Synthesizing Robust Adversarial Examples
- Invisible cloth. confunden a la hora de optimizar una red
- Adversarial images classes
- Single pixel attack para las fotos de las VISAs , se usa para la guerra.
- Robustez a las perturbaciones/distorsion
- Imagen + Delta: Adverasrial image. Maximizar la perturbacion de la imagen para q cambie de clase, para ver que perturbaciones o que patches adversarios se tienen q generar para enga;ar al sistema
- CNN: idea de aprendizaje jerarquico

# WEEK 3 (CNNs Part 2):

- AlexNet vs VGG vs ResNet
    - Convolution, Half-Wave Rectification, Pooling, Fully Connected
- Feature Spaces in CNNs
    - Input, Kernel, Activation Map (Output)
- U-Nets and De-convolution
- Preamble to Generative Models
    - Image-to-Image Translation with Conditional Adversarial Networks
        - Texture Synthesis
        - Image Denoising
        - Image Compression


```
What is learning?
	Optimize action

DL vs ML
	DL: es biologicamente inspirado
	ML: algoritmos 

What is learning
DL vs ML
Texture Bias
Interaccion entre tarea, task (output) y loss function

3 conos de la retina (RGB)

one hot encoding ->
	[1 ... 0]
	cada clase es un eje de un espacio vectorial
		1 dimension distinta de 1 clase
pq en CV no necesitamos Point Cloud

FIJA N.8
	pq la imagen no puede representada por un conjunto?
		pq el orden importa
		el pixel esta correlacionado con su vecindad

embedding -> vector q resume las propiedades del estimulo
vector caracteristico -> la clase

2 redes neuronales que son entrenadas con inputs (con transformaciones previamente distintas) aprenden representaciones distintas

Expresiveness of Architecture should match Task?
FIJA N.9
	Porque el CNN le va mejor en tareas de clasificacion de objetos que la MLP?
		Jerarquia y agrupacion de partes en cascada
		Relacion geometrica para conformar una parte de algo
			ojos -> cara -> cabeza -> cuerpo
		Desde un punto de vista conceptual
		La CNN geometricamente integre localmente la informacion y abstraiga la informacion

Hay una razon especifica por la cual escoger un modelo u otro.

FIJA N.10
	Cuando entrenan un clasificador de objetos. Por lo general se usa cross-entropy loss
	Se puede entrenar un clasificador de objetos con MSE

Empiricamente un loss es mejor q otro

The Three Levels of Marr

Cambiar dataset de CIFAR o MNIST

Con la camara simular el punto de vista del condor, y del raton (de arriba a abajo, de abajo al frente)

Practicar DataAugmentation para el lab del viernes
Practicar AdversarialTraining
```

# WEEK 5 (GAN + Image Pyramids):

- Architecture: Generator + Discriminator
    - Generator: Tries to generates more real-like fake bills
    - Discriminator: Tries to catch fake bills. Penalty if failure
- MinMaxGame

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

- KL-divergence is not simetric

$$
KL(p||q) = \sum_x p(x) \log \frac{p(x)}{q(x)}
$$

- Maximum Likelihood vs Reverse KL (Mode Dropping).
- GAN training algorithm: $k$ step optimization for Generator, 1 step optimization for Discriminator.
- GAN types: CGAN, StyleGAN, WGAN, DCGAN
- Pyramid input
    - Blending + Compression

# WEEK 6 (VAEs + Compression + Perceptual Optimization):

- VAE Architecture
    - Encoder: $q(z|x)$
    - Decoder: $p(x|z)$
- The reparametrization trick
    - $z = \mu + \sigma \odot \epsilon$
- KL Divergence
$$
D_{KL}(p||q) = \sum_x p(x) \log \frac{p(x)}{q(x)} \\
D_{KL}(p||q) = - \sum_x p(x) \log \frac{q(x)}{p(x)} \\ 
~ \\
D_{KL}(q||p) = \sum_x q(x) \log \frac{q(x)}{p(x)}
$$
- Jensen Shannon Divergence
$$
JSD(p || q) = JSD(q || p)
$$
- VAE characteristics:
    - Regularized latent space: Continuity
- ELBO: Evidence Lower Bound
- The reparametrization trick is used to make the VAE differentiable
- Another VAE architectures: Beta-VAE

```
hombre, maquina, bestia	
Como comparar informacion de diversos sistemas

FIJA N 47

	GANs != Adversarial Images

Adversarial Images
	Optimize image to maximize the loss

GANs crea imagen a partir del ruido (noise)

GAN
	que es mode dropping
		dropeo el modo de gatos

	FIJA N 50
		el vector condicional latente es un vector o no 
		
Fija 53: KL Divergence

KL no es una metrica de distancia
	no es simetrica
	DKL >= 0

Jensen Shanon Divergence
	equivalencia con DKL
	optimal transport theory -> mover una distribucion de una a otra
		diferencias entre mover monticulos de arena para volverlos como la otra (los monticulos de arena son las distribuciones)
		variational bayesian methods
	esta boundeado entre 0 y log_b(2)

el VAE explicitamente busca aproximar la distribucion, el GAN lo hace implicitamente

Beta-VAE
```


# WEEK 7-8 (NeuroAI):

- Basic Principles
    - Neuroscience
        - Is the scientific discipline that consists on the study and discovery of the mechanisms of processing and represetion in the brain and neronal systems of biological systems.
            - Experimental of Frog
            - Experimental of ZebraFish
            - Behaviour in Goats
            - Prosopagnosia
    - AI:
        - Consists of the engineering discipline that has the ultimage goal of creating and understanding intelligent machines
            - AI for Learning + Control
            - Computer Vision (Detectron)
            - NLP & Multimodal Understanding
- NeuroAI
    - Engineering
        - How do we use Neuroscience to invent better systems of AI?
        - How do we use AI to discover principles in Neuroscience?
    - Is a symbiotic new discipline between AI and Neuroscience that mixes both aspects of science and engineering
- Feature Representation & Data Collection
    - Non-Invasive Data Collection
        - Tiempo de respuesta del mono presionando los numeros
    - Semi-Invasive Data Collection
        - AR/VR en la mosca
    - Data collection in Humans
        - Neurophysiology
        - EEG, fMRI
        - MEG
- Representational Similarity Analysis (RSA)
    - Assessment of Internal Representation
    - FIJA (RSA) -> Representational Similarity Analysis (Discretization, Dimensionality, Ethics, Noise in Biological System)
        - Dar cantidad de imagenes igual a sistema artificial que a sistema humano/mono, medir la respuesta al estimulo (embedding). 
        - Calcular la variacion interna de los feature vectors (matriz todos contra todos), sacar esa matriz para los 2 sistemas, sacar la correlacion entre las 2 matrices GRAM 
        - Si las correlacion es 1 los sistemas estan alineados. Matriz de Gram. Expected de las Grammian Matrices
    - PyTorch Hooks - Feature ReadOut - Meterle el electrodo o la matriz de respuesta a la red neuronal/Data Collection in Machines -> Extraer la respuesta al estimulo dado un input
- Psychophysics (Behaviour)
    - Physical Domain (Stimulus) => Psychological Domain (Response)
        - The word psychophysics literally comes from a fusion of using methods of physics to find an underlying mechanism, and applying it not to physical laws, but to behaviour and perception
    - Psycophisics: Ir del Estimulo a la Respuesta: NOT-FC, 2AFC (spatial), 2AFC(temporal)
    - 2AFC stands for: Two Alternative Forced Choice
    - Assesment of Behaviour/Output
    - Is My Blue your Blue?
    - Peripheral Oddity Tasks. Curva perceptual. Cambiar como cambia la proyeccion dimensional cambia
    - Human as Ground Truth Observer y viceversa
    - Examples
        - Fitting a distortion variable to human limits of perception
        - Testing limits of perception in humas given renderings of images in machines
- Application of RSA + Psychophysics: Are Transformer good models of the Ventral Stream?
    - Multi-Resolution + Local Texture Computation
    - Dual-branch vision transformer to extract multi-scale feature representations + Gramian-like local texture computation
    - Feature Inversion