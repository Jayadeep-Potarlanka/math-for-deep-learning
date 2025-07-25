# Mathematical Foundations of Deep Learning: Implementation Project

This project implements core mathematical concepts essential for deep learning, drawing from linear algebra, metric spaces, information theory, and dimensionality reduction. The codebase consists of a Jupyter Notebook that demonstrates numerical computations, visualizations, and algorithmic implementations. All work is self-contained, with explanations of methods, functions, and results integrated directly into the notebook.

I developed the project with a focus on practical coding to explore these foundations. Key implementations include visualizations of norms, proofs of information theory identities, entropy calculations for distributions and images, KL divergence, and a from-scratch t-SNE algorithm.

## Prerequisites

To execute the code, install the following Python libraries:

- numpy (for array operations and numerical computations)
- matplotlib (for plotting and visualizations)
- scipy (for integration and scientific functions)
- scikit-image (for image loading and processing)
- scikit-learn (optional, for comparative t-SNE visualizations)

Install them via pip:
``` bash
pip install numpy matplotlib scipy scikit-image scikit-learn
```


## Usage Instructions

1. Download or clone the repository.
2. Place the required grayscale image files (e.g., TIFF images used for entropy calculations) in the same directory as the notebooks or adjust paths.
3. Open the notebook in a Jupyter environment (e.g., JupyterLab, Jupyter Notebook, or VS Code):
   - `math-for-deep-learning.ipynb`
4. Run the cells in sequence to generate plots, compute values, and display results. Each notebook includes inline comments and markdown explanations for reproducibility.

## Project Overview and Implementations

### Core Mathematical Visualizations and Computations

I started by exploring Lp norms through a visualization function that generates and plots unit norm balls in 2D. This involved creating points on a circle, computing their Lp norms using the formula `(|x|^p + |y|^p)^(1/p)`, and normalizing them to form the ball's boundary. I tested this for values like p=1 (resulting in a diamond shape), p=2 (a circle), and p=0.5 (a star-like, non-convex shape), confirming properties like convexity for p ≥ 1.

Next, I demonstrated the incompleteness of the space of continuous functions under the L1 norm. I defined a sequence of piecewise linear continuous functions that ramp up quickly and then plateau, converging to a discontinuous step function. Using numerical integration via `scipy.integrate.quad`, I calculated the L1 norm of the difference between these functions and their limit, showing it approaches zero, thus proving the space's incompleteness.

### Information Theory Implementations

A significant portion of the project focused on information theory. I implemented a Shannon entropy function that takes a probability mass function (PMF) and computes `-Σ p(x) * log2(p(x))`, with epsilon handling for zero probabilities to ensure stability.

For discrete distributions, I applied this to plot entropy for Bernoulli random variables across parameter p from 0 to 1, revealing a peak of 1 bit at p=0.5. I extended this to images by loading them in grayscale, computing normalized pixel intensity histograms as PMFs, and calculating their entropies, yielding values around 6.45 to 7.52 bits for the tested images.

I also handled joint and conditional entropies. For pairs of images, I created normalized joint histograms from flattened pixel arrays and used them to compute joint entropy H(X,Y) and conditional entropies H(X|Y) and H(Y|X), resulting in values like 12.82 bits for joint entropy in one pair.

Additionally, I implemented Kullback-Leibler (KL) divergence to measure differences between distributions. The function uses `Σ p(x) * log(p(x)/q(x))` with safeguards against division by zero. I applied it to image histograms (e.g., yielding 0.23 between two images) and discrete PMFs (e.g., 0.485 for specified distributions).

For a given joint PMF represented as a NumPy array, I derived marginal PMFs by axis summation and computed various measures: entropies H(X) and H(Y) at 0.918 bits each, conditional entropies at 0.667 bits, joint entropy at 1.585 bits, and mutual information at 0.251 bits.

In t-SNE, the crowding problem arises when embedding high-dimensional data into lower dimensions, causing points to cluster too tightly and obscure natural separations. The algorithm addresses this by using a Student's t-distribution for low-dimensional affinities, improving outlier handling and cluster visibility. Similarly, Jensen-Shannon divergence provides a symmetric measure of difference between distributions, enhancing comparisons in tasks like image entropy analysis.

### Dimensionality Reduction with t-SNE

In the advanced section, I implemented the t-SNE algorithm from scratch to embed high-dimensional data into 2D for visualization. Key steps included:
- Computing pairwise affinities in high-dimensional space using Gaussian kernels to form matrix P.
- Initializing random points in low-dimensional space.
- Calculating affinities in low-dimensional space with Student's t-distribution for matrix Q.
- Optimizing the embedding by minimizing the KL divergence between P and Q using gradient descent over multiple iterations.

I visualized the embedding's progression from a random scatter to clustered points and tracked the KL divergence reduction. For comparison, I ran scikit-learn's t-SNE with varying perplexity (5, 30, 50, 100) to observe how it influences cluster separation.

## Key Results and Insights

- Visualizations confirmed theoretical properties of norms and metric spaces, such as non-convexity of norm balls for p<1.
- Entropy calculations highlighted uncertainty in distributions and images, with peaks at balanced probabilities (e.g., entropy peaking at p=0.5 for Bernoulli).
- Divergence measures quantified differences between distributions effectively, with higher perplexity in t-SNE leading to smoother, more global cluster structures.
- The custom t-SNE implementation successfully reduced dimensions, matching professional library outputs while providing insight into the algorithm's mechanics.

The notebooks include all code, plots (e.g., norm balls, entropy curves, t-SNE embeddings), and printed results for easy verification. This project serves as a practical reference for these mathematical concepts in deep learning.

## Troubleshooting and Extensions

Common issues include handling zero probabilities in entropy calculations (use epsilon=1e-10 for stability) or adjusting t-SNE hyperparameters like learning rate eta or iterations for better convergence. For extensions, consider experimenting with additional datasets or integrating other divergences like Wasserstein distance.
