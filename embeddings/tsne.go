package embeddings

import (
	"math"
	"math/rand"
	"time"
)

// This is an attempt to implement t-SNE in Go. It is not complete and
// is not really used in the project. It is here for reference only.

func pairwiseEuclideanDistances(embeddings [][]float64) [][]float64 {
	n := len(embeddings)
	distances := make([][]float64, n)
	for i := range distances {
		distances[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			d := 0.0
			for k, x := range embeddings[i] {
				y := embeddings[j][k]
				d += math.Pow(x-y, 2)
			}
			distances[i][j] = d
			distances[j][i] = d
		}
	}
	return distances
}

// GaussianKernel returns a matrix of similarities using the Gaussian
// kernel. The sigma parameter controls the width of the kernel.
//
// This is used for the t-SNE algorithm.
func GaussianKernel(distances [][]float64, sigma float64) [][]float64 {
	n := len(distances)
	gaussian := make([][]float64, n)

	for i := range gaussian {
		gaussian[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			g := math.Exp(-distances[i][j] / (2 * math.Pow(sigma, 2)))
			gaussian[i][j] = g
			gaussian[j][i] = g
		}
	}
	return gaussian
}

// LowDimensionalEmbeddings returns a matrix of random embeddings.
//
// This is used for the t-SNE algorithm.
func LowDimensionalEmbeddings(n int, outputDims int) [][]float64 {
	embeddings := make([][]float64, n)
	for i := range embeddings {
		embeddings[i] = make([]float64, outputDims)
		for j := range embeddings[i] {
			embeddings[i][j] = rand.Float64() * 0.001
		}
	}
	return embeddings
}

// GradientDescent updates the embeddings using the given matrix of
// similarities and the given learning rate.
//
// This is used for the t-SNE algorithm.
func GradientDescent(m [][]float64, y [][]float64, alpha float64) [][]float64 {
	// fmt.Printf("GradientDescent: alpha=%f\n", alpha)
	n := len(y)
	// fmt.Printf("GradientDescent: n=%d\n", n)
	for i := 0; i < n; i++ {
		// fmt.Printf("GradientDescent: i=%d\n", i)
		for k := range y[i] {
			// fmt.Printf("GradientDescent: k=%d\n", k)
			// Compute gradient
			grad := 0.0
			for j := 0; j < n; j++ {
				if i != j {
					// fmt.Printf("GradientDescent: j=%d grad=%3.f\n", j, grad)
					grad += m[i][j] * (y[i][k] - y[j][k])
				}
			}
			// Update embeddings
			y[i][k] += -alpha * grad

		}
	}
	return y
}

// TSNE performs t-SNE on the given embeddings, using the given perplexity,
// learning rate, number of iterations, and output dimensions. It assumes
// the embeddings are already scaled and normalized.
//
// This process is described in detail in the paper "Learning a Parametric
// Embedding by Preserving Local Structure" by Laurens van der Maaten.
//
// In short: t-distributed stochastic neighbor embedding (t-SNE) is a
// statistical method for visualizing high-dimensional data by giving
// each datapoint a location in a two or three-dimensional map. It
// is based on Stochastic Neighbor Embedding originally developed by
// Sam Roweis and Geoffrey Hinton, where Laurens van der Maaten
// proposed the t-distributed variant.
//
// # Steps
//
//  1. Data preprocessing: Scale and normalize the input dataset (typically
//     using standard scaler or min-max scaler). This step ensures that the features
//     have the same scale, which makes it easier for t-SNE to obtain meaningful
//     distances among data points.
//
//  2. Compute pairwise similarities: Calculate the pairwise similarities
//     between the high-dimensional input data points in the input space. This is
//     done using the Gaussian kernel. For each point i, a conditional probability
//     P(i|j) is computed, representing the pairwise similarities in the high-
//     dimensional space.
//
//  3. Initialize the low-dimensional embeddings: Randomly initialize the lower-
//     dimensional embedding (usually 2D or 3D) for each data point in the dataset.
//
//  4. Compute pairwise similarities in low-dimensional space: Calculate the
//     pairwise similarities between the lower-dimensional embeddings. This is
//     achieved by computing the t-Student distribution for every pair of points.
//
//  5. Calculate the gradient: Compute the gradient of the Kullback-Leibler (KL)
//     divergence between the high-dimensional and the low-dimensional pairwise
//     similarities. Minimizing the KL divergence ensures that the similarities are
//     preserved as much as possible during the dimensionality reduction process.
//
//  6. Update the embedded points: Use gradient descent (or another optimization
//     algorithm) to iteratively update the lower-dimensional embeddings by
//     minimizing the KL divergence. The learning rate and the number of iterations
//     can be tuned for better results.
//
//  7. Repeat steps 4 to 6 until convergence: Continue updating the lower-
//     dimensional embeddings until it converges or reaches the maximum number of
//     iterations.
//
//     Once the algorithm converges, we will have the final lower-dimensional
//     embeddings of the dataset, allowing for visualization and exploration.
//
// https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
func TSNE(embeddings [][]float64, perplexity float64, learningRate float64, nIter int, outputDims int) [][]float64 {
	rand.Seed(time.Now().UnixNano())

	// Compute pairwise distances (euclidean)
	distances := pairwiseEuclideanDistances(embeddings)

	// Compute Gaussian kernel
	sigma := math.Sqrt(perplexity)
	p := GaussianKernel(distances, sigma)

	// Initialize low-dimensional embeddings
	y := LowDimensionalEmbeddings(len(embeddings), outputDims)

	// Optimize embeddings
	alpha := learningRate
	for iter := 0; iter < nIter; iter++ {
		y = GradientDescent(p, y, alpha)
	}

	return y
}
