package embeddings

import (
	"math"
	"math/rand"
	"time"
)

// This is an attempt to implement t-SNE in Go. It is not complete and
// is not really used in the project. It is here for reference only.

func squaredEuclidean(x, y []float64) float64 {
	var result float64
	for i := 0; i < len(x); i++ {
		d := x[i] - y[i]
		result += d * d
	}
	return result
}

func computePerplexity(data [][]float64, distance []float64, perp float64, j int) float64 {
	beta := 1.0
	logU := math.Log(perp)
	var sumP, H, logSumP float64

	dataSize := len(data)
	for i := 0; i < dataSize; i++ {
		if i == j {
			continue
		}

		Pij := math.Exp(-distance[i] * beta)
		sumP += Pij
		logSumP = math.Log(sumP)
		H += beta * distance[i] * Pij
	}

	H = (H / sumP) + logSumP
	diff := H - logU

	return diff
}

func computeJointProbability(data [][]float64, perp float64) [][]float64 {
	dataSize := len(data)
	distance := make([]float64, dataSize)
	P := make([][]float64, dataSize)
	for i := range P {
		P[i] = make([]float64, dataSize)
	}

	for i := 0; i < dataSize; i++ {
		for j := i + 1; j < dataSize; j++ {
			distance[j] = squaredEuclidean(data[i], data[j])
		}
		beta := 1.0
		lower := 0.0
		upper := 1e15
		for k := 0; k < 50; k++ {
			diff := computePerplexity(data, distance, perp, i)
			if math.Abs(diff) < 1e-5 {
				break
			}

			if diff > 0 {
				lower = beta
				beta = (beta + upper) / 2
			} else {
				upper = beta
				beta = (beta + lower) / 2
			}
		}

		var rowSum float64
		for j := 0; j < dataSize; j++ {
			if i == j {
				continue
			}
			Pij := math.Exp(-distance[j] * beta)
			P[i][j] = Pij
			rowSum += Pij
		}

		if rowSum == 0 {
			rowSum = 1
		}

		for j := 0; j < dataSize; j++ {
			if i == j {
				continue
			}
			P[i][j] /= rowSum
		}
	}

	return P
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
// https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
func TSNE(embeddings [][]float64, perp float64, iter int, dim int) [][]float64 {
	// Seed the random number generator with the current time.
	//
	// This is useful for t-SNE because it uses random numbers to
	// initialize the solution vector.
	rand.Seed(time.Now().UnixNano())

	// Initialize the solution vector, and get embeddings size (outer dimension).
	var (
		embeddingsSize = len(embeddings)
		solution       = make([][]float64, embeddingsSize)
	)

	// Initialize the solution vector of the given dimension.
	for i := range solution {
		solution[i] = make([]float64, dim)
	}

	// Initialize the solution vector with random numbers.
	for i := 0; i < embeddingsSize; i++ {
		for j := 0; j < dim; j++ {
			solution[i][j] = rand.Float64()
		}
	}

	// Compute the joint probability distribution of the given embeddings.
	P := computeJointProbability(embeddings, perp)

	// Run the optimization loop for the given number of iterations.
	//
	// This performs gradient descent on the solution vector, using
	// the joint probability distribution as the target.
	for t := 1; t <= iter; t++ {
		// Compute the gradient of the solution vector.
		var grad [][]float64

		// Compute the gradient for each embedding.
		for i := 0; i < embeddingsSize; i++ {
			// Compute the gradient for each dimension.
			gradY := make([]float64, dim)

			// Compute the sum of the joint probability distribution.
			for k := 0; k < dim; k++ {
				// The sum of the joint probability distribution (Q) is
				// computed by summing the squared Euclidean distance
				// between each embedding.
				var sumQi float64

				// Compute the sum of the joint probability distribution.
				for j := 0; j < embeddingsSize; j++ {
					// Skip the current embedding.
					if i == j {
						continue
					}

					// Compute the squared Euclidean distance between the embeddings.
					Qij := 1 / (1 + squaredEuclidean(embeddings[i], embeddings[j]))

					// Add the squared Euclidean distance to the sum.
					sumQi += Qij
				}

				// Compute the gradient for the given dimension.
				for j := 0; j < embeddingsSize; j++ {
					// Skip the current embedding.
					if i == j {
						continue
					}

					// Compute the gradient for the given dimension using the
					// joint probability distribution (P) and the sum of the
					// joint probability distribution (Q) which is computed
					// above.
					f := ((P[i][j] - sumQi) * squaredEuclidean(embeddings[i], embeddings[j]))

					// Add the gradient to the gradient vector.
					gradY[k] += f * (solution[i][k] - solution[j][k])
				}
			}

			// Append the computed gradient.
			grad = append(grad, gradY)
		}

		// Update the solution vector, using the gradient at the given learning rate.
		//
		// The learning rate is decreased over time, to prevent the solution vector
		// from oscillating too much, allowing it to settle into a stable state.
		learningRate := 500 / float64(t)
		for i := 0; i < embeddingsSize; i++ {
			for k := 0; k < dim; k++ {
				solution[i][k] -= learningRate * grad[i][k]
			}
		}
	}

	return solution
}
