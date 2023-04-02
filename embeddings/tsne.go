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
func TSNE(data [][]float64, perp float64, iter int, dim int) [][]float64 {
	dataSize := len(data)
	solution := make([][]float64, dataSize)
	for i := range solution {
		solution[i] = make([]float64, dim)
	}
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < dataSize; i++ {
		for j := 0; j < dim; j++ {
			solution[i][j] = rand.Float64()
		}
	}

	P := computeJointProbability(data, perp)

	for t := 1; t <= iter; t++ {
		var grad [][]float64
		for i := 0; i < dataSize; i++ {

			gradY := make([]float64, dim)

			for k := 0; k < dim; k++ {
				var sumQi float64
				for j := 0; j < dataSize; j++ {
					if i == j {
						continue
					}
					Qij := 1 / (1 + squaredEuclidean(data[i], data[j]))
					sumQi += Qij
				}

				for j := 0; j < dataSize; j++ {
					if i == j {
						continue
					}
					f := ((P[i][j] - sumQi) * squaredEuclidean(data[i], data[j]))
					gradY[k] += f * (solution[i][k] - solution[j][k])
				}
			}

			grad = append(grad, gradY)
		}

		learningRate := 500 / float64(t)
		for i := 0; i < dataSize; i++ {
			for k := 0; k < dim; k++ {
				solution[i][k] -= learningRate * grad[i][k]
			}
		}
	}

	return solution
}
