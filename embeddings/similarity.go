package embeddings

import (
	"errors"
	"math"
)

// CosineSimilarity calculates the cosine similarity between two embeddings.
//
// https://en.wikipedia.org/wiki/Cosine_similarity
func CosineSimilarity(a, b []float64) (float64, error) {
	var dotProduct, magnitude1, magnitude2 float64

	if len(a) == 0 || len(b) == 0 {
		return 0, errors.New("at least one of the embeddings is empty")
	}

	if len(a) != len(b) {
		return 0, errors.New("embeddings must have equal lengths")
	}

	for i := 0; i < len(a); i++ {
		value1 := a[i]
		value2 := b[i]
		dotProduct += value1 * value2
		magnitude1 += value1 * value1
		magnitude2 += value2 * value2
	}

	if magnitude1 == 0.0 || magnitude2 == 0.0 {
		return 0, errors.New("at least one of the embedding magnitudes is zero")
	}

	return dotProduct / (math.Sqrt(magnitude1) * math.Sqrt(magnitude2)), nil
}

// EuclideanDistance calculates the Euclidean distance between two embeddings.
//
// It calculates the sum of squared differences between the two embeddings, then
// takes the square root of that sum to get the distance.
//
// https://en.wikipedia.org/wiki/Euclidean_distance
func EuclideanDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("embeddings must have equal lengths")
	}

	var sumSquares float64
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sumSquares += diff * diff
	}

	distance := math.Sqrt(sumSquares)
	return distance, nil
}
