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

	shortestLength := min(len(a), len(b))

	for i := 0; i < shortestLength; i++ {
		value1 := a[i]
		value2 := b[i]
		dotProduct += value1 * value2
		magnitude1 += value1 * value1
		magnitude2 += value2 * value2
	}

	if magnitude1 == 0.0 || magnitude2 == 0.0 {
		return 0.0, errors.New("at least one of the embedding magnitudes is zero")
	}

	return dotProduct / (math.Sqrt(magnitude1) * math.Sqrt(magnitude2)), nil
}

// min returns the smaller of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
