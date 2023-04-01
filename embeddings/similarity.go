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

// PearsonCorrelationCoefficient returns the Pearson correlation coefficient
// between two embeddings. This is a measure of the linear correlation between
// two embeddings (continuous variables), and it ranges from -1 (perfect negative
// correlation) to 1 (perfect positive correlation). It can be used to compare
// embeddings by quantifying the linear relationship between them.
//
// https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
func PearsonCorrelationCoefficient(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("embeddings must have equal lengths")
	}

	var sumA, sumB, sumASquared, sumBSquared, sumAB float64
	for i := 0; i < len(a); i++ {
		sumA += a[i]
		sumB += b[i]
		sumASquared += a[i] * a[i]
		sumBSquared += b[i] * b[i]
		sumAB += a[i] * b[i]
	}

	numerator := sumAB - (sumA * sumB / float64(len(a)))
	denominator := math.Sqrt((sumASquared - (sumA * sumA / float64(len(a)))) * (sumBSquared - (sumB * sumB / float64(len(a)))))

	return numerator / denominator, nil
}

// ManhattanDistance calculates the Manhattan distance between two embeddings.
// Also known as the L1 distance, it measures the distance between two points
// by summing the absolute differences between their corresponding coordinates.
//
// In other words, it measures the total "city block" distance traveled between
// the two points.
//
// https://en.wikipedia.org/wiki/Taxicab_geometry
func ManhattanDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("embeddings must have equal lengths")
	}

	var sum float64
	for i := 0; i < len(a); i++ {
		sum += math.Abs(a[i] - b[i])
	}

	return sum, nil
}

// JaquardSimilarity calculates the Jaquard similarity between two embeddings.
// This is a similarity measure for sets, defined as thee size of the intersection
// divided by the size of the union of the two sets.
//
// It can be used for binary representations of embeddings, but it's not very
// good for continuous embeddings like OpenAI's embeddings.
//
// https://en.wikipedia.org/wiki/Jaccard_index
func JaquardSimilarity(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("embeddings must have equal lengths")
	}

	var intersection, union float64
	for i := 0; i < len(a); i++ {
		if a[i] == b[i] {
			intersection++
		}
		if a[i] != 0 || b[i] != 0 {
			union++
		}
	}

	return intersection / union, nil
}
