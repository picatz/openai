package embeddings

import (
	"errors"
	"fmt"
	"math"
	"sort"
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

// rank returns the ranks of the values in the given slice, used for Spearman's
// rank correlation coefficient.
func rank(data []float64) []float64 {
	ranks := make([]float64, len(data))
	for idx, value := range data {
		var rank float64
		var rankSum float64
		count := 0.0
		for otherIdx, otherValue := range data {
			if value < otherValue {
				rank++
			} else if value == otherValue {
				if idx != otherIdx {
					count++
					rankSum += float64(otherIdx)
				}
			}
		}
		ranks[idx] = rank + (rankSum+float64(idx))/float64(count+1)
	}
	return ranks
}

// avg returns the average of the values in the given slice.
func avg(data []float64) float64 {
	var sum float64
	for _, value := range data {
		sum += value
	}
	return sum / float64(len(data))
}

// SpearmanRankCorrelationCoefficient returns the Spearman rank correlation
// coefficient between two embeddings. This is a measure of the monotonic
// relationship between two embeddings (continuous variables), and it ranges
// from -1 (perfect negative correlation) to 1 (perfect positive correlation).
//
// You may notice it's very similar to Pearson's correlation coefficient, but
// it's calculated on the ranks of the values instead of the values themselves.
// This makes it more robust to outliers, but it's also more expensive to
// calculate.
//
// https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
func SpearmanRankCorrelationCoefficient(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("arrays must be of the same length")
	}

	ar := rank(a)
	br := rank(b)

	averageA := avg(ar)
	averageB := avg(br)

	var cov, varA, varB float64
	for i := 0; i < len(ar); i++ {
		dA := ar[i] - averageA
		dB := br[i] - averageB
		cov += dA * dB
		varA += dA * dA
		varB += dB * dB
	}

	return cov / math.Sqrt(varA*varB), nil
}

// HammingDistance calculates the Hamming distance between two embeddings.
// It calculates the sum of absolute differences between the two embeddings.
//
// This is primarily used for binary representations of embeddings. It is not
// very good for continuous embeddings like OpenAI's.
//
// https://en.wikipedia.org/wiki/Hamming_distance
func HammingDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("embeddings must have equal lengths")
	}

	var sum float64
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			sum++
		}
	}

	return sum, nil
}

// BrayCurtisDistance calculates the Bray-Curtis distance between two embeddings.
// It calculates the sum of absolute differences between the two embeddings,
// divided by the sum of their absolute values.
//
// https://en.wikipedia.org/wiki/Bray%E2%80%93Curtis_dissimilarity
func BrayCurtisDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("embeddings must have equal lengths")
	}

	var sumA, sumB float64
	for i := 0; i < len(a); i++ {
		sumA += a[i]
		sumB += b[i]
	}

	var sum float64
	for i := 0; i < len(a); i++ {
		sum += math.Abs(a[i]-b[i]) / (sumA + sumB)
	}

	return sum, nil
}

// MahalanobisDistance calculates the Mahalanobis distance between two embeddings.
// It calculates the distance between two embeddings, taking into account the
// variance of the data.
//
// The covariance matrix is a square matrix that describes the covariance
// between the embeddings. It must be the same size as the embeddings.
// https://en.wikipedia.org/wiki/Covariance_matrix
//
// # Example
//
//	covarianceMatrix := [][]float64{
//	    []float64{1, 0.5},
//	    []float64{0.5, 1},
//	}
//	distance, err := MahalanobisDistance([]float64{1, 2}, []float64{3, 4}, covarianceMatrix)
//
// https://en.wikipedia.org/wiki/Mahalanobis_distance
func MahalanobisDistance(a, b []float64, covarianceMatrix [][]float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("embeddings must have equal lengths")
	}

	if len(a) != len(covarianceMatrix) {
		return 0, errors.New("covariance matrix must be square and have the same dimensions as the embeddings")
	}

	var sum float64
	for i := 0; i < len(a); i++ {
		for j := 0; j < len(a); j++ {
			sum += (a[i] - b[i]) * (a[j] - b[j]) * covarianceMatrix[i][j]
		}
	}

	return math.Sqrt(sum), nil
}

// WordMoversDistance calculates the Word Mover's distance between two
// embeddings. It calculates the minimum cumulative distance that the words in
// one embedding need to travel to match the words in the other embedding.
//
// This is primarily used for text embeddings, but it can be used for any
// embedding where the distance between two embeddings is the distance between
// the words in the embeddings.
//
// https://en.wikipedia.org/wiki/Word_Mover%27s_Distance
func WordMoversDistance(a, b []float64, distanceFn func(a, b []float64) (float64, error)) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("embeddings must have equal lengths")
	}

	var sum float64
	for i := 0; i < len(a); i++ {
		min := math.Inf(1)
		for j := 0; j < len(b); j++ {
			d, err := distanceFn(a[i:], b[j:])
			if err != nil {
				return 0, err
			}
			if d < min {
				min = d
			}
		}
		sum += min
	}

	return sum, nil
}

// BhattacharyyaDistance calculates the Bhattacharyya distance between two
// embeddings. This similarity measure is used to compare two probability
// distributions represented by continuous embeddings. Lower values indicate
// higher similarity between the distributions.
//
// Ends up not being very useful for OpenAI's embeddings.
//
// https://en.wikipedia.org/wiki/Bhattacharyya_distance
func BhattacharyyaDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("embeddings must have equal lengths")
	}

	var sum float64
	for i := 0; i < len(a); i++ {
		sum += math.Sqrt(a[i] * b[i])
	}

	return -math.Log(sum), nil
}

// WassersteinDistance calculates the Wasserstein distance between two
// embeddings. Also known as the Earth Mover's distance, this measures
// the minimum amount of work required to transform one distribution
// into another, considering the distance between each data point. It
// can be used to compare probability distributions represented by
// continuous embeddings.
//
// https://en.wikipedia.org/wiki/Wasserstein_metric
func WassersteinDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("embeddings must have equal lengths")
	}

	// Avoid modifying the original embeddings, since we need to sort them.
	sortedA := make([]float64, len(a))
	sortedB := make([]float64, len(b))

	copy(sortedA, a)
	copy(sortedB, b)

	sort.Float64s(sortedA)
	sort.Float64s(sortedB)

	distance := 0.0
	for i := range sortedA {
		distance += math.Abs(sortedA[i] - sortedB[i])
	}

	return distance, nil
}

// Softmax calculates the softmax of a slice of floats. This is used to
// convert a vector of numbers into a probability distribution.
func Softmax(x []float64) []float64 {
	var sum float64
	soft := make([]float64, len(x))
	for _, v := range x {
		sum += math.Exp(v)
	}
	for i, v := range x {
		soft[i] = math.Exp(v) / sum
	}
	return soft
}

// KullbackLeiblerDivergence calculates the Kullback-Leibler divergence between two
// embeddings. This measures the difference between two probability distributions
// represented by continuous embeddings. Lower values indicate higher similarity
// between the distributions.
//
// Ends up not being very useful for OpenAI's embeddings.
//
// https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
func KullbackLeiblerDivergence(p, q []float64) (float64, error) {
	if len(p) != len(q) {
		return 0, errors.New("distributions must have the same dimensions")
	}

	var divergence float64 = 0
	for i := 0; i < len(p); i++ {
		if p[i] > 0 {
			if q[i] > 0 {
				divergence += p[i] * (math.Log2(p[i]) - math.Log2(q[i]))
			} else {
				// If q[i] is 0, then the KL divergence is undefined.
				return 0, errors.New("elements in q distribution should be positive")
			}
		}
	}

	return divergence, nil
}

// ShannonEntropy calculates the entropy of a probability distribution represented by
// a continuous embedding.
//
// https://en.wikipedia.org/wiki/Entropy_(information_theory)
func ShannonEntropy(p []float64) float64 {
	var ent float64 = 0
	for i := 0; i < len(p); i++ {
		if p[i] > 0 {
			ent += p[i] * math.Log2(p[i])
		}
	}
	return -ent
}

// JensenShannonDivergence calculates the Jensen-Shannon divergence between two
// embeddings. This measures the difference between two probability distributions
// represented by continuous embeddings. Lower values indicate higher similarity
// between the distributions.
//
// Ends up not being very useful for OpenAI's embeddings.
//
// https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
func JensenShannonDivergence(p, q []float64) (float64, error) {
	if len(p) != len(q) {
		return 0, errors.New("distributions must have the same dimensions")
	}

	// Handle negative values in the distributions
	// with a softmax, which will normalize
	pSoft := Softmax(p)
	qSoft := Softmax(q)

	m := make([]float64, len(pSoft))
	for i := range pSoft {
		m[i] = 0.5 * (pSoft[i] + qSoft[i])
	}

	dPm, err := KullbackLeiblerDivergence(p, m)
	if err != nil {
		return 0, err
	}

	dQm, err := KullbackLeiblerDivergence(q, m)
	if err != nil {
		return 0, err
	}

	return 0.5 * (dPm + dQm), nil
}

// AngularDistance calculates the angular distance between two embeddings.
// This is a measure of the angle between two vectors, and is useful for
// comparing embeddings that represent directions.
//
// https://en.wikipedia.org/wiki/Angular_distance
func AngularDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("embeddings must have equal lengths")
	}

	var dot, normA, normB float64
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	return math.Acos(dot / (math.Sqrt(normA) * math.Sqrt(normB))), nil
}

// CorrelationDistance calculates the correlation distance between two embeddings.
// This is a measure of the correlation between two vectors, and is useful for
// comparing embeddings that represent directions.
//
// https://en.wikipedia.org/wiki/Correlation_distance
func CorrelationDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("embeddings must have equal lengths")
	}

	var dot, normA, normB float64
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	return 1 - (dot / (math.Sqrt(normA) * math.Sqrt(normB))), nil
}

// PairwiseDistance calculates the pairwise distance between two embeddings.
// This is a measure of the distance between two vectors, and is useful for
// comparing embeddings that represent directions.
func PairwiseDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("embeddings must have equal lengths")
	}

	var distance float64
	for i := 0; i < len(a); i++ {
		distance += math.Pow(a[i]-b[i], 2)
	}

	return math.Sqrt(distance), nil
}

// HellingerDistance calculates the Hellinger distance between two embeddings.
// This is a measure of the distance between two probability distributions
// represented by continuous embeddings. Lower values indicate higher similarity
// between the distributions.
//
// https://en.wikipedia.org/wiki/Hellinger_distance
func HellingerDistance(p, q []float64) (float64, error) {
	if len(p) != len(q) {
		return 0, errors.New("distributions must have the same dimensions")
	}

	var distance float64
	for i := 0; i < len(p); i++ {
		distance += math.Pow(math.Sqrt(p[i])-math.Sqrt(q[i]), 2)
	}

	return math.Sqrt(distance) / math.Sqrt2, nil
}

// TanimotoDistance calculates the Tanimoto distance between two embeddings.
//
// https://en.wikipedia.org/wiki/Jaccard_index
func TanimotoDistance(a, b []float64) (float64, error) {
	return JaquardSimilarity(a, b)
}

// ChebyshevDistance calculates the Chebyshev distance between two embeddings.
// It is a metric defined on a vector space where the distance between two
// vectors is the greatest of their differences along any coordinate dimension.
//
// https://en.wikipedia.org/wiki/Chebyshev_distance
func ChebyshevDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("embeddings must have equal lengths")
	}

	var distance float64
	for i := 0; i < len(a); i++ {
		distance = math.Max(distance, math.Abs(a[i]-b[i]))
	}

	return distance, nil
}

// RuzickaDistance calculates the Ruzicka distance (or weighted Jaccard distance)
// between two embeddings. It is a metric defined on a vector space where the
// distance between two vectors is the sum of the minimum of their values along
// any coordinate dimension.
//
// https://en.wikipedia.org/wiki/Jaccard_index
func RuzickaDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("embeddings must have equal lengths")
	}

	var distance float64
	for i := 0; i < len(a); i++ {
		distance += math.Min(a[i], b[i])
	}

	return distance, nil
}

// Sum returns the sum of elements in a given slice of float64
func Sum(slice []float64) float64 {
	sum := 0.0
	for _, value := range slice {
		sum += value
	}
	return sum
}

// WaveHedgesDistance compares two embeddings and returns the distance between them.
func WaveHedgesDistance(a, b []float64) (float64, error) {
	// Check if the lengths of the input vectors are equal
	if len(a) != len(b) {
		return 0, fmt.Errorf("input vectors should have the same length")
	}

	distance := 0.0
	for i := 0; i < len(a); i++ {
		// Calculate the difference between the two vector components
		diff := a[i] - b[i]

		// Calculate the absolute value of the difference and add
		// it to the overall distance.
		distance += math.Abs(diff)
	}

	// Calculate the Wave Hedges Distance by dividing the overall distance by
	// the sum of components in the input vectors
	waveHedgesDistance := distance / Sum(a)

	return waveHedgesDistance, nil
}

// ClarkDistance compares two embeddings and returns the distance between them.
func ClarkDistance(a, b []float64) (float64, error) {
	// Check if the lengths of the input vectors are equal
	if len(a) != len(b) {
		return 0, fmt.Errorf("input vectors should have the same length")
	}

	distance := 0.0
	for i := 0; i < len(a); i++ {
		// Calculate the difference between the two vector components
		diff := a[i] - b[i]

		// Calculate the absolute value of the difference and add
		// it to the overall distance.
		distance += math.Abs(diff)
	}

	// Calculate the Clark Distance by dividing the overall distance by
	// the sum of components in the input vectors
	clarkDistance := distance / (2 * Sum(a))

	return clarkDistance, nil
}

// MotykaSimpsonDistance compares two embeddings and returns the distance between them.
//
// https://en.wikipedia.org/wiki/Motyka_distance
func MotykaSimpsonDistance(a, b []float64) (float64, error) {
	// Check if the lengths of the input vectors are equal
	if len(a) != len(b) {
		return 0, fmt.Errorf("input vectors should have the same length")
	}

	distance := 0.0
	for i := 0; i < len(a); i++ {
		// Calculate the difference between the two vector components
		diff := a[i] - b[i]

		// Calculate the absolute value of the difference and add
		// it to the overall distance.
		distance += math.Abs(diff)
	}

	// Calculate the Motyka Simpson Distance by dividing the overall distance by
	// the sum of components in the input vectors
	motykaSimpsonDistance := distance / (Sum(a) + Sum(b))

	return motykaSimpsonDistance, nil
}

