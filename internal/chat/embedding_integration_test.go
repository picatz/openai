//go:build integration

package chat_test

import (
	"fmt"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/lipgloss/table"
	"github.com/openai/openai-go/v3"
	"github.com/picatz/openai/internal/chat"
	"github.com/shoenig/test/must"
	"math"
	"os"
	"testing"
)

func TestChunkString_consign_similarity(t *testing.T) {
	if os.Getenv("OPENAI_LIVE_TESTS") != "1" {
		t.Skip("requires -tags=integration and OPENAI_LIVE_TESTS=1; may incur API charges")
	}
	var (
		input     = "I like red cats and blue dogs. Red cats are my favorite."
		chunkSize = int64(6)
		// chunkSize = int64(5)
	)

	chunks, err := chat.ChunkString(input, chunkSize)
	must.NoError(t, err)

	// expectedChunks := []string{
	// 	"I like red cats",
	// 	"and blue dogs.",
	// 	"Red cats are my",
	// 	"favorite.",
	// }

	// must.Eq(t, expectedChunks, chunks)

	client := openai.NewClient()

	type cosinePair struct {
		A          string
		B          string
		Similarity float64
	}

	getPair := func(a, b string) cosinePair {
		aEmbedding, err := client.Embeddings.New(t.Context(), openai.EmbeddingNewParams{
			Model: openai.EmbeddingModelTextEmbedding3Small,
			Input: openai.EmbeddingNewParamsInputUnion{
				OfString: openai.String(a),
			},
		})
		must.NoError(t, err)

		bEmbedding, err := client.Embeddings.New(t.Context(), openai.EmbeddingNewParams{
			Model: openai.EmbeddingModelTextEmbedding3Small,
			Input: openai.EmbeddingNewParamsInputUnion{
				OfString: openai.String(b),
			},
		})
		must.NoError(t, err)

		return cosinePair{
			A: a,
			B: b,
			Similarity: cosignSimilarity(
				aEmbedding.Data[0].Embedding,
				bEmbedding.Data[0].Embedding,
			),
		}
	}

	var cosinePairs []cosinePair

	for i := range chunks {
		for j := i + 1; j < len(chunks); j++ {
			cosinePairs = append(cosinePairs, getPair(chunks[i], chunks[j]))
		}
	}

	for i := range chunks {
		cosinePairs = append(cosinePairs, getPair(chunks[i], chunks[i]))
	}

	for i := range chunks {
		cosinePairs = append(cosinePairs, getPair(chunks[i], "Red cats"))
	}

	for i := range chunks {
		cosinePairs = append(cosinePairs, getPair("Red cats", chunks[i]))
	}

	for i := range chunks {
		cosinePairs = append(cosinePairs, getPair("Blue dogs", chunks[i]))
	}

	rows := make([][]string, 0, len(cosinePairs))
	for _, pair := range cosinePairs {
		rows = append(rows, []string{
			pair.A + "     ",
			pair.B + "     ",
			fmt.Sprintf("%.4f", pair.Similarity),
		})
	}

	tbl := table.New().
		Border(lipgloss.RoundedBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(lipgloss.Color("245"))).
		Headers("A", "B", "Similarity").
		Rows(rows...)

	fmt.Println(tbl.Render())
}

func cosignSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
