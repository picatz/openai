package chat_test

import (
	"bytes"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/lipgloss/table"
	"github.com/cockroachdb/pebble"
	"github.com/cockroachdb/pebble/vfs"
	"github.com/openai/openai-go"
	"github.com/picatz/openai/internal/chat"
	"github.com/picatz/openai/internal/chat/storage"
	pebbleStorage "github.com/picatz/openai/internal/chat/storage/pebble"
	"github.com/shoenig/test/must"
)

func TestChatSession(t *testing.T) {
	var (
		client = openai.NewClient()
		input  = bytes.NewBuffer(nil)
		output = bytes.NewBuffer(nil)
	)

	typeInTerminal := func(s string) {
		for line := range strings.Lines(s) {
			n, err := input.WriteString(line + "\r\n")
			must.NoError(t, err)
			must.Eq(t, len(line)+2, n)
		}
	}

	pebbleOptions := &pebble.Options{
		FS: vfs.NewMem(),
	}

	codec := &storage.JSONCodec[string, chat.ReqRespPair]{}

	memBackend, err := pebbleStorage.NewBackend("", pebbleOptions, codec)
	must.NoError(t, err)
	must.NotNil(t, memBackend)
	t.Cleanup(func() {
		must.NoError(t, memBackend.Close(t.Context()))
	})

	chatSession, restore, err := chat.NewSession(t.Context(), client, openai.ChatModelGPT4o, input, output, memBackend)
	must.NoError(t, err)
	t.Cleanup(restore)
	must.NotNil(t, chatSession)

	typeInTerminal("hello")

	done, err := chatSession.RunOnce(t.Context())
	must.NoError(t, err)
	must.False(t, done)

	t.Log(output.String())
}

func TestChunkString(t *testing.T) {
	var (
		input     = "This is a test string that is longer than the chunk size."
		chunkSize = int64(8)
	)

	chunks, err := chat.ChunkString(input, chunkSize)
	must.NoError(t, err)

	expectedChunks := []string{
		"This is a test string",
		"that is longer than",
		"the chunk size.",
	}

	must.Eq(t, expectedChunks, chunks)
}

func TestChunkString_consign_similarity(t *testing.T) {
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
			Model: openai.F(openai.EmbeddingModelTextEmbedding3Small),
			Input: openai.F(
				openai.EmbeddingNewParamsInputUnion(
					openai.EmbeddingNewParamsInputArrayOfStrings{
						a,
					},
				),
			),
		})
		must.NoError(t, err)

		bEmbedding, err := client.Embeddings.New(t.Context(), openai.EmbeddingNewParams{
			Model: openai.F(openai.EmbeddingModelTextEmbedding3Small),
			Input: openai.F(
				openai.EmbeddingNewParamsInputUnion(
					openai.EmbeddingNewParamsInputArrayOfStrings{
						b,
					},
				),
			),
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
