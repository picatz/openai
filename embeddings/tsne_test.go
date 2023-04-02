package embeddings

import (
	"context"
	"image/png"
	"os"
	"testing"
	"time"

	"github.com/picatz/openai"
)

func TestTSNEVisualizePNG(t *testing.T) {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	getEmbedding := func(t *testing.T, input string) []float64 {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		resp, err := client.CreateEmbedding(ctx, &openai.CreateEmbeddingRequest{
			Model: openai.ModelTextEmbeddingAda002,
			Input: input,
		})
		if err != nil {
			t.Fatalf("failed to create embedding: %v", err)
		}

		return resp.Data[0].Embedding
	}

	// Note: this is a very small sample size and the results are not
	//       representative of the model's performance. This is just a
	//       simple test to make sure the code works, and it draws
	//       something that looks kind'a cool.

	// war := getEmbedding(t, "WAR")
	// peace := getEmbedding(t, "PEACE")
	// dogs := getEmbedding(t, "DOGS")
	cats := getEmbedding(t, "CATS")
	// superheroes := getEmbedding(t, "SUPERHEROES")
	// villains := getEmbedding(t, "VILLAINS")

	embeddings := [][]float64{}

	for i := 0; i < 5; i++ {
		// embeddings = append(embeddings, war)
		// embeddings = append(embeddings, peace)
		// embeddings = append(embeddings, dogs)
		embeddings = append(embeddings, cats)
		// embeddings = append(embeddings, superheroes)
		// embeddings = append(embeddings, villains)
	}

	var (
		perplexity   float64 = 300
		learningRate float64 = 0.5
		nIter        int     = 5
		outputDims   int     = 10

		// perplexity   float64 = 300
		// learningRate float64 = 0.5
		// nIter        int     = 2
		// outputDims   int     = 2

		// perplexity   float64 = 120
		// learningRate float64 = 0.2
		// nIter        int     = 2
		// outputDims   int     = 2

		// perplexity   float64 = 300
		// learningRate float64 = 100
		// nIter        int     = 2
		// outputDims   int     = 2
	)

	tSNEembeddings := TSNE(embeddings, perplexity, learningRate, nIter, outputDims)

	img, err := Visualize(tSNEembeddings, 5, 512, 512)
	if err != nil {
		t.Fatalf("failed to visualize embeddings: %v", err)
	}

	// Delete the file if it already exists
	if _, err := os.Stat("tsne-embeddings.png"); err == nil {
		if err := os.Remove("tsne-embeddings.png"); err != nil {
			t.Fatalf("failed to remove file: %v", err)
		}
	}

	// Save the image to a file.
	fh, err := os.Create("tsne-embeddings.png")
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	defer fh.Close()

	// Encode the image as a PNG.
	if err := png.Encode(fh, img); err != nil {
		t.Fatalf("failed to encode image: %v", err)
	}
}
