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
	//       representative of the model's performance.

	a := getEmbedding(t, "WHALES")
	b := getEmbedding(t, "DOLPHINS")
	c := getEmbedding(t, "BIRDS")

	embeddings := [][]float64{
		a, // red
		b, // green
		c, // blue
	}

	// These values were chosen after some experimentation, which
	// is required to get good results depending on the data.
	var (
		perplexity       float64 = 200
		nIter            int     = 20
		outputDimensions int     = 2 // 2D or 3D space (2 or 3)
	)

	tSNEDimensions := TSNE(embeddings, perplexity, nIter, outputDimensions)

	for i, dimension := range tSNEDimensions {
		t.Logf("embedding %d dimensions: %-2.f, %-2.f", i, dimension[0], dimension[1])
	}

	// We should see close red and green dots, and further away blue dots.
	img, err := Visualize(tSNEDimensions, 5, 800, 800)
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
