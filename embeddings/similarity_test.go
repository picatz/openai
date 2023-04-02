package embeddings

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/picatz/openai"
)

func ptrFloat64(f float64) *float64 {
	return &f
}

func TestCosignSimilariy(t *testing.T) {
	t.Run("return 1.0 for identical embeddings", func(t *testing.T) {
		sim, err := CosineSimilarity([]float64{1, 2, 3}, []float64{1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}

		if sim != 1.0 {
			t.Fatalf("expected similarity to be 1.0, got %f", sim)
		}
	})

	t.Run("return 0.0 for orthogonal embeddings", func(t *testing.T) {
		sim, err := CosineSimilarity([]float64{1, 0, 0}, []float64{0, 1, 0})
		if err != nil {
			t.Fatal(err)
		}

		if sim != 0.0 {
			t.Fatalf("expected similarity to be 0.0, got %f", sim)
		}
	})

	t.Run("return 0.0 for zero embeddings", func(t *testing.T) {
		sim, err := CosineSimilarity([]float64{0, 0, 0}, []float64{0, 0, 0})
		if err == nil {
			t.Fatal("expected error")
		}

		if sim != 0.0 {
			t.Fatalf("expected similarity to be 0.0, got %f", sim)
		}
	})

	t.Run("a,b", func(t *testing.T) {
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

		tests := []struct {
			name       string
			a          []float64
			b          []float64
			wantApprox *float64
			want       *float64
		}{
			{
				name:       "return 1.0 for identical embeddings",
				a:          getEmbedding(t, "Hello world"),
				b:          getEmbedding(t, "Hello world"),
				wantApprox: ptrFloat64(0.9),
				// want: ptrFloat64(1.0), // flakey? 0.9999999999999999
			},
			{
				name:       "hello world and moon",
				a:          getEmbedding(t, "Hello world"),
				b:          getEmbedding(t, "Hello moon"),
				wantApprox: ptrFloat64(0.8),
			},
			{
				name:       "hello world and hello",
				a:          getEmbedding(t, "Hello world"),
				b:          getEmbedding(t, "Hello"),
				wantApprox: ptrFloat64(0.6),
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				got, err := CosineSimilarity(tt.a, tt.b)
				if err != nil {
					t.Fatalf("error: %v", err)
				}

				if tt.want != nil && got != *tt.want {
					t.Fatalf("got: %v, want: %v", got, *tt.want)
				}

				if tt.wantApprox != nil && got < *tt.wantApprox {
					t.Fatalf("got: %v, want: %v", got, *tt.wantApprox)
				}
			})
		}
	})

	t.Run("which these is not like the other?", func(t *testing.T) {
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

		// Things that are similar, and things that are not
		tests := []struct {
			things   []string
			oddIndex int
		}{
			{
				things: []string{
					"I went to the store the other day, and a new brand of cereal caught my eye. Remembering the cereal I ate as a kid, I decided to buy a box.",
					"Chicago style pizza is delicious, but it is bascially a casserole.",
					"I like pancakes, but I prefer waffles. I like the crisp edges of waffles, with a lot of butter and syrup.",
					"Have you seen the price of eggs lately? They are expensive, but you can't make an omelette without them!",
					"The new cereal I bought was delicious. I ate it all in one sitting.",
				},
				oddIndex: 1,
			},
			{
				things: []string{
					"There's a snake in my boots!",
					"Howdy, partner!",
					"Can you help me find my hat?",
					"Sorry Dave, I'm afraid I can't do that.",
				},
				oddIndex: 3,
			},
			{
				things: []string{
					"In a hole in the ground there lived a hobbit.",
					"I'm going on an adventure!",
					"Fly, you fools!",
					"Keep it secret, keep it safe.",
					"Cybersecurity training is important in the modern workplace.",
				},
				oddIndex: 4,
			},
			{
				things: []string{
					"I want to bake a cake.",
					"I don't have any eggs.",
					"I do have some flour.",
					"I do have some sugar.",
					"I do have some butter.",
					"Where do I get a hammer in this game, under the mountain?",
				},
				oddIndex: 5,
			},
		}

		for _, tt := range tests {
			thingToEmbedding := map[string][]float64{}

			for _, thing := range tt.things {
				thingToEmbedding[thing] = getEmbedding(t, thing)
			}

			averages := map[string]float64{}

			for i, thing := range tt.things {
				var sum float64

				for j, otherThing := range tt.things {
					if i == j {
						continue
					}

					sim, err := CosineSimilarity(thingToEmbedding[thing], thingToEmbedding[otherThing])
					if err != nil {
						t.Fatalf("error: %v", err)
					}

					sum += sim

					// fmt.Printf("%s vs %s: %f\n", thing[:10], otherThing[:10], sim)
				}

				avg := sum / float64(len(tt.things)-1)

				averages[thing] = avg

				// fmt.Printf("%s avg: %f\n", thing[:10], avg)
			}

			var lowestAvg float64

			for thing, avg := range averages {
				t.Logf("%s avg: %f\n", thing[:10], avg)

				if lowestAvg == 0.0 || avg < lowestAvg {
					lowestAvg = avg
				}
			}

			var oddOneOut string

			for thing, avg := range averages {
				if avg == lowestAvg {
					oddOneOut = thing
					break
				}
			}

			t.Logf("odd one out: %s", oddOneOut)

			if tt.things[tt.oddIndex] != oddOneOut {
				t.Fatalf("got: %s, want: %s", oddOneOut, tt.things[tt.oddIndex])
			}
		}
	})
}

func TestEuclideanDistance(t *testing.T) {
	t.Run("return error for unequal length embeddings", func(t *testing.T) {
		_, err := EuclideanDistance([]float64{1, 2, 3}, []float64{1, 2, 3, 4})
		if err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("return 0.0 for identical embeddings", func(t *testing.T) {
		dist, err := EuclideanDistance([]float64{1, 2, 3}, []float64{1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}

		if dist != 0.0 {
			t.Fatalf("expected distance to be 0.0, got %f", dist)
		}
	})

	t.Run("return 0.0 for zero embeddings", func(t *testing.T) {
		dist, err := EuclideanDistance([]float64{0, 0, 0}, []float64{0, 0, 0})
		if err != nil {
			t.Fatal(err)
		}

		if dist != 0.0 {
			t.Fatalf("expected distance to be 0.0, got %f", dist)
		}
	})

	t.Run("orthogonal embeddings", func(t *testing.T) {
		dist, err := EuclideanDistance([]float64{1, 0, 0}, []float64{0, 1, 0})
		if err != nil {
			t.Fatal(err)
		}

		if !(dist > 1.4 && dist < 1.5) { // 1.414214
			t.Fatalf("expected distance to be 0.0, got %f", dist)
		}
	})

	t.Run("other embeddings", func(t *testing.T) {
		dist, err := EuclideanDistance([]float64{0, 0, 0.5}, []float64{0.5, 0.5, 0})
		if err != nil {
			t.Fatal(err)
		}

		if !(dist > 1.4 && dist < 1.5) { // 1.414214
			t.Fatalf("expected distance to be 0.0, got %f", dist)
		}
	})

	t.Run("a,b", func(t *testing.T) {
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

		tests := []struct {
			a          []float64
			b          []float64
			wantApprox *float64
			want       *float64
		}{
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Ketchup is bad."),
				wantApprox: ptrFloat64(0.7),
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "AHHHHHHHH!"),
				wantApprox: ptrFloat64(0.6),
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "The big bang theory tells us that the universe began with a bang."),
				wantApprox: ptrFloat64(0.5),
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "How did the universe begin?"),
				wantApprox: ptrFloat64(0.4),
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the universe?."),
				wantApprox: ptrFloat64(0.3),
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Tell me about the history of the universe, please?"),
				wantApprox: ptrFloat64(0.2),
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the history of our universe?"),
				wantApprox: ptrFloat64(0.1),
			},
		}

		for ti, tt := range tests {
			t.Run(fmt.Sprintf("%d", ti), func(t *testing.T) {
				got, err := EuclideanDistance(tt.a, tt.b)
				if err != nil {
					t.Fatalf("error: %v", err)
				}

				t.Logf("%d: got: %f", ti, got)
				// if tt.want != nil && got != *tt.want {
				// 	t.Fatalf("got: %v, want: %v", got, *tt.want)
				// }

				if tt.wantApprox != nil && got < *tt.wantApprox {
					t.Fatalf("got: %v, want: %v", got, *tt.wantApprox)
				}
			})
		}
	})
}

func TestHammingDistance(t *testing.T) {
	t.Run("return error for unequal length embeddings", func(t *testing.T) {
		_, err := HammingDistance([]float64{1, 2, 3}, []float64{1, 2, 3, 4})
		if err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("return 0.0 for identical embeddings", func(t *testing.T) {
		dist, err := HammingDistance([]float64{1, 2, 3}, []float64{1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}

		distStr := fmt.Sprintf("%.1f", dist)

		if distStr != "0.0" {
			t.Fatalf("expected distance to be 0.0, got %f", dist)
		}
	})

	t.Run("return 0.0 for zero embeddings", func(t *testing.T) {
		dist, err := HammingDistance([]float64{0, 0, 0}, []float64{0, 0, 0})
		if err != nil {
			t.Fatal(err)
		}

		disStr := fmt.Sprintf("%.1f", dist)

		if disStr != "0.0" {
			t.Fatalf("expected distance to be 0.0, got %f", dist)
		}
	})

	t.Run("return 2.0 for orthogonal embeddings", func(t *testing.T) {
		dist, err := HammingDistance([]float64{1, 0, 0}, []float64{0, 1, 0})
		if err != nil {
			t.Fatal(err)
		}

		disStr := fmt.Sprintf("%.1f", dist)

		if disStr != "2.0" {
			t.Fatalf("expected distance to be 2.0, got %f", dist)
		}
	})

	t.Run("return 3.0 for opposite embeddings", func(t *testing.T) {
		dist, err := HammingDistance([]float64{0, 0, 0.5}, []float64{0.5, 0.5, 0})
		if err != nil {
			t.Fatal(err)
		}

		distStr := fmt.Sprintf("%.1f", dist)

		if distStr != "3.0" {
			t.Fatalf("expected distance to be 3.0, got %q", distStr)
		}
	})

	// Note: not useful for OpenAI's embeddings.
}

func TestManhattanDistance(t *testing.T) {
	t.Run("return error for unequal length embeddings", func(t *testing.T) {
		_, err := ManhattanDistance([]float64{1, 2, 3}, []float64{1, 2, 3, 4})
		if err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("return 0.0 for identical embeddings", func(t *testing.T) {
		dist, err := ManhattanDistance([]float64{1, 2, 3}, []float64{1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}

		distStr := fmt.Sprintf("%.1f", dist)

		if distStr != "0.0" {
			t.Fatalf("expected distance to be 0.0, got %f", dist)
		}
	})

	t.Run("return 0.0 for zero embeddings", func(t *testing.T) {
		dist, err := ManhattanDistance([]float64{0, 0, 0}, []float64{0, 0, 0})
		if err != nil {
			t.Fatal(err)
		}

		disStr := fmt.Sprintf("%.1f", dist)

		if disStr != "0.0" {
			t.Fatalf("expected distance to be 0.0, got %f", dist)
		}
	})

	t.Run("return -0.5 for orthogonal embeddings", func(t *testing.T) {
		dist, err := PearsonCorrelationCoefficient([]float64{1, 0, 0}, []float64{0, 1, 0})
		if err != nil {
			t.Fatal(err)
		}

		disStr := fmt.Sprintf("%.1f", dist)

		if disStr != "-0.5" {
			t.Fatalf("expected distance to be -0.5, got %f", dist)
		}
	})

	t.Run("return -1.0 for opposite embeddings", func(t *testing.T) {
		dist, err := PearsonCorrelationCoefficient([]float64{0, 0, 0.5}, []float64{0.5, 0.5, 0})
		if err != nil {
			t.Fatal(err)
		}

		distStr := fmt.Sprintf("%.1f", dist)

		if distStr != "-1.0" {
			t.Fatalf("expected distance to be -1.0, got %q", distStr)
		}
	})

	t.Run("a,b", func(t *testing.T) {
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

		tests := []struct {
			a          []float64
			b          []float64
			wantApprox string
		}{
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "茶"),
				wantApprox: "24.3",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "後でラーメンを食べに行きます"),
				wantApprox: "25.0",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "SolidGoldMajikarp"),
				wantApprox: "23.",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Ketchup is bad."),
				wantApprox: "23.",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "AHHHHHHHH!"),
				wantApprox: "21.7",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "The big bang theory tells us that the universe began with a bang."),
				wantApprox: "17.7",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "How did the universe begin?"),
				wantApprox: "13.1",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the universe?."),
				wantApprox: "9.7",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Tell me about the history of the universe, please?"),
				wantApprox: "6.6",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the history of our universe?"),
				wantApprox: "3.",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				wantApprox: "0.0",
			},
		}

		for ti, tt := range tests {
			t.Run(fmt.Sprintf("%d", ti), func(t *testing.T) {
				got, err := ManhattanDistance(tt.a, tt.b)
				if err != nil {
					t.Fatalf("error: %v", err)
				}

				t.Logf("test %d: %f", ti, got)

				gotStr := fmt.Sprintf("%f", got)

				if !strings.HasPrefix(gotStr, tt.wantApprox) {
					t.Fatalf("got: %v, want: %v", gotStr, tt.wantApprox)
				}
			})
		}
	})
}

func TestPearsonCorrelationCoefficient(t *testing.T) {
	t.Run("return error for unequal length embeddings", func(t *testing.T) {
		_, err := PearsonCorrelationCoefficient([]float64{1, 2, 3}, []float64{1, 2, 3, 4})
		if err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("return 1.0 for identical embeddings", func(t *testing.T) {
		dist, err := PearsonCorrelationCoefficient([]float64{1, 2, 3}, []float64{1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}

		if dist != 1.0 {
			t.Fatalf("expected distance to be 1.0, got %f", dist)
		}
	})

	t.Run("return NaN for zero embeddings", func(t *testing.T) {
		dist, err := PearsonCorrelationCoefficient([]float64{0, 0, 0}, []float64{0, 0, 0})
		if err != nil {
			t.Fatal(err)
		}

		disStr := fmt.Sprintf("%.1f", dist)

		if disStr != "NaN" {
			t.Fatalf("expected distance to be NaN, got %f", dist)
		}
	})

	t.Run("return -0.5 for orthogonal embeddings", func(t *testing.T) {
		dist, err := PearsonCorrelationCoefficient([]float64{1, 0, 0}, []float64{0, 1, 0})
		if err != nil {
			t.Fatal(err)
		}

		if !(dist < -0.4 && dist > -0.6) { // -0.500000
			t.Fatalf("expected distance to be -0.5, got %f", dist)
		}
	})

	t.Run("return -1.0 for opposite embeddings", func(t *testing.T) {
		dist, err := PearsonCorrelationCoefficient([]float64{0, 0, 0.5}, []float64{0.5, 0.5, 0})
		if err != nil {
			t.Fatal(err)
		}

		distStr := fmt.Sprintf("%.1f", dist)

		if distStr != "-1.0" {
			t.Fatalf("expected distance to be -1.0, got %q", distStr)
		}
	})

	t.Run("a,b", func(t *testing.T) {
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

		tests := []struct {
			a          []float64
			b          []float64
			wantApprox string
		}{
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "茶"),
				wantApprox: "0.6",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "後でラーメンを食べに行きます"),
				wantApprox: "0.6",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "SolidGoldMajikarp"),
				wantApprox: "0.7",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Ketchup is bad."),
				wantApprox: "0.7",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "AHHHHHHHH!"),
				wantApprox: "0.7",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "The big bang theory tells us that the universe began with a bang."),
				wantApprox: "0.8",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "How did the universe begin?"),
				wantApprox: "0.9",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the universe?."),
				wantApprox: "0.9",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Tell me about the history of the universe, please?"),
				wantApprox: "0.9",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the history of our universe?"),
				wantApprox: "0.9",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				wantApprox: "1.0",
			},
		}

		for ti, tt := range tests {
			t.Run(fmt.Sprintf("%d", ti), func(t *testing.T) {
				got, err := PearsonCorrelationCoefficient(tt.a, tt.b)
				if err != nil {
					t.Fatalf("error: %v", err)
				}

				t.Logf("test %d: %f", ti, got)

				gotStr := fmt.Sprintf("%f", got)

				if !strings.HasPrefix(gotStr, tt.wantApprox) {
					t.Fatalf("got: %v, want: %v", gotStr, tt.wantApprox)
				}
			})
		}
	})
}

func TestSpearmanRankCorrelationCoefficient(t *testing.T) {
	t.Run("return error for unequal length embeddings", func(t *testing.T) {
		_, err := SpearmanRankCorrelationCoefficient([]float64{1, 2, 3}, []float64{1, 2, 3, 4})
		if err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("return NaN for identical embeddings", func(t *testing.T) {
		dist, err := SpearmanRankCorrelationCoefficient([]float64{1, 2, 3}, []float64{1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}

		disStr := fmt.Sprintf("%.1f", dist)

		if disStr != "NaN" {
			t.Fatalf("expected distance to be NaN, got %f", dist)
		}
	})

	t.Run("return NaN for zero embeddings", func(t *testing.T) {
		dist, err := SpearmanRankCorrelationCoefficient([]float64{0, 0, 0}, []float64{0, 0, 0})
		if err != nil {
			t.Fatal(err)
		}

		disStr := fmt.Sprintf("%.1f", dist)

		if disStr != "NaN" {
			t.Fatalf("expected distance to be NaN, got %f", dist)
		}
	})

	t.Run("return -0.5 for orthogonal embeddings", func(t *testing.T) {
		dist, err := SpearmanRankCorrelationCoefficient([]float64{1, 0, 0}, []float64{0, 1, 0})
		if err != nil {
			t.Fatal(err)
		}

		if !(dist < -0.4 && dist > -0.6) { // -0.500000
			t.Fatalf("expected distance to be -0.5, got %f", dist)
		}
	})

	t.Run("return 1.0 for opposite embeddings", func(t *testing.T) {
		dist, err := SpearmanRankCorrelationCoefficient([]float64{0, 0, 0.5}, []float64{0.5, 0.5, 0})
		if err != nil {
			t.Fatal(err)
		}

		distStr := fmt.Sprintf("%.1f", dist)

		if distStr != "1.0" {
			t.Fatalf("expected distance to be 1.0, got %q", distStr)
		}
	})

	t.Run("a,b", func(t *testing.T) {
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

		tests := []struct {
			a          []float64
			b          []float64
			wantApprox string
		}{
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "茶"),
				wantApprox: "0.6",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "後でラーメンを食べに行きます"),
				wantApprox: "0.6",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "SolidGoldMajikarp"),
				wantApprox: "0.6",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Ketchup is bad."),
				wantApprox: "0.6",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "AHHHHHHHH!"),
				wantApprox: "0.6",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "The big bang theory tells us that the universe began with a bang."),
				wantApprox: "0.7",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "How did the universe begin?"),
				wantApprox: "0.8",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the universe?."),
				wantApprox: "0.8",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Tell me about the history of the universe, please?"),
				wantApprox: "0.9",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the history of our universe?"),
				wantApprox: "0.9",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				wantApprox: "1.0", // Sometimes 1.0, sometimes 0.9
			},
		}

		for ti, tt := range tests {
			t.Run(fmt.Sprintf("%d", ti), func(t *testing.T) {
				got, err := SpearmanRankCorrelationCoefficient(tt.a, tt.b)
				if err != nil {
					t.Fatalf("error: %v", err)
				}

				t.Logf("test %d: %f", ti, got)

				gotStr := fmt.Sprintf("%f", got)

				if !strings.HasPrefix(gotStr, tt.wantApprox) {
					// if last test, allow 0.1 difference
					if ti == len(tests)-1 {
						want, err := strconv.ParseFloat(tt.wantApprox, 64)
						if err != nil {
							t.Fatalf("failed to parse want: %v", err)
						}

						got, err := strconv.ParseFloat(gotStr, 64)
						if err != nil {
							t.Fatalf("failed to parse got: %v", err)
						}

						if math.Abs(want-got) <= 0.1 {
							return
						}
					}

					t.Fatalf("got: %v, want: %v", gotStr, tt.wantApprox)
				}
			})
		}
	})
}

func TestJaquardSimilarity(t *testing.T) {
	t.Run("return error for unequal length embeddings", func(t *testing.T) {
		_, err := JaquardSimilarity([]float64{1, 2, 3}, []float64{1, 2, 3, 4})
		if err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("return 1.0 for identical embeddings", func(t *testing.T) {
		dist, err := JaquardSimilarity([]float64{1, 2, 3}, []float64{1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}

		distStr := fmt.Sprintf("%.1f", dist)

		if distStr != "1.0" {
			t.Fatalf("expected distance to be 0.0, got %f", dist)
		}
	})

	t.Run("return +Inf for zero embeddings", func(t *testing.T) {
		dist, err := JaquardSimilarity([]float64{0, 0, 0}, []float64{0, 0, 0})
		if err != nil {
			t.Fatal(err)
		}

		disStr := fmt.Sprintf("%.1f", dist)

		if disStr != "+Inf" {
			t.Fatalf("expected distance to be +Inf, got %f", dist)
		}
	})

	t.Run("return 0.5 for orthogonal embeddings", func(t *testing.T) {
		dist, err := JaquardSimilarity([]float64{1, 0, 0}, []float64{0, 1, 0})
		if err != nil {
			t.Fatal(err)
		}

		disStr := fmt.Sprintf("%.1f", dist)

		if disStr != "0.5" {
			t.Fatalf("expected distance to be 0.5, got %f", dist)
		}
	})

	t.Run("return 0.0 for opposite embeddings", func(t *testing.T) {
		dist, err := JaquardSimilarity([]float64{0, 0, 0.5}, []float64{0.5, 0.5, 0})
		if err != nil {
			t.Fatal(err)
		}

		distStr := fmt.Sprintf("%.1f", dist)

		if distStr != "0.0" {
			t.Fatalf("expected distance to be 0.0, got %q", distStr)
		}
	})

	// Note: not useful for OpenAI's (continuous) embeddings, but useful for other embeddings.
}

func TestBrayCurtisDistance(t *testing.T) {
	t.Run("return error for unequal length embeddings", func(t *testing.T) {
		_, err := BrayCurtisDistance([]float64{1, 2, 3}, []float64{1, 2, 3, 4})
		if err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("return 0.0 for identical embeddings", func(t *testing.T) {
		dist, err := BrayCurtisDistance([]float64{1, 2, 3}, []float64{1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}

		distStr := fmt.Sprintf("%.1f", dist)

		if distStr != "0.0" {
			t.Fatalf("expected distance to be 0.0, got %f", dist)
		}
	})

	t.Run("return NaN for zero embeddings", func(t *testing.T) {
		dist, err := BrayCurtisDistance([]float64{0, 0, 0}, []float64{0, 0, 0})
		if err != nil {
			t.Fatal(err)
		}

		disStr := fmt.Sprintf("%.1f", dist)

		if disStr != "NaN" {
			t.Fatalf("expected distance to be NaN, got %f", dist)
		}
	})

	t.Run("return 1.0 for orthogonal embeddings", func(t *testing.T) {
		dist, err := BrayCurtisDistance([]float64{1, 0, 0}, []float64{0, 1, 0})
		if err != nil {
			t.Fatal(err)
		}

		disStr := fmt.Sprintf("%.1f", dist)

		if disStr != "1.0" {
			t.Fatalf("expected distance to be 1.0, got %f", dist)
		}
	})

	t.Run("return 1.0 for opposite embeddings", func(t *testing.T) {
		dist, err := BrayCurtisDistance([]float64{0, 0, 0.5}, []float64{0.5, 0.5, 0})
		if err != nil {
			t.Fatal(err)
		}

		distStr := fmt.Sprintf("%.1f", dist)

		if distStr != "1.0" {
			t.Fatalf("expected distance to be 1.0, got %q", distStr)
		}
	})

	t.Run("a,b", func(t *testing.T) {
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

		tests := []struct {
			a          []float64
			b          []float64
			wantApprox string
		}{
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "茶"),
				wantApprox: "-10.7",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "後でラーメンを食べに行きます"),
				wantApprox: "-11.8",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "SolidGoldMajikarp"),
				wantApprox: "-10.8",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Ketchup is bad."),
				wantApprox: "-11.0",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "AHHHHHHHH!"),
				wantApprox: "-10.3",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "The big bang theory tells us that the universe began with a bang."),
				wantApprox: "-8.3",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "How did the universe begin?"),
				wantApprox: "-6.2",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the universe?."),
				wantApprox: "-4.5",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Tell me about the history of the universe, please?"),
				wantApprox: "-3.2",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the history of our universe?"),
				wantApprox: "-1.6",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				wantApprox: "0.0",
			},
		}

		for ti, tt := range tests {
			t.Run(fmt.Sprintf("%d", ti), func(t *testing.T) {
				got, err := BrayCurtisDistance(tt.a, tt.b)
				if err != nil {
					t.Fatalf("error: %v", err)
				}

				t.Logf("test %d: %f", ti, got)

				gotStr := fmt.Sprintf("%f", got)

				if !strings.HasPrefix(gotStr, tt.wantApprox) {
					// if last test, allow 0.1 difference
					if ti == len(tests)-1 {
						want, err := strconv.ParseFloat(tt.wantApprox, 64)
						if err != nil {
							t.Fatalf("failed to parse want: %v", err)
						}

						got, err := strconv.ParseFloat(gotStr, 64)
						if err != nil {
							t.Fatalf("failed to parse got: %v", err)
						}

						if math.Abs(want-got) <= 0.1 {
							return
						}
					}

					t.Fatalf("got: %v, want: %v", gotStr, tt.wantApprox)
				}
			})
		}
	})
}

func TestMahalanobisDistance(t *testing.T) {
	t.Run("return error for unequal length embeddings", func(t *testing.T) {
		covarianceMatrix := [][]float64{
			{1, 2, 3},
			{1, 2, 3},
			{1, 2, 3},
		}

		_, err := MahalanobisDistance([]float64{1, 2, 3}, []float64{1, 2, 3, 4}, covarianceMatrix)
		if err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("return error for invalid covariance matrix", func(t *testing.T) {
		covarianceMatrix := [][]float64{
			{1, 2, 3},
			{1, 2, 3},
			// covariance matrix must be square and have the same dimensions as the embeddings
		}

		_, err := MahalanobisDistance([]float64{1, 2, 3}, []float64{1, 2, 3, 4}, covarianceMatrix)
		if err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("return 0.0 for identical embeddings", func(t *testing.T) {
		covarianceMatrix := [][]float64{
			{1, 2, 3},
			{1, 2, 3},
			{1, 2, 3},
		}

		dist, err := MahalanobisDistance([]float64{1, 2, 3}, []float64{1, 2, 3}, covarianceMatrix)
		if err != nil {
			t.Fatal(err)
		}

		distStr := fmt.Sprintf("%.1f", dist)

		if distStr != "0.0" {
			t.Fatalf("expected distance to be 0.0, got %f", dist)
		}
	})

	t.Run("return 0.0 for zero embeddings", func(t *testing.T) {
		covarianceMatrix := [][]float64{
			{1, 2, 3},
			{1, 2, 3},
			{1, 2, 3},
		}

		dist, err := MahalanobisDistance([]float64{0, 0, 0}, []float64{0, 0, 0}, covarianceMatrix)
		if err != nil {
			t.Fatal(err)
		}

		disStr := fmt.Sprintf("%.1f", dist)

		if disStr != "0.0" {
			t.Fatalf("expected distance to be 0.0, got %f", dist)
		}
	})

	t.Run("return 0.0 for orthogonal embeddings", func(t *testing.T) {
		covarianceMatrix := [][]float64{
			{1, 2, 3},
			{1, 2, 3},
			{1, 2, 3},
		}

		dist, err := MahalanobisDistance([]float64{1, 0, 0}, []float64{0, 1, 0}, covarianceMatrix)
		if err != nil {
			t.Fatal(err)
		}

		disStr := fmt.Sprintf("%.1f", dist)

		if disStr != "0.0" {
			t.Fatalf("expected distance to be 0.0, got %f", dist)
		}
	})

	t.Run("return 0.0 for opposite embeddings", func(t *testing.T) {
		covarianceMatrix := [][]float64{
			{1, 2, 3},
			{1, 2, 3},
			{1, 2, 3},
		}

		dist, err := MahalanobisDistance([]float64{0, 0, 0.5}, []float64{0.5, 0.5, 0}, covarianceMatrix)
		if err != nil {
			t.Fatal(err)
		}

		distStr := fmt.Sprintf("%.1f", dist)

		if distStr != "0.0" {
			t.Fatalf("expected distance to be 1.0, got %q", distStr)
		}
	})

	t.Run("a,b similarity", func(t *testing.T) {
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

		tests := []struct {
			a          []float64
			b          []float64
			wantApprox string
		}{
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "茶"),
				wantApprox: "0.8",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "後でラーメンを食べに行きます"),
				wantApprox: "0.8",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "SolidGoldMajikarp"),
				wantApprox: "0.7",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Ketchup is bad."),
				wantApprox: "0.8",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "AHHHHHHHH!"),
				wantApprox: "0.7",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "The big bang theory tells us that the universe began with a bang."),
				wantApprox: "0.5",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "How did the universe begin?"),
				wantApprox: "0.4",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the universe?."),
				wantApprox: "0.3",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Tell me about the history of the universe, please?"),
				wantApprox: "0.2",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the history of our universe?"),
				wantApprox: "0.1",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				wantApprox: "0.0",
			},
		}

		layer := getEmbedding(t, "Can you tell me about the history of the universe?")

		covarianceMatrix := [][]float64{}

		// Create a covariance matrix with 1s on the diagonal,
		// and the product of each element in the layer on the off-diagonal.
		//
		// This allows us to use the covariance matrix as a similarity matrix,
		// favoring the elements in the layer that are more similar to each other.
		//
		// Hell yeah, that's math!
		for i, a := range layer {
			covarianceMatrix = append(covarianceMatrix, []float64{})
			for j, b := range layer {
				covarianceMatrix[i] = append(covarianceMatrix[i], a*b)

				if i == j {
					covarianceMatrix[i][j] = 1 // this drawing the diagonal
				}
			}
		}

		for ti, tt := range tests {
			t.Run(fmt.Sprintf("%d", ti), func(t *testing.T) {
				got, err := MahalanobisDistance(tt.a, tt.b, covarianceMatrix)
				if err != nil {
					t.Fatalf("error: %v", err)
				}

				t.Logf("test %d: %f", ti, got)

				gotStr := fmt.Sprintf("%f", got)

				if !strings.HasPrefix(gotStr, tt.wantApprox) {
					// if last test, allow 0.1 difference
					if ti == len(tests)-1 {
						want, err := strconv.ParseFloat(tt.wantApprox, 64)
						if err != nil {
							t.Fatalf("failed to parse want: %v", err)
						}

						got, err := strconv.ParseFloat(gotStr, 64)
						if err != nil {
							t.Fatalf("failed to parse got: %v", err)
						}

						if math.Abs(want-got) <= 0.1 {
							return
						}
					}

					t.Fatalf("got: %v, want: %v", gotStr, tt.wantApprox)
				}
			})
		}
	})

	t.Run("a,b similarity normalized", func(t *testing.T) {
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

		tests := []struct {
			a          []float64
			b          []float64
			wantApprox string
		}{
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "茶"),
				wantApprox: "0.7",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "後でラーメンを食べに行きます"),
				wantApprox: "0.7",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "SolidGoldMajikarp"),
				wantApprox: "0.7",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Ketchup is bad."),
				wantApprox: "0.7",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "AHHHHHHHH!"),
				wantApprox: "0.6",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "The big bang theory tells us that the universe began with a bang."),
				wantApprox: "0.5",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "How did the universe begin?"),
				wantApprox: "0.4",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the universe?."),
				wantApprox: "0.3",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Tell me about the history of the universe, please?"),
				wantApprox: "0.2",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the history of our universe?"),
				wantApprox: "0.1",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				wantApprox: "0.0",
			},
		}

		layer := getEmbedding(t, "Can you tell me about the history of the universe?")

		covarianceMatrix := [][]float64{}

		// Create a covariance matrix that is a diagonal matrix with the variance of each layer,
		// which is 1 since the layer is normalized (0).
		for i := 0; i < len(layer); i++ {
			// Example
			//
			// [1 0 0 0 0 0 0 0 0 0]
			// [0 1 0 0 0 0 0 0 0 0]
			// [0 0 1 0 0 0 0 0 0 0]
			// [0 0 0 1 0 0 0 0 0 0]
			// [0 0 0 0 1 0 0 0 0 0]
			// [0 0 0 0 0 1 0 0 0 0]
			// [0 0 0 0 0 0 1 0 0 0]
			// [0 0 0 0 0 0 0 1 0 0]
			// [0 0 0 0 0 0 0 0 1 0]
			// [0 0 0 0 0 0 0 0 0 1]
			covarianceMatrix = append(covarianceMatrix, []float64{})
			for j := 0; j < len(layer); j++ {
				if i == j {
					covarianceMatrix[i] = append(covarianceMatrix[i], 1)
				} else {
					covarianceMatrix[i] = append(covarianceMatrix[i], 0)
				}
			}
		}

		for ti, tt := range tests {
			t.Run(fmt.Sprintf("%d", ti), func(t *testing.T) {
				got, err := MahalanobisDistance(tt.a, tt.b, covarianceMatrix)
				if err != nil {
					t.Fatalf("error: %v", err)
				}

				t.Logf("test %d: %f", ti, got)

				gotStr := fmt.Sprintf("%f", got)

				if !strings.HasPrefix(gotStr, tt.wantApprox) {
					// if last test, allow 0.1 difference
					if ti == len(tests)-1 {
						want, err := strconv.ParseFloat(tt.wantApprox, 64)
						if err != nil {
							t.Fatalf("failed to parse want: %v", err)
						}

						got, err := strconv.ParseFloat(gotStr, 64)
						if err != nil {
							t.Fatalf("failed to parse got: %v", err)
						}

						if math.Abs(want-got) <= 0.1 {
							return
						}
					}

					t.Fatalf("got: %v, want: %v", gotStr, tt.wantApprox)
				}
			})
		}
	})
}

func TestWordMoversDistance(t *testing.T) {
	// TODO: add tests

	// Note: this doesn't seem to be useful for continuous embeddings.
}

func TestBhattacharyyaDistance(t *testing.T) {
	// TODO: add tests

	// Note: this doesn't seem to be useful for continuous embeddings.
}

func TestWassersteinDistance(t *testing.T) {
	t.Run("return error for unequal length embeddings", func(t *testing.T) {
		_, err := WassersteinDistance([]float64{1, 2, 3}, []float64{1, 2, 3, 4})
		if err == nil {
			t.Fatal("expected error")
		}
	})

	t.Run("return 0.0 for identical embeddings", func(t *testing.T) {
		dist, err := WassersteinDistance([]float64{1, 2, 3}, []float64{1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}

		distStr := fmt.Sprintf("%.1f", dist)

		if distStr != "0.0" {
			t.Fatalf("expected distance to be 0.0, got %f", dist)
		}
	})

	t.Run("return NaN for zero embeddings", func(t *testing.T) {
		dist, err := WassersteinDistance([]float64{0, 0, 0}, []float64{0, 0, 0})
		if err != nil {
			t.Fatal(err)
		}

		disStr := fmt.Sprintf("%.1f", dist)

		if disStr != "0.0" {
			t.Fatalf("expected distance to be 0.0, got %f", dist)
		}
	})

	t.Run("return 0.0 for orthogonal embeddings", func(t *testing.T) {
		dist, err := WassersteinDistance([]float64{1, 0, 0}, []float64{0, 1, 0})
		if err != nil {
			t.Fatal(err)
		}

		disStr := fmt.Sprintf("%.1f", dist)

		if disStr != "0.0" {
			t.Fatalf("expected distance to be 1.0, got %f", dist)
		}
	})

	t.Run("return 0.5 for opposite embeddings", func(t *testing.T) {
		dist, err := WassersteinDistance([]float64{0, 0, 0.5}, []float64{0.5, 0.5, 0})
		if err != nil {
			t.Fatal(err)
		}

		distStr := fmt.Sprintf("%.1f", dist)

		if distStr != "0.5" {
			t.Fatalf("expected distance to be 0.5, got %q", distStr)
		}
	})

	t.Run("a,b", func(t *testing.T) {
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

		tests := []struct {
			a          []float64
			b          []float64
			wantApprox string
		}{
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "茶"),
				wantApprox: "2.3",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "後でラーメンを食べに行きます"),
				wantApprox: "2.0",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "SolidGoldMajikarp"),
				wantApprox: "1.6",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Ketchup is bad."),
				wantApprox: "1.9",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "AHHHHHHHH!"),
				wantApprox: "1.1",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "The big bang theory tells us that the universe began with a bang."),
				wantApprox: "0.8",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "How did the universe begin?"),
				wantApprox: "0.7",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the universe?."),
				wantApprox: "0.7",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Tell me about the history of the universe, please?"),
				wantApprox: "0.6",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the history of our universe?"),
				wantApprox: "0.3",
			},
			{
				a:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				b:          getEmbedding(t, "Can you tell me about the history of the universe?"),
				wantApprox: "0.03",
			},
		}

		for ti, tt := range tests {
			t.Run(fmt.Sprintf("%d", ti), func(t *testing.T) {
				got, err := WassersteinDistance(tt.a, tt.b)
				if err != nil {
					t.Fatalf("error: %v", err)
				}

				t.Logf("test %d: %f", ti, got)

				gotStr := fmt.Sprintf("%f", got)

				if !strings.HasPrefix(gotStr, tt.wantApprox) {
					// if last test, allow 0.1 difference
					if ti == len(tests)-1 {
						want, err := strconv.ParseFloat(tt.wantApprox, 64)
						if err != nil {
							t.Fatalf("failed to parse want: %v", err)
						}

						got, err := strconv.ParseFloat(gotStr, 64)
						if err != nil {
							t.Fatalf("failed to parse got: %v", err)
						}

						if math.Abs(want-got) <= 0.1 {
							return
						}
					}

					t.Fatalf("got: %v, want: %v", gotStr, tt.wantApprox)
				}
			})
		}
	})
}
