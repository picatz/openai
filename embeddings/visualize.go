package embeddings

import (
	"image"
	"image/color"
	"math"
)

// float64ToUint8 converts a float64 to a uint8. It will clamp the value
// between 0 and 255.
func float64ToUint8(f float64) uint8 {
	return uint8(math.Max(math.Min(f*255, 255), 0))
}

// Visualize returns an in-memory RGB image of the given embeddings which
// can be written to disk or used in other ways. The pointStroke determines
// the size of the points in the image. The imgWidth and imgHeight determine
// the size of the image in pixels.
//
// This can be useful for visualizing the embeddings in a 2D space to get
// a sense of how they are distributed.
func Visualize(embeddings [][]float64, pointStroke, imgWidth, imgHeight int) (*image.RGBA, error) {
	// Find the min and max values for each embedding.
	minX, minY, maxX, maxY := math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64, -math.MaxFloat64

	for _, embedding := range embeddings {
		for _, e := range embedding {
			if e < minX {
				minX = e
			}
			if e > maxX {
				maxX = e
			}
			if e < minY {
				minY = e
			}
			if e > maxY {
				maxY = e
			}
		}
	}

	// Create a new image with the given width and height.
	img := image.NewRGBA(image.Rect(0, 0, imgWidth, imgHeight))

	// Set background to black.
	for x := 0; x < imgWidth; x++ {
		for y := 0; y < imgHeight; y++ {
			img.Set(x, y, color.RGBA{R: 0, G: 0, B: 0, A: 255})
		}
	}

	// Color each point.
	for embeddingIndex, embedding := range embeddings {
		for _, e := range embedding {
			// Get the x and y coordinates for the embedding.
			x := int((e - minX) * float64(imgWidth-1) / (maxX - minX))
			y := int((e - minY) * float64(imgHeight-1) / (maxY - minY))

			// Set the color of the point based on the embedding value.
			pointColor := color.RGBA{
				R: float64ToUint8(math.Abs(e / maxX)),
				G: float64ToUint8(math.Abs(e / maxY)),
				B: float64ToUint8(math.Abs(float64(x*y) / float64((imgWidth-1)*(imgHeight-1)))),
				A: 255,
			}

			// Adjust the color based on the embedding index so that
			// the points are easier to distinguish.
			if len(embeddings) > 1 {
				// TODO: make this more robust.
				switch embeddingIndex % 3 {
				case 0: // Red
					pointColor.R = 255

					// TODO: make this more robust.
					// pointColor.B = 0
					// pointColor.G = 0
				case 1: // Green
					pointColor.G = 255

					// TODO: make this more robust.
					// pointColor.B = 0
					// pointColor.R = 0
				case 2: // Blue
					pointColor.B = 255

					// TODO: make this more robust.
					// pointColor.G = 0
					// pointColor.R = 0
				}
			}

			// Add the point to the image.
			img.Set(x, y, pointColor)

			// Make the point a little bigger, easier to see.
			if pointStroke >= 1 {
				for i := -1; i < pointStroke; i++ {
					for j := -1; j < pointStroke; j++ {
						img.Set(x+i, y+j, pointColor)
					}
				}
			}
		}
	}

	return img, nil
}
