package main

import (
	"io"
	"os"
	"encoding/csv"
	"fmt"
	"github.com/Rhymen/ml/dense"
	"math"
	"strconv"
	"sort"
)

type mnist struct {
	target int
	pxls dense.Matrix
}

func readFile(path string) ([]mnist, error) {
	testFile, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer testFile.Close()

	r := csv.NewReader(testFile)
	ret := make([]mnist, 0)

	for {
		rec, err := r.Read()
		if err != nil {
			if err == io.EOF {
				break
			}

			return nil, err
		}

		target, err := strconv.Atoi(rec[0])
		if err != nil {
			return nil, err
		}

		pxls, err := dense.FromList(rec[1:])
		if err != nil {
			return nil, err
		}

		ret = append(ret, mnist{target, pxls})
	}

	return ret, nil
}

func distance(sample mnist, data mnist) float64 {
	return math.Sqrt(sample.pxls.Subtract(data.pxls).Apply(func(i float64) float64 { return i * i }).Sum())
}

func query(sample mnist, data []mnist, k int) int {
	type result struct {
		value int
		distance float64
	}

	distances := make([]result, k)

	for i := range distances {
		distances[i] = result{-1, math.MaxFloat64}
	}

	for i := range data {
		d := distance(sample, data[i])

		if d < distances[0].distance {
			distances[0].distance = d
			distances[0].value = data[i].target
		}

		sort.Slice(distances, func(i, j int) bool { return distances[i].distance > distances[j].distance })
	}

	prop := make([]float64, 10)
	for i := range distances {
		prop[distances[i].value] += 1.0 / float64(k)
	}

	max := 0
	for i, l := 1, len(prop); i < l; i++ {
		if prop[max] < prop[i] {
			max = i
		}
	}

	return max
}

func main() {
	data, err := readFile("./mnist_dataset/mnist_train_100.csv")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading data: %v", err)
	}

	samples, err := readFile("./mnist_dataset/mnist_test_10.csv")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading testData: %v", err)
	}

	hit := 0.0
	for _, s := range samples {
		result := query(s, data, 5)
		if result == s.target {
			hit++
		}
	}

	fmt.Printf("hit rate: %v", hit / float64(len(samples)))
}
