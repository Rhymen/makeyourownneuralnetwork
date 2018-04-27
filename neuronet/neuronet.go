package neuronet

import (
	"github.com/Rhymen/ml/dense"
	"math"
	"bufio"
	"encoding/csv"
	"strconv"
	"os"
	"fmt"
	"encoding/gob"
)

type neuralNetwork struct {
	iNodes             int
	hNodes             int
	oNodes             int
	lr                 float64
	wih                dense.Matrix
	who                dense.Matrix
	activationFunction func(float64) float64
}

type neuralNetworkGob struct {
	INodes             int
	HNodes             int
	ONodes             int
	Lr                 float64
	Wih                dense.Matrix
	Who                dense.Matrix
	ActivationFunction func(float64) float64
}

func New(iNodes, hNodes, oNodes int, lr float64) *neuralNetwork {
	n := &neuralNetwork{
		iNodes,
		hNodes,
		oNodes,
		lr,
		dense.Random(hNodes, iNodes).SubtractScalar(0.5), //TODO: check optional code
		dense.Random(oNodes, hNodes).SubtractScalar(0.5),
		func(sum float64) float64 { return 1.0/(1.0 + math.Exp(-sum)) },
	}

	return n
}

func FromCheckpoint(filePath string) (*neuralNetwork, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	n := &neuralNetworkGob{}
	err = decoder.Decode(n)

	return &neuralNetwork{
		n.INodes,
		n.HNodes,
		n.ONodes,
		n.Lr,
		n.Wih,
		n.Who,
		func(sum float64) float64 { return 1.0/(1.0 + math.Exp(-sum)) },
	}, nil
}

func (n *neuralNetwork) CreateCheckpoint(filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	encoder.Encode(neuralNetworkGob{
		n.iNodes,
		n.hNodes,
		n.oNodes,
		n.lr,
		n.wih,
		n.who,
		n.activationFunction,
	})

	return nil
}

func (n *neuralNetwork) Train(input, target dense.Matrix) {
	hiddenInput := n.wih.Multiply(input)
	hiddenOutput := hiddenInput.Apply(n.activationFunction)

	finalInput := n.who.Multiply(hiddenOutput)
	finalOutput := finalInput.Apply(n.activationFunction)

	outputErrors := target.Subtract(finalOutput)
	hiddenErrors := n.who.Transpose().Multiply(outputErrors)

	// ((outputErrors * finalOutputs * (-1 * finalOutputs + 1)) * target.T) * lr
	t1 := outputErrors.MultiplyComponent(finalOutput)
	t2 := finalOutput.MultiplyScalar(-1).AddScalar(1)
	t3 := t1.MultiplyComponent(t2)
	t4 := t3.Multiply(hiddenOutput.Transpose())
	t5 := t4.MultiplyScalar(n.lr)
	n.who = n.who.Add(t5)

	// ((hiddenErrors * hiddenOuputs * (-1 * hiddenOuputs + 1)) * inputs.T) * lr
	n.wih = n.wih.Add(hiddenErrors.MultiplyComponent(hiddenOutput).MultiplyComponent(hiddenOutput.MultiplyScalar(-1).AddScalar(1)).Multiply(input.Transpose()).MultiplyScalar(n.lr))
}

func (n *neuralNetwork) Query(input dense.Matrix) dense.Matrix {
	hiddenInput := n.wih.Multiply(input)
	hiddenOutput := hiddenInput.Apply(n.activationFunction)

	finalInput := n.who.Multiply(hiddenOutput)
	finalOutput := finalInput.Apply(n.activationFunction)

	return finalOutput
}

func buildTargetVector(target int) dense.Matrix {
	t := dense.Zeros(10, 1).AddScalar(0.01)
	t[target][0] = 0.99
	return t
}

func (n *neuralNetwork) TestNetwork(path string) (float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, err
	}

	reader := bufio.NewReader(file)

	r := csv.NewReader(reader)
	data, err := r.ReadAll()
	if err != nil {
		return 0, err
	}

	scorecard := make([]int, len(data))
	res := 0

	for j := range data {
		target, err := strconv.Atoi(data[j][0])
		if err != nil {
			return 0, err
		}

		pxl, err := dense.FromList(data[j][1:])
		if err != nil {
			return 0, err
		}

		pxl = pxl.MultiplyScalar(0.99 / 255.0).AddScalar(0.01)
		result := n.Query(pxl)

		var max int
		for i := range result {
			if result[max][0] < result[i][0] {
				max = i
			}
		}

		if max == target {
			scorecard[j] = 1
			res++
		} else {
			scorecard[j] = 0
			fmt.Printf("{line: %v, target: %v, found: %v},\n", j, target, max)
		}
	}

	return float64(res) / float64(len(scorecard)), nil
}

func (n *neuralNetwork) TrainNetwork(path string, epochs int) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}

	reader := bufio.NewReader(file)

	r := csv.NewReader(reader)
	data, err := r.ReadAll()
	if err != nil {
		return err
	}

	for i := 0; i < epochs; i++ {
		for j := range data {
			t, err := strconv.Atoi(data[j][0])
			if err != nil {
				return err
			}

			target := buildTargetVector(t)
			pxl, err := dense.FromList(data[j][1:])
			if err != nil {
				return err
			}

			pxl = pxl.MultiplyScalar(0.99 / 255.0).AddScalar(0.01)
			n.Train(pxl, target)
		}
	}

	return nil
}


