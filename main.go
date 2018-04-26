package main

import (
	"github.com/Rhymen/ml/neuronet"
	"fmt"
	"os"
)

func main() {
	n := neuronet.New(784, 100, 10, 0.3)

	err := n.TrainNetwork("./mnist_dataset/mnist_train.csv", 1)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading training data: %v\n", err)
		os.Exit(1)
	}
	fmt.Print("finished training\n")

	rate, err := n.TestNetwork("./mnist_dataset/mnist_test.csv")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading training data: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("finished testing: %v\n", rate)
}
